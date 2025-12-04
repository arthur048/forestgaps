"""Training optimization utilities.

Conforme "Audit du workflow PyTorch": gradient clipping, AMP, and other
training optimizations.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Iterator
from contextlib import contextmanager


class GradientClipper:
    """Gradient clipping utility.

    Supports both value clipping and norm clipping.
    """

    def __init__(
        self,
        clip_value: Optional[float] = None,
        clip_norm: Optional[float] = None
    ):
        """Initialize gradient clipper.

        Args:
            clip_value: Clip gradients by value (None = disabled)
            clip_norm: Clip gradients by norm (None = disabled)
        """
        self.clip_value = clip_value
        self.clip_norm = clip_norm

    def clip(self, parameters: Iterator[torch.nn.Parameter]) -> Optional[float]:
        """Clip gradients.

        Args:
            parameters: Model parameters

        Returns:
            Total norm if norm clipping is used, else None
        """
        total_norm = None

        # Clip by value
        if self.clip_value is not None:
            torch.nn.utils.clip_grad_value_(parameters, self.clip_value)

        # Clip by norm
        if self.clip_norm is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.clip_norm)

        return total_norm

    def __call__(self, parameters: Iterator[torch.nn.Parameter]) -> Optional[float]:
        """Call clip() directly."""
        return self.clip(parameters)


class AMPManager:
    """Automatic Mixed Precision (AMP) manager.

    Conforme Document 2 "Audit du workflow PyTorch": use AMP for training speedup.
    """

    def __init__(self, enabled: bool = True, device: str = "cuda"):
        """Initialize AMP manager.

        Args:
            enabled: Enable AMP
            device: Device type ('cuda' or 'cpu')
        """
        self.enabled = enabled and device == "cuda" and torch.cuda.is_available()
        self.scaler = GradScaler() if self.enabled else None

    @contextmanager
    def autocast_context(self):
        """Context manager for autocast.

        Usage:
            with amp_manager.autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        """
        if self.enabled:
            with autocast():
                yield
        else:
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass.

        Args:
            loss: Loss tensor

        Returns:
            Scaled loss if AMP is enabled, else original loss
        """
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        """Optimizer step with gradient unscaling.

        Args:
            optimizer: Optimizer instance
        """
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before gradient clipping.

        Args:
            optimizer: Optimizer instance
        """
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        if self.enabled and self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        if self.enabled and self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)


class GradientAccumulator:
    """Gradient accumulation utility.

    Accumulates gradients over multiple batches before optimizer step.
    """

    def __init__(self, accumulate_steps: int = 1):
        """Initialize gradient accumulator.

        Args:
            accumulate_steps: Number of batches to accumulate
        """
        self.accumulate_steps = max(1, accumulate_steps)
        self.current_step = 0

    def should_step(self) -> bool:
        """Check if optimizer should step.

        Returns:
            True if we should call optimizer.step()
        """
        self.current_step += 1
        should = (self.current_step % self.accumulate_steps == 0)
        return should

    def reset(self):
        """Reset accumulation counter."""
        self.current_step = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by accumulation steps.

        Args:
            loss: Loss tensor

        Returns:
            Scaled loss
        """
        return loss / self.accumulate_steps


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """Enable gradient checkpointing for memory optimization.

    Args:
        model: PyTorch model

    Returns:
        Model with gradient checkpointing enabled
    """
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        # Try to enable for common model types
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True

    return model


def compile_model(
    model: nn.Module,
    mode: str = "default",
    backend: Optional[str] = None
) -> nn.Module:
    """Compile model with torch.compile() (PyTorch 2.0+).

    Args:
        model: PyTorch model
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        backend: Compilation backend (None = default)

    Returns:
        Compiled model

    Note:
        Requires PyTorch 2.0+. Falls back to original model if unavailable.
    """
    if hasattr(torch, "compile"):
        try:
            compiled_model = torch.compile(
                model,
                mode=mode,
                backend=backend
            )
            return compiled_model
        except Exception as e:
            print(f"Warning: torch.compile() failed: {e}")
            print("Falling back to eager mode")
            return model
    else:
        print("Warning: torch.compile() not available (requires PyTorch 2.0+)")
        return model


class TrainingOptimizer:
    """Unified training optimization manager.

    Combines gradient clipping, AMP, and gradient accumulation.
    """

    def __init__(
        self,
        gradient_clip_value: Optional[float] = None,
        gradient_clip_norm: Optional[float] = None,
        use_amp: bool = True,
        accumulate_grad_batches: int = 1,
        device: str = "cuda"
    ):
        """Initialize training optimizer.

        Args:
            gradient_clip_value: Gradient value clipping
            gradient_clip_norm: Gradient norm clipping
            use_amp: Use Automatic Mixed Precision
            accumulate_grad_batches: Gradient accumulation steps
            device: Device type
        """
        self.gradient_clipper = GradientClipper(
            clip_value=gradient_clip_value,
            clip_norm=gradient_clip_norm
        )
        self.amp_manager = AMPManager(enabled=use_amp, device=device)
        self.accumulator = GradientAccumulator(accumulate_steps=accumulate_grad_batches)

    @contextmanager
    def forward_context(self):
        """Context for forward pass with AMP."""
        with self.amp_manager.autocast_context():
            yield

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        model_parameters: Iterator[torch.nn.Parameter]
    ) -> Dict[str, Any]:
        """Perform backward pass with all optimizations.

        Args:
            loss: Loss tensor
            optimizer: Optimizer instance
            model_parameters: Model parameters for gradient clipping

        Returns:
            Dictionary with step info (grad_norm, etc.)
        """
        info = {}

        # Scale loss for gradient accumulation
        loss = self.accumulator.scale_loss(loss)

        # Backward pass (with AMP scaling if enabled)
        if self.amp_manager.enabled:
            self.amp_manager.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Check if we should step
        if self.accumulator.should_step():
            # Unscale gradients before clipping
            if self.amp_manager.enabled:
                self.amp_manager.unscale_gradients(optimizer)

            # Clip gradients
            grad_norm = self.gradient_clipper.clip(model_parameters)
            if grad_norm is not None:
                info["grad_norm"] = float(grad_norm)

            # Optimizer step
            self.amp_manager.step(optimizer)

            # Zero gradients
            optimizer.zero_grad()

            info["stepped"] = True
        else:
            info["stepped"] = False

        return info

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            "amp_state": self.amp_manager.state_dict(),
            "accumulator_step": self.accumulator.current_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        if "amp_state" in state_dict:
            self.amp_manager.load_state_dict(state_dict["amp_state"])
        if "accumulator_step" in state_dict:
            self.accumulator.current_step = state_dict["accumulator_step"]
