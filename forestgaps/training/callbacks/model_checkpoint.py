"""Model checkpoint callback."""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from .base import Callback


class ModelCheckpointCallback(Callback):
    """
    Save model checkpoints during training.

    Can save:
    - Best model only (based on monitored metric)
    - Every epoch
    - Every N epochs
    """

    def __init__(
        self,
        save_dir: str,
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_frequency: int = 1,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback.

        Args:
            save_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for the monitored metric
            save_frequency: Save every N epochs (if not save_best_only)
            verbose: Whether to print messages
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_frequency = save_frequency
        self.verbose = verbose

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Track best value
        self.best_value = float('inf') if mode == "min" else float('-inf')

    def on_epoch_end(self, epoch: int, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Save checkpoint at epoch end."""
        if logs is None:
            return

        current = logs.get(self.monitor)

        # Save regular checkpoint
        if not self.save_best_only and (epoch + 1) % self.save_frequency == 0:
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
            self._save_checkpoint(trainer, checkpoint_path, epoch, logs)
            if self.verbose:
                print(f"  → Checkpoint saved: {checkpoint_path}")

        # Save best model
        if current is not None:
            improved = (
                (self.mode == "min" and current < self.best_value) or
                (self.mode == "max" and current > self.best_value)
            )

            if improved:
                self.best_value = current
                best_path = self.save_dir / "best_model.pt"
                self._save_checkpoint(trainer, best_path, epoch, logs)
                if self.verbose:
                    print(f"  → Best model saved ({self.monitor}={current:.6f}): {best_path}")

    def _save_checkpoint(
        self,
        trainer: Any,
        path: Path,
        epoch: int,
        logs: Dict[str, Any]
    ):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "logs": logs,
        }

        # Add scheduler if available
        if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        torch.save(checkpoint, path)
