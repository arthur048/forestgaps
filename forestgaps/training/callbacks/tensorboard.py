"""TensorBoard callback for logging."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
from .base import Callback

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorBoardCallback(Callback):
    """
    Log metrics and visualizations to TensorBoard.

    Conforme "Audit du workflow PyTorch": unified monitoring system.
    """

    def __init__(self, log_dir: str, comment: str = ""):
        """
        Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs
            comment: Comment to append to log dir name
        """
        super().__init__()

        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available. Install with: pip install tensorboard")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir), comment=comment)
        self.global_step = 0

    def on_train_begin(self, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Log model graph at training start."""
        # Log model architecture
        try:
            dummy_input = torch.randn(1, trainer.model.in_channels, 256, 256)
            if hasattr(trainer.model, 'to'):
                dummy_input = dummy_input.to(next(trainer.model.parameters()).device)
            self.writer.add_graph(trainer.model, dummy_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")

    def on_epoch_end(self, epoch: int, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Log metrics at epoch end."""
        if logs is None:
            return

        # Log all scalar metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)

        # Log learning rate
        if hasattr(trainer, "optimizer"):
            lr = trainer.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("learning_rate", lr, epoch)

    def on_batch_end(self, batch_idx: int, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Log batch-level metrics."""
        if logs is None:
            return

        self.global_step += 1

        # Log batch loss
        if "loss" in logs:
            self.writer.add_scalar("batch/loss", logs["loss"], self.global_step)

    def on_train_end(self, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Close writer at training end."""
        self.writer.close()
