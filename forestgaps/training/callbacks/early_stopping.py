"""Early stopping callback."""

import numpy as np
from typing import Dict, Any, Optional
from .base import Callback


class EarlyStoppingCallback(Callback):
    """
    Stop training when monitored metric stops improving.

    Conforme Document 3: patience=10 epochs pour val_loss.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            monitor: Metric to monitor (e.g., 'val_loss', 'val_iou')
            patience: Number of epochs with no improvement to wait
            mode: 'min' for loss, 'max' for metrics
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_value = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Reset state at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = np.inf if self.mode == "min" else -np.inf

    def on_epoch_end(self, epoch: int, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Check if we should stop training."""
        if logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            if self.verbose:
                print(f"Warning: Early stopping requires {self.monitor} but it's not available.")
            return

        # Check if improved
        if self.mode == "min":
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.wait = 0
            if self.verbose:
                print(f"  → {self.monitor} improved to {current:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"  → {self.monitor} did not improve from {self.best_value:.6f} (patience: {self.wait}/{self.patience})")

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch}: {self.monitor} has not improved for {self.patience} epochs")

    def on_train_end(self, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Print final message."""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Training stopped at epoch {self.stopped_epoch}")
            print(f"Best {self.monitor}: {self.best_value:.6f}")
