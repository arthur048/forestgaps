"""Progress bar callback."""

from typing import Dict, Any, Optional
from .base import Callback

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressBarCallback(Callback):
    """
    Display training progress with tqdm.

    Conforme "Audit du workflow PyTorch": enhanced progress bars.
    """

    def __init__(self, total_epochs: int):
        """
        Initialize progress bar callback.

        Args:
            total_epochs: Total number of training epochs
        """
        super().__init__()

        if not TQDM_AVAILABLE:
            raise ImportError("tqdm not available. Install with: pip install tqdm")

        self.total_epochs = total_epochs
        self.epoch_pbar = None
        self.batch_pbar = None

    def on_train_begin(self, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Create epoch progress bar."""
        self.epoch_pbar = tqdm(total=self.total_epochs, desc="Training", unit="epoch")

    def on_epoch_begin(self, epoch: int, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Update epoch progress bar."""
        if self.epoch_pbar:
            self.epoch_pbar.set_description(f"Epoch {epoch+1}/{self.total_epochs}")

    def on_epoch_end(self, epoch: int, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Update epoch progress bar with metrics."""
        if self.epoch_pbar:
            # Format metrics for display
            if logs:
                postfix = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in logs.items()}
                self.epoch_pbar.set_postfix(postfix)
            self.epoch_pbar.update(1)

    def on_train_end(self, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Close progress bars."""
        if self.epoch_pbar:
            self.epoch_pbar.close()
        if self.batch_pbar:
            self.batch_pbar.close()
