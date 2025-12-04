"""LR Scheduler callback."""

from typing import Dict, Any, Optional
from .base import Callback


class LRSchedulerCallback(Callback):
    """
    Wrapper callback for LR schedulers.

    Calls scheduler.step() at the appropriate time.
    """

    def __init__(self, scheduler, step_on: str = "epoch", monitor: Optional[str] = None):
        """
        Initialize LR scheduler callback.

        Args:
            scheduler: PyTorch LR scheduler instance
            step_on: When to step ('epoch' or 'batch')
            monitor: Metric to pass to scheduler (for ReduceLROnPlateau)
        """
        super().__init__()
        self.scheduler = scheduler
        self.step_on = step_on
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Step scheduler at epoch end if configured."""
        if self.step_on == "epoch":
            if self.monitor and logs:
                # For ReduceLROnPlateau
                metric_value = logs.get(self.monitor)
                if metric_value is not None:
                    self.scheduler.step(metric_value)
            else:
                self.scheduler.step()

    def on_batch_end(self, batch_idx: int, trainer: Any, logs: Optional[Dict[str, Any]] = None):
        """Step scheduler at batch end if configured (for OneCycleLR)."""
        if self.step_on == "batch":
            self.scheduler.step()
