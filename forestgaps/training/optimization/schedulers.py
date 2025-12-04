"""
LR Schedulers pour optimisation de l'entraînement.

Implémente OneCycleLR, CosineAnnealing, etc. conformément aux recommandations
du document "Audit du workflow PyTorch".
"""

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional


def create_scheduler(
    optimizer: Optimizer,
    scheduler_config: Dict[str, Any],
    steps_per_epoch: int,
    epochs: int
) -> Optional[_LRScheduler]:
    """
    Factory pour créer des LR schedulers.

    Args:
        optimizer: Optimiseur PyTorch
        scheduler_config: Configuration du scheduler
        steps_per_epoch: Nombre de steps par epoch
        epochs: Nombre total d'epochs

    Returns:
        Scheduler PyTorch ou None

    Exemple de config:
        {
            "type": "onecycle",
            "max_lr": 0.01,
            "pct_start": 0.3
        }
    """
    scheduler_type = scheduler_config.get("type", "none")

    if scheduler_type == "none" or scheduler_type is None:
        return None

    elif scheduler_type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config.get("max_lr", 0.01),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=scheduler_config.get("pct_start", 0.3),
            anneal_strategy=scheduler_config.get("anneal_strategy", "cos"),
            div_factor=scheduler_config.get("div_factor", 25.0),
            final_div_factor=scheduler_config.get("final_div_factor", 1e4)
        )

    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get("T_0", 10),
            T_mult=scheduler_config.get("T_mult", 2),
            eta_min=scheduler_config.get("eta_min", 1e-6)
        )

    elif scheduler_type == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=scheduler_config.get("eta_min", 0)
        )

    elif scheduler_type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5),
            min_lr=scheduler_config.get("min_lr", 1e-7),
            verbose=scheduler_config.get("verbose", True)
        )

    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 10),
            gamma=scheduler_config.get("gamma", 0.1)
        )

    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get("gamma", 0.95)
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class WarmupScheduler(_LRScheduler):
    """
    Warmup LR Scheduler.

    Augmente linéairement le learning rate pendant les premières epochs,
    puis applique un scheduler sous-jacent.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[_LRScheduler] = None,
        warmup_start_lr: float = 1e-7
    ):
        """
        Initialise le warmup scheduler.

        Args:
            optimizer: Optimiseur PyTorch
            warmup_epochs: Nombre d'epochs de warmup
            base_scheduler: Scheduler à appliquer après warmup
            warmup_start_lr: LR initial pour le warmup
        """
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Phase warmup: augmentation linéaire
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            # Après warmup: utiliser base_scheduler si disponible
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            else:
                return self.base_lrs

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super().step(epoch)
        elif self.base_scheduler is not None:
            self.base_scheduler.step(epoch)
