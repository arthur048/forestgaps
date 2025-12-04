# 🎯 PLAN D'ACTION PRIORITAIRE - ForestGaps

**Date**: 2025-12-04
**Contexte**: Post-analyse complète des documents de référence
**État actuel**: 8/9 modèles fonctionnels (88.9%)

---

## ✅ TRAVAIL ACCOMPLI AUJOURD'HUI

### 1. Analyse Exhaustive de la Documentation

**Documents analysés**:
- ✅ `Entraîner efficacement un modèle U.docx`: Roadmap priorisée par phases
- ✅ `Audit du workflow PyTorch.docx`: 50+ pages de recommandations techniques
- ✅ `U-Net_ForestGaps_DSM_Matériel_Méthode.docx`: Méthodologie de référence complète
- ✅ `docs/archive/*`: Documentation technique existante (context_llm, package_reference, dev_guide)

### 2. Gap Analysis Complet

**Fichier créé**: [`ANALYSE_COMPLETE_GAPS.md`](./ANALYSE_COMPLETE_GAPS.md)

**Contenu**:
- Matrice de comparaison exhaustive (Implementation vs Recommandations)
- Analyse par module (Architecture, Data Pipeline, Training, Monitoring, etc.)
- Priorisation roadmap (Phase 1/2/3)
- Estimation efforts (~15 jours total)

### 3. Décision Architecturale: Suppression attention_unet

**Fichier créé**: [`docs/ARCHITECTURE_DECISIONS.md`](./docs/ARCHITECTURE_DECISIONS.md)

**Rationale**:
- ❌ Attention Gates non justifiées pour données monocanal DSM
- ✅ ASPP + FiLM + CBAM suffisent et fonctionnent
- ✅ Simplification codebase: 8/8 modèles → 100%
- 📚 Conformité best practices segmentation géospatiale

**Actions effectuées**:
- ✅ Décorateur `@model_registry.register("attention_unet")` commenté
- ✅ Code archivé dans `docs/archive/deprecated/attention_unet.py.bak`
- ✅ Documentation justification complète

---

## 🚀 PROCHAINES ÉTAPES IMMÉDIATES

### Étape 1: Finaliser Suppression attention_unet (5 min)

**Actions**:
1. Redémarrer container Docker pour recharger modules Python
2. Vérifier registry: doit lister 8 modèles au lieu de 9
3. Re-run `test_all_models.py` → 100% success attendu

**Commandes**:
```powershell
# Redémarrer Docker
docker restart forestgaps-main

# Vérifier registry
docker exec forestgaps-main python -c "from forestgaps.models import model_registry; print(sorted(model_registry.list_models()))"

# Test complet
docker exec forestgaps-main python scripts/test_all_models.py
```

**Résultat attendu**:
```
Nombre de modèles à tester: 8
Modèles: unet, resunet, film_unet, unet_all_features, deeplabv3_plus,
         deeplabv3_plus_threshold, regression_unet, regression_unet_threshold

Résultat: 8/8 modèles OK (100.0%)
✅ TOUS LES MODÈLES FONCTIONNENT!
```

---

## 🔴 PHASE 1: FONDATIONS (Priorité MAXIMALE)

**Effort total**: ~6 jours
**Impact**: CRITIQUE pour entraînement efficace

### 1.1 Configuration System (2-3 jours)

**Objectif**: Externaliser tous les paramètres en YAML + validation Pydantic

**Structure cible**:
```
configs/
├── defaults/
│   ├── training.yaml      # Optimizer, scheduler, loss, epochs, etc.
│   ├── data.yaml          # Batch size, augmentations, normalization
│   ├── model.yaml         # Architecture, channels, depth
│   └── paths.yaml         # Data dirs, output dirs, logs
├── experiments/
│   ├── baseline_unet.yaml
│   ├── deeplabv3_aspp.yaml
│   └── film_threshold.yaml
└── README.md
```

**Exemple `training.yaml`**:
```yaml
training:
  epochs: 30
  batch_size: 16

  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.0001
    betas: [0.9, 0.999]

  scheduler:
    type: "onecycle"
    max_lr: 0.01
    pct_start: 0.3

  loss:
    type: "combo"
    bce_weight: 0.5
    dice_weight: 0.3
    focal_weight: 0.2
    focal_gamma: 2.0

  early_stopping:
    enabled: true
    patience: 10
    monitor: "val_loss"
    mode: "min"
```

**Schema Pydantic**:
```python
# forestgaps/config/schemas.py
from pydantic import BaseModel, Field
from typing import Literal, List

class OptimizerConfig(BaseModel):
    type: Literal["adam", "sgd", "adamw"]
    lr: float = Field(gt=0, le=1)
    weight_decay: float = Field(ge=0)

class TrainingConfig(BaseModel):
    epochs: int = Field(gt=0, le=1000)
    batch_size: int = Field(gt=0, le=256)
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
```

**Dépendances**:
```bash
pip install pydantic pyyaml
```

**Checklist**:
- [ ] Créer structure `configs/`
- [ ] Implémenter schemas Pydantic
- [ ] Fonction `load_config(path) -> Config`
- [ ] Validation automatique
- [ ] Tests unitaires pour configs

### 1.2 Combo Loss Implementation (1 jour)

**Objectif**: BCE + Dice + Focal Loss pour gérer class imbalance

**Implémentation**:
```python
# forestgaps/training/losses/combo_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice loss pour segmentation."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  # Si logits
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss pour déséquilibre de classes."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class ComboLoss(nn.Module):
    """Combo Loss = BCE + Dice + Focal"""
    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.3,
        focal_weight=0.2,
        focal_gamma=2.0,
        focal_alpha=0.25
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        self.weights = (bce_weight, dice_weight, focal_weight)
        assert sum(self.weights) == 1.0, "Weights must sum to 1"

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)

        total = (
            self.weights[0] * bce_loss +
            self.weights[1] * dice_loss +
            self.weights[2] * focal_loss
        )

        return total, {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'total': total.item()
        }
```

**Checklist**:
- [ ] Implémenter DiceLoss
- [ ] Implémenter FocalLoss
- [ ] Implémenter ComboLoss
- [ ] Tests unitaires (formes, gradients)
- [ ] Intégration config YAML
- [ ] Benchmark vs BCE seule

### 1.3 LR Scheduling (0.5 jour)

**Objectif**: OneCycleLR ou Cosine Annealing avec Restarts

**Implémentation**:
```python
# forestgaps/training/optimization/schedulers.py
import torch

def create_scheduler(optimizer, config, steps_per_epoch):
    """Factory pour créer LR schedulers."""
    if config.scheduler.type == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.scheduler.max_lr,
            epochs=config.training.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config.scheduler.pct_start,
            anneal_strategy='cos'
        )

    elif config.scheduler.type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.scheduler.T_0,
            T_mult=config.scheduler.T_mult,
            eta_min=config.scheduler.eta_min
        )

    elif config.scheduler.type == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config.scheduler.patience,
            verbose=True
        )

    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler.type}")
```

**Checklist**:
- [ ] Implémenter factory `create_scheduler()`
- [ ] Support OneCycleLR
- [ ] Support CosineAnnealingWarmRestarts
- [ ] Support ReduceLROnPlateau
- [ ] Tests avec différentes configs
- [ ] Intégration dans training loop

### 1.4 Callback System (2 jours)

**Objectif**: Event-driven hooks pour monitoring et contrôle

**Architecture**:
```python
# forestgaps/training/callbacks/base.py
class Callback:
    """Base class for all callbacks."""

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, epoch, trainer):
        pass

    def on_epoch_end(self, epoch, logs, trainer):
        pass

    def on_batch_begin(self, batch_idx, trainer):
        pass

    def on_batch_end(self, batch_idx, logs, trainer):
        pass


# forestgaps/training/callbacks/early_stopping.py
class EarlyStoppingCallback(Callback):
    """Stop training when monitored metric stops improving."""

    def __init__(self, monitor='val_loss', patience=10, mode='min', min_delta=0):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs, trainer):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best_value is None:
            self.best_value = current
            return

        improved = (
            (self.mode == 'min' and current < self.best_value - self.min_delta) or
            (self.mode == 'max' and current > self.best_value + self.min_delta)
        )

        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                trainer.stop_training = True


# forestgaps/training/callbacks/checkpoint.py
class ModelCheckpointCallback(Callback):
    """Save model checkpoints during training."""

    def __init__(self, save_dir, save_best_only=True, monitor='val_loss', mode='min'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.best_value = None

    def on_epoch_end(self, epoch, logs, trainer):
        current = logs.get(self.monitor)

        if not self.save_best_only:
            # Save every epoch
            trainer.save_checkpoint(self.save_dir / f"checkpoint_epoch_{epoch}.pt")

        if current is not None:
            improved = (
                self.best_value is None or
                (self.mode == 'min' and current < self.best_value) or
                (self.mode == 'max' and current > self.best_value)
            )

            if improved:
                self.best_value = current
                trainer.save_checkpoint(self.save_dir / "best_model.pt")
                print(f"\n  → Best model saved (