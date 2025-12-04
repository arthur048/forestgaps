"""
Combo Loss: BCE + Dice + Focal Loss pour class imbalance.

Basé sur les recommandations du document "Entraîner efficacement un modèle U".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss pour segmentation.

    Le coefficient de Dice mesure le chevauchement entre la prédiction et la cible.
    """

    def __init__(self, smooth: float = 1e-6):
        """
        Initialise le Dice Loss.

        Args:
            smooth: Facteur de lissage pour éviter la division par zéro
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcule le Dice Loss.

        Args:
            pred: Logits prédits [B, C, H, W]
            target: Masques cibles [B, C, H, W]

        Returns:
            Dice loss scalaire
        """
        # Appliquer sigmoid si pred contient des logits
        pred = torch.sigmoid(pred)

        # Aplatir pour le calcul
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calcul Dice
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss pour déséquilibre de classes.

    Basé sur "Focal Loss for Dense Object Detection" (Lin et al., 2017).
    Réduit le poids des exemples faciles et se concentre sur les difficiles.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialise le Focal Loss.

        Args:
            alpha: Poids de la classe positive
            gamma: Facteur de modulation (0 = BCE standard)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calcule le Focal Loss.

        Args:
            pred: Logits prédits [B, C, H, W]
            target: Masques cibles [B, C, H, W]

        Returns:
            Focal loss scalaire
        """
        # BCE de base
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Probabilités prédites
        pred_prob = torch.sigmoid(pred)

        # p_t pour la formulation du Focal Loss
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)

        # Poids alpha
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)

        # Facteur de modulation
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Focal Loss
        focal_loss = focal_weight * bce

        return focal_loss.mean()


class ComboLoss(nn.Module):
    """
    Combo Loss = BCE + Dice + Focal.

    Combine trois loss functions pour:
    - BCE: Loss de base
    - Dice: Overlap directionnel
    - Focal: Gestion du class imbalance

    Recommandé par "Entraîner efficacement un modèle U" (Priorité MAX).
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.3,
        focal_weight: float = 0.2,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25
    ):
        """
        Initialise le Combo Loss.

        Args:
            bce_weight: Poids pour BCE
            dice_weight: Poids pour Dice
            focal_weight: Poids pour Focal
            focal_gamma: Paramètre gamma du Focal Loss
            focal_alpha: Paramètre alpha du Focal Loss
        """
        super().__init__()

        # Vérifier que les poids somment à 1
        total = bce_weight + dice_weight + focal_weight
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1, got {total}"

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        # Loss functions
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcule le Combo Loss.

        Args:
            pred: Logits prédits [B, C, H, W]
            target: Masques cibles [B, C, H, W]

        Returns:
            Tuple (loss_total, dict_losses)
            - loss_total: Loss combinée pour backward
            - dict_losses: Dictionnaire avec breakdown des losses
        """
        # Calcul des trois losses
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)

        # Combinaison pondérée
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss
        )

        # Breakdown pour logging
        loss_dict = {
            'bce': bce_loss.item(),
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def __repr__(self):
        return (
            f"ComboLoss(bce={self.bce_weight:.2f}, "
            f"dice={self.dice_weight:.2f}, "
            f"focal={self.focal_weight:.2f})"
        )
