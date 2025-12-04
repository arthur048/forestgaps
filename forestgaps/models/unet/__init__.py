"""
Implémentations de l'architecture U-Net pour la détection des trouées forestières.

Ce module fournit différentes variantes de l'architecture U-Net adaptées
à la segmentation d'images pour la détection des trouées forestières.
"""

from .basic import UNet
from .attention_unet import AttentionUNet  # Fixed: dimension mismatch in decoder resolved
from .residual_unet import ResUNet
from .film_unet import FiLMUNet
from .all_features import UNetWithAllFeatures

__all__ = [
    "UNet",
    "AttentionUNet",
    "ResUNet",
    "FiLMUNet",
    "UNetWithAllFeatures"
] 