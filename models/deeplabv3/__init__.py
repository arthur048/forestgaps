"""
Implémentations de l'architecture DeepLabV3+ pour la détection des trouées forestières.

Ce module fournit différentes variantes de l'architecture DeepLabV3+ adaptées
à la segmentation d'images pour la détection des trouées forestières.
"""

from .basic import DeepLabV3Plus
from .condition import ThresholdConditionedDeepLabV3Plus

__all__ = [
    "DeepLabV3Plus",
    "ThresholdConditionedDeepLabV3Plus"
] 