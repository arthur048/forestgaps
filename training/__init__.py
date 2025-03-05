"""
Module d'entraînement pour le projet forestgaps-dl.

Ce module fournit des classes et des fonctions pour l'entraînement des modèles de segmentation
pour la détection des trouées forestières.
"""

from .trainer import Trainer
from .metrics.segmentation import SegmentationMetrics
from .loss.combined import CombinedFocalDiceLoss
from .callbacks.base import Callback

# Exporter les classes et fonctions principales
__all__ = [
    'Trainer',
    'SegmentationMetrics',
    'CombinedFocalDiceLoss',
    'Callback'
] 