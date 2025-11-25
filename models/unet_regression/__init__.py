"""
Implémentations de l'architecture U-Net pour les tâches de régression.

Ce module fournit différentes variantes de l'architecture U-Net adaptées
à la prédiction de valeurs continues (régression) pour les données forestières.
"""

from .basic import RegressionUNet
from .condition import ThresholdConditionedRegressionUNet

__all__ = [
    "RegressionUNet",
    "ThresholdConditionedRegressionUNet"
] 