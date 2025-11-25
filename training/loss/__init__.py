"""
Module de fonctions de perte pour l'entraînement des modèles.

Ce module fournit des implémentations de diverses fonctions de perte
pour l'entraînement des modèles de détection des trouées forestières.
"""

from .combined import CombinedFocalDiceLoss
from .factory import create_loss_function, create_loss_with_threshold_weights
from .regression import MSELoss, MAELoss, HuberLoss, CombinedRegressionLoss, RegressionLossWithWeights

__all__ = [
    # Fonctions de perte de segmentation
    'CombinedFocalDiceLoss',
    
    # Fonctions de perte de régression
    'MSELoss',
    'MAELoss',
    'HuberLoss',
    'CombinedRegressionLoss',
    'RegressionLossWithWeights',
    
    # Fonctions de fabrique
    'create_loss_function',
    'create_loss_with_threshold_weights'
] 