"""
Module d'optimisation pour l'entraînement des modèles.

Ce module fournit des classes et des fonctions pour l'optimisation
des paramètres d'entraînement des modèles de segmentation.
"""

from .lr_schedulers import create_scheduler
from .regularization import CompositeRegularization, DropPathScheduler

__all__ = [
    'create_scheduler',
    'CompositeRegularization',
    'DropPathScheduler'
] 