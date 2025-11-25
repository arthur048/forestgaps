"""
Module de callbacks pour l'entraînement des modèles.

Ce module fournit des classes et des fonctions pour gérer des événements
durant l'entraînement des modèles de segmentation.
"""

from .base import Callback
from .logging import LoggingCallback
from .checkpointing import CheckpointingCallback
from .visualization import VisualizationCallback

__all__ = [
    'Callback',
    'LoggingCallback',
    'CheckpointingCallback',
    'VisualizationCallback'
] 