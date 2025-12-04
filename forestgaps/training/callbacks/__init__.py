"""
Module de callbacks pour l'entraînement des modèles.

Ce module fournit des classes et des fonctions pour gérer des événements
durant l'entraînement des modèles de segmentation.

Conforme "Audit du workflow PyTorch": event-driven callback system.
"""

from .base import Callback, CallbackList
from .logging import LoggingCallback
from .checkpointing import CheckpointingCallback
from .visualization import VisualizationCallback
from .early_stopping import EarlyStoppingCallback
from .model_checkpoint import ModelCheckpointCallback
from .lr_scheduler import LRSchedulerCallback
from .tensorboard import TensorBoardCallback
from .progress import ProgressBarCallback

__all__ = [
    'Callback',
    'CallbackList',
    'LoggingCallback',
    'CheckpointingCallback',
    'VisualizationCallback',
    'EarlyStoppingCallback',
    'ModelCheckpointCallback',
    'LRSchedulerCallback',
    'TensorBoardCallback',
    'ProgressBarCallback',
] 