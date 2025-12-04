"""
Module d'optimisation pour l'entraînement des modèles.

Ce module fournit des classes et des fonctions pour l'optimisation
des paramètres d'entraînement des modèles de segmentation.

Conforme "Audit du workflow PyTorch": LR scheduling, gradient clipping, AMP.
"""

from .lr_schedulers import create_scheduler as create_scheduler_old
from .schedulers import create_scheduler
from .regularization import CompositeRegularization, DropPathScheduler
from .optimization_utils import (
    GradientClipper,
    AMPManager,
    GradientAccumulator,
    TrainingOptimizer,
    enable_gradient_checkpointing,
    compile_model,
)

__all__ = [
    # LR Schedulers
    'create_scheduler',
    'create_scheduler_old',
    # Regularization
    'CompositeRegularization',
    'DropPathScheduler',
    # Optimization utilities
    'GradientClipper',
    'AMPManager',
    'GradientAccumulator',
    'TrainingOptimizer',
    'enable_gradient_checkpointing',
    'compile_model',
] 