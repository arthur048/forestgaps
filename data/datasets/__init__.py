"""
Module de gestion des datasets pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour créer, gérer et manipuler des datasets
pour l'entraînement des modèles de détection des trouées forestières.
"""

from .gap_dataset import (
    ForestGapDataset,
    create_gap_dataset,
    load_dataset_from_metadata,
    split_dataset
)

from .samplers import (
    BalancedGapSampler,
    calculate_gap_ratios,
    create_weighted_sampler
)

from .transforms import (
    ForestGapTransforms,
    create_transform_pipeline,
    elastic_transform
)

from .regression_dataset import (
    ForestRegressionDataset,
    create_regression_dataset,
    create_regression_dataloader,
    split_regression_dataset
)

__all__ = [
    # Classes et fonctions de datasets
    'ForestGapDataset',
    'create_gap_dataset',
    'load_dataset_from_metadata',
    'split_dataset',
    
    # Classes et fonctions d'échantillonnage
    'BalancedGapSampler',
    'calculate_gap_ratios',
    'create_weighted_sampler',
    
    # Classes et fonctions de transformation
    'ForestGapTransforms',
    'create_transform_pipeline',
    'elastic_transform',
    
    # Classes et fonctions de dataset de régression
    'ForestRegressionDataset',
    'create_regression_dataset',
    'create_regression_dataloader',
    'split_regression_dataset'
]
