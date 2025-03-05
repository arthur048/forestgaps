"""
Module de gestion des datasets pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour créer, gérer et manipuler des datasets
pour l'entraînement des modèles de détection des trouées forestières.
"""

from data.datasets.gap_dataset import (
    ForestGapDataset,
    create_gap_dataset,
    load_dataset_from_metadata,
    split_dataset
)

from data.datasets.samplers import (
    BalancedGapSampler,
    calculate_gap_ratios,
    create_weighted_sampler
)

from data.datasets.transforms import (
    ForestGapTransforms,
    create_transform_pipeline,
    elastic_transform
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
    'elastic_transform'
]
