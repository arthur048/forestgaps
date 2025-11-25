"""
Module de chargement des données pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour charger et préparer les données
pour l'entraînement des modèles de détection des trouées forestières,
notamment des DataLoaders optimisés pour PyTorch.
"""

from data.loaders.factory import (
    create_dataloader,
    create_dataset,
    create_train_val_dataloaders
)

from data.loaders.optimization import (
    optimize_batch_size,
    optimize_num_workers,
    benchmark_dataloader,
    prefetch_data,
    optimize_dataloader
)

from data.loaders.calibration import (
    DataLoaderCalibrator,
    create_calibrated_dataloader,
    create_calibrated_train_val_dataloaders
)

from data.loaders.archive import (
    TarArchiveDataset,
    IterableTarArchiveDataset,
    create_tar_archive,
    convert_dataset_to_tar
)

__all__ = [
    # Fonctions de création de DataLoaders
    'create_dataloader',
    'create_dataset',
    'create_train_val_dataloaders',
    
    # Fonctions d'optimisation
    'optimize_batch_size',
    'optimize_num_workers',
    'benchmark_dataloader',
    'prefetch_data',
    'optimize_dataloader',
    
    # Fonctions de calibration
    'DataLoaderCalibrator',
    'create_calibrated_dataloader',
    'create_calibrated_train_val_dataloaders',
    
    # Classes et fonctions d'archive
    'TarArchiveDataset',
    'IterableTarArchiveDataset',
    'create_tar_archive',
    'convert_dataset_to_tar'
]
