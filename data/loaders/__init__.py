"""
Module de chargement des données pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour charger et préparer les données
pour l'entraînement des modèles de détection des trouées forestières,
notamment des DataLoaders optimisés pour PyTorch.
"""

from .factory import (
    create_dataloader,
    create_dataset,
    create_train_val_dataloaders
)

from .optimization import (
    optimize_batch_size,
    optimize_num_workers,
    benchmark_dataloader,
    prefetch_data,
    optimize_dataloader
)

from .calibration import (
    DataLoaderCalibrator,
    create_calibrated_dataloader,
    create_calibrated_train_val_dataloaders
)

from .archive import (
    TarArchiveDataset,
    IterableTarArchiveDataset,
    create_tar_archive,
    convert_dataset_to_tar
)

def create_data_loaders(config, batch_size=None, num_workers=None, **kwargs):
    """
    Wrapper de compatibilité pour create_train_val_dataloaders.

    Cette fonction adapte l'interface pour pipeline/complete_workflow.py
    et autres scripts existants qui utilisent create_data_loaders.

    Args:
        config: Objet de configuration (ConfigurationManager ou dict-like)
        batch_size (int, optional): Taille des batches. Si None, extrait de config.
        num_workers (int, optional): Nombre de workers. Si None, extrait de config.
        **kwargs: Arguments supplémentaires passés à create_train_val_dataloaders

    Returns:
        dict: Dictionnaire avec 'train', 'val', 'test' DataLoaders
    """
    import glob
    from pathlib import Path

    # Extraire les paramètres de config
    if hasattr(config, 'get'):
        # ConfigurationManager style
        tiles_dir = config.get('data.tiles_dir', config.get('TILES_DIR', './tiles'))
        batch_size = batch_size or config.get('training.batch_size', config.get('BATCH_SIZE', 8))
        num_workers = num_workers or config.get('training.num_workers', config.get('NUM_WORKERS', 4))
    else:
        # Object attributes style
        tiles_dir = getattr(config, 'TILES_DIR', './tiles')
        batch_size = batch_size or getattr(config, 'BATCH_SIZE', 8)
        num_workers = num_workers or getattr(config, 'NUM_WORKERS', 4)

    # Collecter les fichiers de tuiles
    tiles_path = Path(tiles_dir)
    input_paths = sorted(glob.glob(str(tiles_path / "**" / "*_dsm.tif"), recursive=True))
    mask_paths = sorted(glob.glob(str(tiles_path / "**" / "*_mask.tif"), recursive=True))

    if not input_paths:
        raise FileNotFoundError(f"Aucune tuile DSM trouvée dans {tiles_dir}")
    if not mask_paths:
        raise FileNotFoundError(f"Aucun masque trouvé dans {tiles_dir}")

    return create_train_val_dataloaders(
        input_paths=input_paths,
        mask_paths=mask_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )


__all__ = [
    # Fonctions de création de DataLoaders
    'create_dataloader',
    'create_dataset',
    'create_train_val_dataloaders',
    'create_data_loaders',  # Wrapper de compatibilité

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
