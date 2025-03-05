"""
Module de génération de données pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour générer des tuiles et des masques
à partir des données raster (DSM/CHM) pour l'entraînement des modèles de détection
des trouées forestières.
"""

from data.generation.tiling import (
    generate_tiles,
    create_tile_grid,
    extract_tile,
    save_tile_metadata
)

from data.generation.masks import (
    generate_gap_masks,
    create_binary_mask,
    refine_mask,
    calculate_mask_statistics
)

__all__ = [
    # Fonctions de tuilage
    'generate_tiles',
    'create_tile_grid',
    'extract_tile',
    'save_tile_metadata',
    
    # Fonctions de génération de masques
    'generate_gap_masks',
    'create_binary_mask',
    'refine_mask',
    'calculate_mask_statistics'
]
