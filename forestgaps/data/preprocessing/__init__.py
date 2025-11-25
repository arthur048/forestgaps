"""
Sous-module de prétraitement des données pour la détection des trouées forestières.

Ce sous-module fournit des fonctionnalités pour l'analyse, l'alignement et la conversion
des données raster (DSM/CHM) avant leur utilisation pour l'entraînement des modèles.
"""

from .alignment import align_rasters, check_alignment
from .analysis import analyze_raster_pair, verify_raster_integrity
from .conversion import convert_to_numpy, convert_to_geotiff

__all__ = [
    # Fonctions d'alignement
    'align_rasters',
    'check_alignment',
    
    # Fonctions d'analyse
    'analyze_raster_pair',
    'verify_raster_integrity',
    
    # Fonctions de conversion
    'convert_to_numpy',
    'convert_to_geotiff'
]
