"""
Module de gestion des données pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour la préparation, le traitement et le chargement
des données pour l'entraînement et l'inférence des modèles de détection de trouées forestières.
"""

# Importations des sous-modules
from . import preprocessing
from . import datasets
from . import generation
from . import loaders
from . import normalization
from . import storage

# Exposition des fonctionnalités principales
__all__ = [
    'preprocessing',
    'datasets',
    'generation',
    'loaders',
    'normalization',
    'storage'
]
