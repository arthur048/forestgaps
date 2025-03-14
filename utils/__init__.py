"""
Module d'utilitaires pour ForestGaps.

Ce module fournit des fonctionnalités communes utilisées dans l'ensemble du package,
notamment pour la visualisation, les entrées/sorties, le profilage et la gestion des erreurs.
"""

# Importer les sous-modules pour les rendre disponibles via utils
from . import visualization
from . import io
from . import profiling

# Importer les classes d'erreurs pour les rendre disponibles directement
from .errors import (
    ForestGapsError, DataError, ModelError, TrainingError, ConfigError, EnvironmentError,
    InvalidDataFormatError, DataProcessingError, ModelInitializationError, ModelLoadingError,
    OutOfMemoryError, TrainingDivergenceError, ErrorHandler
)

# Définir les exports publics
__all__ = [
    'visualization', 'io', 'profiling',
    'ForestGapsError', 'DataError', 'ModelError', 'TrainingError', 'ConfigError', 'EnvironmentError',
    'InvalidDataFormatError', 'DataProcessingError', 'ModelInitializationError', 'ModelLoadingError',
    'OutOfMemoryError', 'TrainingDivergenceError', 'ErrorHandler'
]
