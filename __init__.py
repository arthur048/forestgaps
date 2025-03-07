"""
ForestGaps-DL: Bibliothèque PyTorch pour la détection et l'analyse des trouées forestières.

Ce package fournit des outils pour l'entraînement de modèles de deep learning
pour la détection et l'analyse des trouées forestières à partir d'images de télédétection.
"""

from .__version__ import __version__

# Imports principaux pour faciliter l'accès aux fonctionnalités
try:
    from forestgaps_dl import environment
    from forestgaps_dl.environment import setup_environment, get_device

    # Modules optionnels qui peuvent être importés séparément
    # selon les besoins et la disponibilité
    __all__ = [
        "__version__",
        "environment",
        "setup_environment",
        "get_device"
    ]

except ImportError as e:
    import warnings
    warnings.warn(f"Certains modules n'ont pas pu être importés: {e}")
    __all__ = ["__version__"]
