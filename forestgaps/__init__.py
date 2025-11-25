"""
ForestGaps: Bibliothèque PyTorch pour la détection et l'analyse des trouées forestières.

Ce package fournit des outils pour l'entraînement de modèles de deep learning
pour la détection et l'analyse des trouées forestières à partir d'images de télédétection.

Modules principaux:
    environment - Gestion de l'environnement d'exécution (Colab ou local)
    evaluation - Évaluation des modèles entraînés sur des données externes
    inference - Application des modèles entraînés à de nouvelles données

Utilisation basique:
    from forestgaps.environment import setup_environment
    env = setup_environment()  # Configure automatiquement l'environnement

    from forestgaps.inference import run_inference
    result = run_inference(model_path, dsm_path, output_path)
"""

from .__version__ import __version__

# Imports principaux pour faciliter l'accès aux fonctionnalités
try:
    from . import environment
    from .environment import setup_environment, get_device
    from . import evaluation
    from . import inference

    # Modules optionnels qui peuvent être importés séparément
    # selon les besoins et la disponibilité
    __all__ = [
        "__version__",
        "environment",
        "setup_environment",
        "get_device",
        "evaluation",
        "inference"
    ]

except ImportError as e:
    import warnings
    warnings.warn(f"Certains modules n'ont pas pu être importés: {e}")
    __all__ = ["__version__"]
