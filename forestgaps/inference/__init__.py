"""
Module d'inférence pour ForestGaps.

Ce module fournit les fonctionnalités nécessaires pour appliquer les modèles entraînés
à de nouvelles données DSM. Il comprend des outils pour le prétraitement des données,
l'application des modèles, et le post-traitement des prédictions.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from .core import (
    InferenceManager,
    InferenceResult,
    InferenceConfig
)
from .utils.geospatial import (
    load_raster,
    save_raster,
    preserve_metadata
)
from .utils.processing import (
    preprocess_dsm,
    postprocess_prediction,
    batch_predict
)
from .utils.visualization import (
    visualize_predictions,
    create_comparison_figure
)

# Configuration du logging
logger = logging.getLogger(__name__)

# Exposer les fonctions principales
__all__ = [
    # Classes principales
    "InferenceManager",
    "InferenceResult",
    "InferenceConfig",
    
    # Fonctions d'inférence
    "run_inference",
    "run_batch_inference",
    
    # Fonctions utilitaires
    "load_raster",
    "save_raster",
    "preprocess_dsm",
    "postprocess_prediction",
    "batch_predict",
    "visualize_predictions",
    "create_comparison_figure"
]

def run_inference(
    model_path: str,
    dsm_path: str,
    output_path: Optional[str] = None,
    threshold: float = 5.0,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    visualize: bool = False
) -> InferenceResult:
    """
    Exécute l'inférence sur un fichier DSM unique avec un modèle préentraîné.
    
    Args:
        model_path: Chemin vers le modèle préentraîné (.pt)
        dsm_path: Chemin vers le fichier DSM d'entrée (GeoTIFF, etc.)
        output_path: Chemin pour sauvegarder la prédiction (optionnel)
        threshold: Seuil de hauteur pour la détection des trouées (en mètres)
        config: Configuration supplémentaire pour l'inférence (optionnel)
        device: Dispositif sur lequel exécuter l'inférence ('cpu', 'cuda', etc.)
        visualize: Générer des visualisations des prédictions
        
    Returns:
        Résultat de l'inférence contenant la prédiction et les métadonnées
    """
    # Créer un gestionnaire d'inférence
    manager = InferenceManager(
        model_path=model_path,
        config=config,
        device=device
    )
    
    # Exécuter l'inférence
    result = manager.predict(
        dsm_path=dsm_path,
        threshold=threshold,
        output_path=output_path,
        visualize=visualize
    )
    
    return result

def run_batch_inference(
    model_path: str,
    dsm_paths: List[str],
    output_dir: str,
    threshold: float = 5.0,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    visualize: bool = False
) -> Dict[str, InferenceResult]:
    """
    Exécute l'inférence sur plusieurs fichiers DSM avec un modèle préentraîné.
    
    Args:
        model_path: Chemin vers le modèle préentraîné (.pt)
        dsm_paths: Liste des chemins vers les fichiers DSM d'entrée
        output_dir: Répertoire pour sauvegarder les prédictions
        threshold: Seuil de hauteur pour la détection des trouées (en mètres)
        config: Configuration supplémentaire pour l'inférence (optionnel)
        device: Dispositif sur lequel exécuter l'inférence ('cpu', 'cuda', etc.)
        batch_size: Taille des lots pour le traitement par lots
        num_workers: Nombre de processus parallèles pour le chargement des données
        visualize: Générer des visualisations des prédictions
        
    Returns:
        Dictionnaire de résultats d'inférence, indexé par chemin de fichier DSM
    """
    # Créer un gestionnaire d'inférence
    manager = InferenceManager(
        model_path=model_path,
        config=config,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Exécuter l'inférence par lots
    results = manager.predict_batch(
        dsm_paths=dsm_paths,
        threshold=threshold,
        output_dir=output_dir,
        visualize=visualize
    )
    
    return results 