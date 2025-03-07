"""
Module d'évaluation externe pour ForestGaps-DL.

Ce module fournit les fonctionnalités nécessaires pour évaluer les modèles entraînés
sur des paires DSM/CHM indépendantes. Il permet de calculer des métriques détaillées
de performance et de générer des rapports d'évaluation complets.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from .core import (
    ExternalEvaluator,
    EvaluationResult,
    EvaluationConfig
)
from .utils.metrics import (
    calculate_metrics,
    calculate_threshold_metrics,
    calculate_confusion_matrix
)
from .utils.visualization import (
    visualize_metrics,
    visualize_comparison,
    create_metrics_table
)
from .utils.reporting import (
    generate_evaluation_report,
    save_metrics_to_csv,
    create_site_comparison
)

# Configuration du logging
logger = logging.getLogger(__name__)

# Exposer les fonctions principales
__all__ = [
    # Classes principales
    "ExternalEvaluator",
    "EvaluationResult",
    "EvaluationConfig",
    
    # Fonctions d'évaluation
    "evaluate_model",
    "evaluate_site",
    "evaluate_model_on_sites",
    "compare_models",
    
    # Fonctions utilitaires
    "calculate_metrics",
    "calculate_threshold_metrics",
    "calculate_confusion_matrix",
    "visualize_metrics",
    "visualize_comparison",
    "create_metrics_table",
    "generate_evaluation_report",
    "save_metrics_to_csv",
    "create_site_comparison"
]

def evaluate_model(
    model_path: str,
    dsm_path: str,
    chm_path: str,
    output_dir: Optional[str] = None,
    thresholds: Optional[List[float]] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    visualize: bool = False
) -> EvaluationResult:
    """
    Évalue un modèle sur une paire DSM/CHM unique.
    
    Args:
        model_path: Chemin vers le modèle préentraîné (.pt)
        dsm_path: Chemin vers le fichier DSM d'entrée
        chm_path: Chemin vers le fichier CHM pour la vérité terrain
        output_dir: Répertoire pour sauvegarder les résultats d'évaluation (optionnel)
        thresholds: Liste des seuils de hauteur à évaluer (par défaut: [2.0, 5.0, 10.0, 15.0])
        config: Configuration supplémentaire pour l'évaluation (optionnel)
        device: Dispositif sur lequel exécuter l'évaluation ('cpu', 'cuda', etc.)
        visualize: Générer des visualisations des résultats
        
    Returns:
        Résultat de l'évaluation contenant les métriques et les métadonnées
    """
    # Créer un évaluateur
    evaluator = ExternalEvaluator(
        model_path=model_path,
        config=config,
        device=device
    )
    
    # Définir les seuils par défaut si non spécifiés
    if thresholds is None:
        thresholds = [2.0, 5.0, 10.0, 15.0]
    
    # Exécuter l'évaluation
    result = evaluator.evaluate(
        dsm_path=dsm_path,
        chm_path=chm_path,
        thresholds=thresholds,
        output_dir=output_dir,
        visualize=visualize
    )
    
    return result

def evaluate_site(
    model_path: str,
    site_dsm_dir: str,
    site_chm_dir: str,
    output_dir: str,
    thresholds: Optional[List[float]] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    visualize: bool = False
) -> EvaluationResult:
    """
    Évalue un modèle sur un site complet (plusieurs paires DSM/CHM).
    
    Args:
        model_path: Chemin vers le modèle préentraîné (.pt)
        site_dsm_dir: Répertoire contenant les fichiers DSM du site
        site_chm_dir: Répertoire contenant les fichiers CHM du site
        output_dir: Répertoire pour sauvegarder les résultats d'évaluation
        thresholds: Liste des seuils de hauteur à évaluer (par défaut: [2.0, 5.0, 10.0, 15.0])
        config: Configuration supplémentaire pour l'évaluation (optionnel)
        device: Dispositif sur lequel exécuter l'évaluation ('cpu', 'cuda', etc.)
        batch_size: Taille des lots pour le traitement par lots
        num_workers: Nombre de processus parallèles pour le chargement des données
        visualize: Générer des visualisations des résultats
        
    Returns:
        Résultat de l'évaluation agrégé pour tout le site
    """
    # Créer un évaluateur
    evaluator = ExternalEvaluator(
        model_path=model_path,
        config=config,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Définir les seuils par défaut si non spécifiés
    if thresholds is None:
        thresholds = [2.0, 5.0, 10.0, 15.0]
    
    # Exécuter l'évaluation sur le site
    result = evaluator.evaluate_site(
        site_dsm_dir=site_dsm_dir,
        site_chm_dir=site_chm_dir,
        thresholds=thresholds,
        output_dir=output_dir,
        visualize=visualize
    )
    
    return result

def evaluate_model_on_sites(
    model_path: str,
    sites_config: Dict[str, Dict[str, str]],
    output_dir: str,
    thresholds: Optional[List[float]] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    visualize: bool = False,
    aggregate_results: bool = True
) -> Dict[str, EvaluationResult]:
    """
    Évalue un modèle sur plusieurs sites.
    
    Args:
        model_path: Chemin vers le modèle préentraîné (.pt)
        sites_config: Configuration des sites à évaluer {nom_site: {"dsm_dir": path, "chm_dir": path}}
        output_dir: Répertoire pour sauvegarder les résultats d'évaluation
        thresholds: Liste des seuils de hauteur à évaluer (par défaut: [2.0, 5.0, 10.0, 15.0])
        config: Configuration supplémentaire pour l'évaluation (optionnel)
        device: Dispositif sur lequel exécuter l'évaluation ('cpu', 'cuda', etc.)
        batch_size: Taille des lots pour le traitement par lots
        num_workers: Nombre de processus parallèles pour le chargement des données
        visualize: Générer des visualisations des résultats
        aggregate_results: Agréger les résultats de tous les sites
        
    Returns:
        Dictionnaire des résultats d'évaluation pour chaque site et résultat agrégé si demandé
    """
    # Créer un évaluateur
    evaluator = ExternalEvaluator(
        model_path=model_path,
        config=config,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Définir les seuils par défaut si non spécifiés
    if thresholds is None:
        thresholds = [2.0, 5.0, 10.0, 15.0]
    
    # Exécuter l'évaluation sur tous les sites
    results = evaluator.evaluate_multi_sites(
        sites_config=sites_config,
        thresholds=thresholds,
        output_dir=output_dir,
        visualize=visualize,
        aggregate_results=aggregate_results
    )
    
    return results

def compare_models(
    model_paths: Dict[str, str],
    dsm_path: str,
    chm_path: str,
    output_dir: str,
    thresholds: Optional[List[float]] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    visualize: bool = True
) -> Dict[str, EvaluationResult]:
    """
    Compare plusieurs modèles sur une même paire DSM/CHM.
    
    Args:
        model_paths: Dictionnaire des chemins de modèles {nom_modèle: chemin}
        dsm_path: Chemin vers le fichier DSM d'entrée
        chm_path: Chemin vers le fichier CHM pour la vérité terrain
        output_dir: Répertoire pour sauvegarder les résultats de comparaison
        thresholds: Liste des seuils de hauteur à évaluer (par défaut: [2.0, 5.0, 10.0, 15.0])
        config: Configuration supplémentaire pour l'évaluation (optionnel)
        device: Dispositif sur lequel exécuter l'évaluation ('cpu', 'cuda', etc.)
        visualize: Générer des visualisations des résultats
        
    Returns:
        Dictionnaire des résultats d'évaluation pour chaque modèle
    """
    # Définir les seuils par défaut si non spécifiés
    if thresholds is None:
        thresholds = [2.0, 5.0, 10.0, 15.0]
    
    # Résultats pour chaque modèle
    results = {}
    
    # Évaluer chaque modèle
    for model_name, model_path in model_paths.items():
        # Créer un évaluateur pour ce modèle
        evaluator = ExternalEvaluator(
            model_path=model_path,
            config=config,
            device=device
        )
        
        # Créer le répertoire de sortie pour ce modèle
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Exécuter l'évaluation
        result = evaluator.evaluate(
            dsm_path=dsm_path,
            chm_path=chm_path,
            thresholds=thresholds,
            output_dir=model_output_dir,
            visualize=visualize
        )
        
        # Stocker le résultat
        results[model_name] = result
    
    # Générer un rapport de comparaison
    if visualize:
        from .utils.reporting import generate_comparison_report
        generate_comparison_report(
            results=results,
            output_dir=output_dir,
            thresholds=thresholds
        )
    
    return results 