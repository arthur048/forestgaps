"""
Utilitaires pour l'évaluation des modèles de détection de trouées.

Ce module fournit des fonctions utilitaires pour faciliter l'évaluation
des modèles, le chargement des données, et la génération de rapports.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional, Union, Set
from datetime import datetime

# Imports locaux
from forestgaps_dl.evaluation.metrics import (
    compute_all_metrics, 
    create_metrics_report,
    find_optimal_threshold
)
from forestgaps_dl.inference.utils.geospatial import load_raster

# Configuration du logging
logger = logging.getLogger(__name__)

def create_ground_truth_from_chm(
    chm_path: str,
    gap_threshold: float = 2.0,
    min_gap_area: int = 10,
    save_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Crée un masque de vérité terrain à partir d'un fichier CHM.
    
    Args:
        chm_path: Chemin vers le fichier CHM
        gap_threshold: Seuil de hauteur en dessous duquel on considère qu'il y a une trouée
        min_gap_area: Taille minimale (en pixels) des trouées à conserver
        save_path: Chemin où sauvegarder le masque (optionnel)
        metadata: Métadonnées à associer au masque (optionnel)
        
    Returns:
        Tuple contenant le masque de vérité terrain et les métadonnées
    """
    # Charger le CHM
    chm_data, chm_meta = load_raster(chm_path)
    
    # Créer le masque binaire (pixels < seuil = trouées)
    gap_mask = (chm_data < gap_threshold).astype(np.uint8)
    
    # Filtrer les petites trouées si nécessaire
    if min_gap_area > 1:
        from scipy import ndimage
        
        # Étiqueter les composantes connexes
        labeled_mask, num_features = ndimage.label(gap_mask)
        
        # Calculer la taille de chaque composante
        component_sizes = np.zeros(num_features + 1, dtype=int)
        for i in range(1, num_features + 1):
            component_sizes[i] = np.sum(labeled_mask == i)
        
        # Filtrer les composantes par taille
        size_mask = component_sizes > min_gap_area
        filtered_mask = np.zeros_like(gap_mask)
        for i in range(1, num_features + 1):
            if size_mask[i]:
                filtered_mask[labeled_mask == i] = 1
        
        gap_mask = filtered_mask
    
    # Préparer les métadonnées
    result_meta = {
        "source_file": os.path.basename(chm_path),
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gap_threshold": gap_threshold,
        "min_gap_area": min_gap_area
    }
    
    # Ajouter les métadonnées géospatiales
    if metadata:
        result_meta.update(metadata)
    else:
        result_meta.update({
            "transform": chm_meta.get("transform"),
            "crs": chm_meta.get("crs"),
            "width": chm_meta.get("width"),
            "height": chm_meta.get("height"),
            "nodata": chm_meta.get("nodata")
        })
    
    # Sauvegarder si nécessaire
    if save_path:
        # Sauvegarder le masque
        from forestgaps_dl.inference.utils.geospatial import save_raster
        save_raster(
            gap_mask, 
            save_path, 
            metadata={
                "transform": chm_meta.get("transform"),
                "crs": chm_meta.get("crs"),
                "nodata": 255
            }
        )
        
        # Sauvegarder les métadonnées dans un fichier JSON
        meta_path = os.path.splitext(save_path)[0] + "_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(result_meta, f, indent=2)
    
    return gap_mask, result_meta

def evaluate_prediction(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    prediction_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    model_name: str = "model",
    save_dir: Optional[str] = None,
    save_visualizations: bool = True
) -> Dict[str, float]:
    """
    Évalue une prédiction par rapport à une vérité terrain.
    
    Args:
        prediction: Prédiction (binaire ou probabilité)
        ground_truth: Vérité terrain (binaire)
        prediction_prob: Probabilités de prédiction (optionnel)
        threshold: Seuil pour binariser les probabilités
        model_name: Nom du modèle (pour les sorties)
        save_dir: Répertoire où sauvegarder les résultats
        save_visualizations: Sauvegarder des visualisations
        
    Returns:
        Dictionnaire des métriques d'évaluation
    """
    # Vérifier que les dimensions correspondent
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Les dimensions ne correspondent pas: prediction {prediction.shape}, ground_truth {ground_truth.shape}")
    
    # Déterminer si la prédiction est une probabilité ou binaire
    is_probability = (prediction.dtype == np.float32 or prediction.dtype == np.float64) and np.max(prediction) <= 1.0
    
    # Binariser si nécessaire
    if is_probability:
        prediction_bin = (prediction >= threshold).astype(np.uint8)
        prediction_prob = prediction if prediction_prob is None else prediction_prob
    else:
        prediction_bin = prediction.astype(np.uint8)
    
    # Calculer les métriques
    metrics = compute_all_metrics(ground_truth, prediction_bin, prediction_prob)
    
    # Ajouter des informations sur le modèle
    metrics["model_name"] = model_name
    metrics["threshold"] = threshold
    
    # Sauvegarder les résultats si nécessaire
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Sauvegarder les métriques au format JSON
        metrics_path = os.path.join(save_dir, f"metrics_{model_name}.json")
        with open(metrics_path, 'w') as f:
            # Convertir les valeurs numpy en types Python natifs
            metrics_to_save = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in metrics.items()}
            json.dump(metrics_to_save, f, indent=2)
        
        # Sauvegarder les visualisations
        if save_visualizations:
            from forestgaps_dl.inference.utils.visualization import (
                visualize_prediction,
                visualize_error_map
            )
            
            # Visualisation de la prédiction
            fig_pred = visualize_prediction(
                prediction=prediction_bin,
                ground_truth=ground_truth,
                title=f"Prédiction vs Vérité terrain - {model_name}",
                save_path=os.path.join(save_dir, f"prediction_{model_name}.png")
            )
            plt.close(fig_pred)
            
            # Carte d'erreurs
            fig_error = visualize_error_map(
                prediction=prediction_bin,
                ground_truth=ground_truth,
                title=f"Carte d'erreurs - {model_name}",
                save_path=os.path.join(save_dir, f"error_map_{model_name}.png")
            )
            plt.close(fig_error)
            
            # Si nous avons des probabilités, sauvegarder la courbe ROC et PR
            if prediction_prob is not None:
                from forestgaps_dl.evaluation.metrics import (
                    compute_roc_curve,
                    compute_precision_recall_curve
                )
                from forestgaps_dl.inference.utils.visualization import (
                    plot_roc_curve,
                    plot_precision_recall_curve
                )
                
                # Calculer et tracer la courbe ROC
                fpr, tpr, _ = compute_roc_curve(ground_truth, prediction_prob)
                fig_roc = plot_roc_curve(
                    fpr=fpr,
                    tpr=tpr,
                    auc=metrics.get("roc_auc", 0),
                    model_name=model_name,
                    save_path=os.path.join(save_dir, f"roc_curve_{model_name}.png")
                )
                plt.close(fig_roc)
                
                # Calculer et tracer la courbe précision-rappel
                precision, recall, _ = compute_precision_recall_curve(ground_truth, prediction_prob)
                fig_pr = plot_precision_recall_curve(
                    precision=precision,
                    recall=recall,
                    average_precision=metrics.get("pr_auc", 0),
                    model_name=model_name,
                    save_path=os.path.join(save_dir, f"pr_curve_{model_name}.png")
                )
                plt.close(fig_pr)
    
    return metrics

def compare_models(
    predictions: Dict[str, np.ndarray],
    ground_truth: np.ndarray,
    probabilities: Optional[Dict[str, np.ndarray]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    save_dir: Optional[str] = None,
    save_visualizations: bool = True
) -> pd.DataFrame:
    """
    Compare plusieurs modèles par rapport à une vérité terrain.
    
    Args:
        predictions: Dictionnaire des prédictions par modèle
        ground_truth: Vérité terrain
        probabilities: Dictionnaire des probabilités par modèle (optionnel)
        thresholds: Dictionnaire des seuils par modèle (optionnel)
        save_dir: Répertoire où sauvegarder les résultats
        save_visualizations: Sauvegarder des visualisations
        
    Returns:
        DataFrame contenant les métriques de tous les modèles
    """
    # Initialiser le dictionnaire de métriques
    all_metrics = {}
    
    # Évaluer chaque modèle
    for model_name, prediction in predictions.items():
        # Obtenir la probabilité si disponible
        probability = None
        if probabilities and model_name in probabilities:
            probability = probabilities[model_name]
        
        # Obtenir le seuil
        threshold = 0.5
        if thresholds and model_name in thresholds:
            threshold = thresholds[model_name]
        
        # Créer un sous-répertoire pour ce modèle si nécessaire
        model_dir = None
        if save_dir:
            model_dir = os.path.join(save_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
        
        # Évaluer le modèle
        metrics = evaluate_prediction(
            prediction=prediction,
            ground_truth=ground_truth,
            prediction_prob=probability,
            threshold=threshold,
            model_name=model_name,
            save_dir=model_dir,
            save_visualizations=save_visualizations
        )
        
        # Ajouter les métriques au dictionnaire global
        all_metrics[model_name] = metrics
    
    # Créer un rapport de comparaison
    report_df = create_metrics_report(all_metrics)
    
    # Sauvegarder le rapport si nécessaire
    if save_dir:
        # Sauvegarder en CSV
        report_path = os.path.join(save_dir, "model_comparison.csv")
        report_df.to_csv(report_path, index=True)
        
        # Générer des visualisations de comparaison
        if save_visualizations and len(predictions) > 1:
            from forestgaps_dl.inference.utils.visualization import (
                visualize_comparison,
                create_evaluation_plot
            )
            
            # Préparer les données pour la comparaison visuelle
            pred_list = list(predictions.values())
            model_names = list(predictions.keys())
            
            # Visualiser la comparaison des prédictions
            fig_comp = visualize_comparison(
                predictions=pred_list,
                labels=model_names,
                ground_truth=ground_truth,
                title="Comparaison des modèles",
                save_path=os.path.join(save_dir, "model_comparison.png")
            )
            plt.close(fig_comp)
            
            # Visualiser les métriques clés
            key_metrics = {
                name: {
                    metric: all_metrics[name][metric]
                    for metric in ["accuracy", "precision", "recall", "f1_score", "iou", "object_f1"]
                    if metric in all_metrics[name]
                }
                for name in all_metrics
            }
            
            fig_metrics = create_evaluation_plot(
                metrics=key_metrics,
                title="Comparaison des performances",
                save_path=os.path.join(save_dir, "metrics_comparison.png")
            )
            plt.close(fig_metrics)
    
    return report_df

def batch_evaluate(
    prediction_dir: str,
    ground_truth_dir: str,
    pattern: str = "*.tif",
    threshold: float = 0.5,
    model_name: str = "model",
    save_dir: Optional[str] = None,
    recursive: bool = False
) -> Dict[str, Any]:
    """
    Évalue un ensemble de prédictions par rapport aux vérités terrain.
    
    Args:
        prediction_dir: Répertoire contenant les prédictions
        ground_truth_dir: Répertoire contenant les vérités terrain
        pattern: Motif de filtrage des fichiers
        threshold: Seuil pour binariser les probabilités
        model_name: Nom du modèle
        save_dir: Répertoire où sauvegarder les résultats
        recursive: Chercher récursivement dans les sous-répertoires
        
    Returns:
        Dictionnaire contenant les métriques moyennes et par fichier
    """
    import glob
    
    # Trouver les fichiers de vérité terrain
    if recursive:
        search_path = os.path.join(ground_truth_dir, "**", pattern)
        truth_files = glob.glob(search_path, recursive=True)
    else:
        search_path = os.path.join(ground_truth_dir, pattern)
        truth_files = glob.glob(search_path)
    
    if not truth_files:
        raise ValueError(f"Aucun fichier de vérité terrain trouvé dans {ground_truth_dir} avec le motif {pattern}")
    
    # Initialiser les résultats
    all_metrics = {}
    aggregated_metrics = {}
    files_evaluated = 0
    
    # Évaluer chaque fichier
    for truth_file in truth_files:
        # Déduire le nom du fichier de prédiction correspondant
        filename = os.path.basename(truth_file)
        pred_file = os.path.join(prediction_dir, filename)
        
        # Vérifier si le fichier de prédiction existe
        if not os.path.exists(pred_file):
            logger.warning(f"Fichier de prédiction manquant: {pred_file}")
            continue
        
        # Charger les fichiers
        try:
            truth_data, _ = load_raster(truth_file)
            pred_data, _ = load_raster(pred_file)
            
            # Vérifier que les dimensions correspondent
            if truth_data.shape != pred_data.shape:
                logger.warning(f"Dimensions non correspondantes pour {filename}: vérité {truth_data.shape}, prédiction {pred_data.shape}")
                continue
            
            # Créer un sous-répertoire pour ce fichier si nécessaire
            file_dir = None
            if save_dir:
                file_dir = os.path.join(save_dir, os.path.splitext(filename)[0])
                os.makedirs(file_dir, exist_ok=True)
            
            # Évaluer la prédiction
            metrics = evaluate_prediction(
                prediction=pred_data,
                ground_truth=truth_data,
                threshold=threshold,
                model_name=model_name,
                save_dir=file_dir,
                save_visualizations=True
            )
            
            # Enregistrer les métriques
            all_metrics[filename] = metrics
            files_evaluated += 1
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation de {filename}: {str(e)}")
    
    if files_evaluated == 0:
        logger.warning("Aucun fichier n'a pu être évalué.")
        return {"all_metrics": {}, "aggregated_metrics": {}}
    
    # Agréger les métriques
    # Trouver toutes les clés de métriques communes
    common_metrics = set.intersection(
        *[set(metrics.keys()) for metrics in all_metrics.values()]
    )
    
    # Exclure les métriques non numériques
    common_metrics = {m for m in common_metrics if all(
        isinstance(metrics[m], (int, float, np.number)) 
        for metrics in all_metrics.values()
    )}
    
    # Calculer la moyenne, l'écart-type, le min et le max pour chaque métrique
    for metric in common_metrics:
        values = [metrics[metric] for metrics in all_metrics.values()]
        aggregated_metrics[f"{metric}_mean"] = float(np.mean(values))
        aggregated_metrics[f"{metric}_std"] = float(np.std(values))
        aggregated_metrics[f"{metric}_min"] = float(np.min(values))
        aggregated_metrics[f"{metric}_max"] = float(np.max(values))
    
    # Ajouter des informations supplémentaires
    aggregated_metrics["num_files"] = files_evaluated
    aggregated_metrics["model_name"] = model_name
    aggregated_metrics["threshold"] = threshold
    
    # Sauvegarder les résultats si nécessaire
    if save_dir:
        # Sauvegarder les métriques agrégées
        agg_path = os.path.join(save_dir, f"aggregated_metrics_{model_name}.json")
        with open(agg_path, 'w') as f:
            json.dump(aggregated_metrics, f, indent=2)
        
        # Sauvegarder toutes les métriques dans un fichier CSV
        all_metrics_df = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in all_metrics.items()})
        all_metrics_df.to_csv(os.path.join(save_dir, f"all_metrics_{model_name}.csv"))
    
    return {
        "all_metrics": all_metrics,
        "aggregated_metrics": aggregated_metrics
    }

def find_best_model(
    models_dir: str,
    validation_dir: str,
    pattern: str = "*.tif",
    metric: str = "f1_score",
    save_report: bool = True,
    report_path: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Trouve le meilleur modèle en évaluant sur un ensemble de validation.
    
    Args:
        models_dir: Répertoire contenant les prédictions des différents modèles
        validation_dir: Répertoire contenant les vérités terrain de validation
        pattern: Motif de filtrage des fichiers
        metric: Métrique à optimiser
        save_report: Sauvegarder un rapport de comparaison
        report_path: Chemin où sauvegarder le rapport
        
    Returns:
        Tuple contenant le nom du meilleur modèle et ses métriques
    """
    import glob
    
    # Trouver les sous-répertoires (un par modèle)
    model_dirs = [d for d in os.listdir(models_dir) 
                  if os.path.isdir(os.path.join(models_dir, d))]
    
    if not model_dirs:
        raise ValueError(f"Aucun répertoire de modèle trouvé dans {models_dir}")
    
    # Évaluer chaque modèle
    model_metrics = {}
    
    for model_name in model_dirs:
        model_dir = os.path.join(models_dir, model_name)
        
        # Créer un répertoire de sortie pour les résultats si nécessaire
        save_dir = None
        if report_path:
            save_dir = os.path.join(report_path, model_name)
            os.makedirs(save_dir, exist_ok=True)
        
        # Évaluer le modèle
        result = batch_evaluate(
            prediction_dir=model_dir,
            ground_truth_dir=validation_dir,
            pattern=pattern,
            model_name=model_name,
            save_dir=save_dir
        )
        
        # Stocker les métriques agrégées
        model_metrics[model_name] = result["aggregated_metrics"]
    
    # Trouver le meilleur modèle selon la métrique choisie
    target_metric = f"{metric}_mean"
    best_model = max(model_metrics.keys(), 
                     key=lambda m: model_metrics[m].get(target_metric, 0))
    
    best_metrics = model_metrics[best_model]
    
    # Créer un rapport de comparaison
    comparison = {
        "models_compared": list(model_metrics.keys()),
        "best_model": best_model,
        "target_metric": metric,
        "best_score": best_metrics.get(target_metric, 0),
        "all_models_scores": {
            m: metrics.get(target_metric, 0) for m, metrics in model_metrics.items()
        }
    }
    
    # Sauvegarder le rapport si demandé
    if save_report and report_path:
        # Créer un DataFrame pour la comparaison
        metrics_keys = set()
        for model_name, metrics in model_metrics.items():
            metrics_keys.update(metrics.keys())
        
        # Ne garder que les métriques communes à tous les modèles
        common_keys = metrics_keys.intersection(*[set(metrics.keys()) for metrics in model_metrics.values()])
        
        # Créer le DataFrame
        comparison_df = pd.DataFrame(index=common_keys)
        for model_name, metrics in model_metrics.items():
            comparison_df[model_name] = pd.Series({k: metrics.get(k, np.nan) for k in common_keys})
        
        # Sauvegarder
        comparison_df.to_csv(os.path.join(report_path, "models_comparison.csv"))
        
        # Sauvegarder également un récapitulatif au format JSON
        with open(os.path.join(report_path, "best_model_summary.json"), 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Créer un graphique de comparaison
        from forestgaps_dl.inference.utils.visualization import create_evaluation_plot
        
        # Extraire les métriques clés pour le graphique
        key_metrics = ["accuracy_mean", "precision_mean", "recall_mean", "f1_score_mean", "iou_mean"]
        key_metrics = [m for m in key_metrics if all(m in metrics for metrics in model_metrics.values())]
        
        if key_metrics:
            # Préparer les données
            plot_data = {
                model_name: {
                    k.replace("_mean", ""): v[k] for k in key_metrics if k in v
                }
                for model_name, v in model_metrics.items()
            }
            
            # Créer le graphique
            fig = create_evaluation_plot(
                metrics=plot_data,
                title="Comparaison des modèles",
                save_path=os.path.join(report_path, "models_comparison.png")
            )
            plt.close(fig)
    
    return best_model, best_metrics 