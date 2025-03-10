"""
Métriques d'évaluation pour la détection des trouées forestières.

Ce module fournit un ensemble de métriques pour évaluer les performances
des modèles de détection de trouées, y compris des métriques binaires,
des métriques basées sur les objets, et des métriques spatiales.
"""

import os
import logging
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from scipy import ndimage
from typing import Dict, Any, Tuple, List, Optional, Union, Set, Callable

# Configuration du logging
logger = logging.getLogger(__name__)

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de confusion.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_pred: Prédictions (binaire)
        
    Returns:
        Matrice de confusion [[TN, FP], [FN, TP]]
    """
    # Convertir en binaire si nécessaire
    y_true_bin = y_true.astype(bool)
    y_pred_bin = y_pred.astype(bool)
    
    # Calculer les éléments de la matrice
    tn = np.sum(~y_pred_bin & ~y_true_bin)
    fp = np.sum(y_pred_bin & ~y_true_bin)
    fn = np.sum(~y_pred_bin & y_true_bin)
    tp = np.sum(y_pred_bin & y_true_bin)
    
    return np.array([[tn, fp], [fn, tp]])

def confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcule diverses métriques à partir de la matrice de confusion.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_pred: Prédictions (binaire)
        
    Returns:
        Dictionnaire contenant diverses métriques
    """
    # Calculer la matrice de confusion
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    # Éviter les divisions par zéro
    eps = 1e-8
    
    # Calculer les métriques
    metrics = {}
    
    # Métriques de base
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn + eps)
    metrics["precision"] = tp / (tp + fp + eps)
    metrics["recall"] = tp / (tp + fn + eps)
    metrics["f1_score"] = 2 * tp / (2 * tp + fp + fn + eps)
    
    # Métriques supplémentaires
    metrics["specificity"] = tn / (tn + fp + eps)
    metrics["iou"] = tp / (tp + fp + fn + eps)  # Intersection over Union (Jaccard)
    metrics["dice"] = 2 * tp / (2 * tp + fp + fn + eps)  # Dice coefficient (same as F1)
    
    # Métriques avancées
    metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2
    metrics["prevalence"] = (tp + fn) / (tp + tn + fp + fn + eps)
    metrics["false_discovery_rate"] = fp / (tp + fp + eps)
    metrics["false_negative_rate"] = fn / (tp + fn + eps)
    metrics["false_positive_rate"] = fp / (tn + fp + eps)
    metrics["negative_predictive_value"] = tn / (tn + fn + eps)
    
    # Matthews Correlation Coefficient (MCC)
    metrics["mcc"] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    
    return metrics

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    object_level: bool = True,
    size_metrics: bool = True
) -> Dict[str, float]:
    """
    Calcule toutes les métriques disponibles.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_pred: Prédictions (binaire)
        y_prob: Probabilités prédites (optionnel)
        object_level: Inclure les métriques au niveau des objets
        size_metrics: Inclure les métriques de taille
        
    Returns:
        Dictionnaire contenant toutes les métriques
    """
    # Métriques de la matrice de confusion
    metrics = confusion_matrix_metrics(y_true, y_pred)
    
    # Métriques basées sur la probabilité si disponible
    if y_prob is not None:
        prob_metrics = probability_metrics(y_true, y_prob)
        metrics.update(prob_metrics)
    
    # Métriques au niveau des objets
    if object_level:
        obj_metrics = object_level_metrics(y_true, y_pred)
        metrics.update(obj_metrics)
    
    # Métriques de taille
    if size_metrics:
        size_met = gap_size_metrics(y_true, y_pred)
        metrics.update(size_met)
    
    return metrics

def probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calcule des métriques basées sur les probabilités prédites.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_prob: Probabilités prédites
        
    Returns:
        Dictionnaire contenant les métriques basées sur les probabilités
    """
    # Convertir en vecteurs 1D si nécessaire
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()
    
    metrics = {}
    
    # Aire sous la courbe ROC
    try:
        metrics["roc_auc"] = skm.roc_auc_score(y_true_flat, y_prob_flat)
    except ValueError as e:
        logger.warning(f"Impossible de calculer ROC AUC: {str(e)}")
        metrics["roc_auc"] = np.nan
    
    # Aire sous la courbe précision-rappel
    try:
        metrics["pr_auc"] = skm.average_precision_score(y_true_flat, y_prob_flat)
    except ValueError as e:
        logger.warning(f"Impossible de calculer PR AUC: {str(e)}")
        metrics["pr_auc"] = np.nan
    
    # Log loss (cross-entropy)
    try:
        metrics["log_loss"] = skm.log_loss(y_true_flat, y_prob_flat)
    except ValueError as e:
        logger.warning(f"Impossible de calculer Log Loss: {str(e)}")
        metrics["log_loss"] = np.nan
    
    # Brier score (erreur quadratique moyenne)
    metrics["brier_score"] = skm.brier_score_loss(y_true_flat, y_prob_flat)
    
    return metrics

def label_objects(binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Étiquette les objets connectés dans un masque binaire.
    
    Args:
        binary_mask: Masque binaire
        
    Returns:
        Tuple contenant le masque étiqueté et le nombre d'objets
    """
    # Utiliser l'étiquetage par composantes connexes
    labeled_mask, num_objects = ndimage.label(binary_mask)
    return labeled_mask, num_objects

def object_level_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcule des métriques au niveau des objets.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_pred: Prédictions (binaire)
        
    Returns:
        Dictionnaire contenant les métriques au niveau des objets
    """
    # Étiqueter les objets dans les masques
    y_true_labeled, num_true = label_objects(y_true)
    y_pred_labeled, num_pred = label_objects(y_pred)
    
    metrics = {}
    
    # Nombre d'objets
    metrics["num_true_objects"] = num_true
    metrics["num_pred_objects"] = num_pred
    metrics["object_count_diff"] = num_pred - num_true
    metrics["object_count_ratio"] = num_pred / max(1, num_true)
    
    # Métriques de détection d'objets
    if num_true > 0 and num_pred > 0:
        # Initialiser les compteurs
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Pour chaque objet de la vérité terrain, vérifier s'il est détecté
        for i in range(1, num_true + 1):
            # Créer un masque pour l'objet i
            true_obj = y_true_labeled == i
            # Vérifier s'il y a une intersection avec un objet prédit
            if np.any(true_obj & y_pred):
                true_positives += 1
            else:
                false_negatives += 1
        
        # Pour chaque objet prédit, vérifier s'il est un faux positif
        for i in range(1, num_pred + 1):
            # Créer un masque pour l'objet i
            pred_obj = y_pred_labeled == i
            # Vérifier s'il y a une intersection avec un objet de la vérité terrain
            if not np.any(pred_obj & y_true):
                false_positives += 1
        
        # Calculer les métriques
        precision = true_positives / max(1, true_positives + false_positives)
        recall = true_positives / max(1, true_positives + false_negatives)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        
        metrics["object_precision"] = precision
        metrics["object_recall"] = recall
        metrics["object_f1"] = f1
    else:
        # Si l'un des ensembles est vide, les métriques sont indéfinies
        metrics["object_precision"] = 0 if num_pred > 0 else 1
        metrics["object_recall"] = 0 if num_true > 0 else 1
        metrics["object_f1"] = 0
    
    return metrics

def gap_size_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcule des métriques liées à la taille des trouées.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_pred: Prédictions (binaire)
        
    Returns:
        Dictionnaire contenant les métriques de taille
    """
    # Étiqueter les objets
    y_true_labeled, num_true = label_objects(y_true)
    y_pred_labeled, num_pred = label_objects(y_pred)
    
    metrics = {}
    
    # Si aucun objet, retourner des métriques par défaut
    if num_true == 0 or num_pred == 0:
        metrics["mean_size_error"] = np.nan
        metrics["median_size_error"] = np.nan
        metrics["size_correlation"] = np.nan
        metrics["size_rmse"] = np.nan
        metrics["size_mae"] = np.nan
        return metrics
    
    # Calculer les tailles des objets de la vérité terrain
    true_sizes = np.zeros(num_true)
    for i in range(1, num_true + 1):
        true_sizes[i-1] = np.sum(y_true_labeled == i)
    
    # Calculer les tailles des objets prédits
    pred_sizes = np.zeros(num_pred)
    for i in range(1, num_pred + 1):
        pred_sizes[i-1] = np.sum(y_pred_labeled == i)
    
    # Métriques de base sur les tailles
    metrics["true_gap_total_area"] = np.sum(y_true)
    metrics["pred_gap_total_area"] = np.sum(y_pred)
    metrics["true_gap_mean_size"] = np.mean(true_sizes)
    metrics["pred_gap_mean_size"] = np.mean(pred_sizes)
    metrics["true_gap_median_size"] = np.median(true_sizes)
    metrics["pred_gap_median_size"] = np.median(pred_sizes)
    metrics["true_gap_std_size"] = np.std(true_sizes)
    metrics["pred_gap_std_size"] = np.std(pred_sizes)
    metrics["true_gap_min_size"] = np.min(true_sizes)
    metrics["pred_gap_min_size"] = np.min(pred_sizes)
    metrics["true_gap_max_size"] = np.max(true_sizes)
    metrics["pred_gap_max_size"] = np.max(pred_sizes)
    
    # Erreur relative totale des surfaces
    total_area_diff = np.sum(y_pred) - np.sum(y_true)
    metrics["total_area_diff"] = total_area_diff
    metrics["relative_area_error"] = total_area_diff / max(1, np.sum(y_true))
    
    # Associer les objets de la vérité terrain aux objets prédits
    # pour calculer des métriques d'erreur de taille plus précises
    matched_sizes = []
    for i in range(1, num_true + 1):
        true_obj = y_true_labeled == i
        true_size = np.sum(true_obj)
        
        # Trouver l'objet prédit avec la plus grande intersection
        max_overlap = 0
        max_overlap_size = 0
        
        for j in range(1, num_pred + 1):
            pred_obj = y_pred_labeled == j
            overlap = np.sum(true_obj & pred_obj)
            
            if overlap > max_overlap:
                max_overlap = overlap
                max_overlap_size = np.sum(pred_obj)
        
        # Si un objet prédit correspond, ajouter la paire de tailles
        if max_overlap > 0:
            matched_sizes.append((true_size, max_overlap_size))
    
    # Si des objets ont été associés, calculer les métriques d'erreur
    if matched_sizes:
        true_matched_sizes = np.array([s[0] for s in matched_sizes])
        pred_matched_sizes = np.array([s[1] for s in matched_sizes])
        
        # Erreur absolue et relative de taille
        size_errors = pred_matched_sizes - true_matched_sizes
        rel_size_errors = size_errors / np.maximum(1, true_matched_sizes)
        
        metrics["mean_size_error"] = np.mean(size_errors)
        metrics["median_size_error"] = np.median(size_errors)
        metrics["mean_relative_size_error"] = np.mean(rel_size_errors)
        metrics["median_relative_size_error"] = np.median(rel_size_errors)
        metrics["size_rmse"] = np.sqrt(np.mean(np.square(size_errors)))
        metrics["size_mae"] = np.mean(np.abs(size_errors))
        
        # Corrélation entre les tailles
        if len(true_matched_sizes) > 1:
            try:
                metrics["size_correlation"] = np.corrcoef(true_matched_sizes, pred_matched_sizes)[0, 1]
            except:
                metrics["size_correlation"] = np.nan
        else:
            metrics["size_correlation"] = np.nan
    else:
        # Si aucun objet n'a été associé
        metrics["mean_size_error"] = np.nan
        metrics["median_size_error"] = np.nan
        metrics["mean_relative_size_error"] = np.nan
        metrics["median_relative_size_error"] = np.nan
        metrics["size_rmse"] = np.nan
        metrics["size_mae"] = np.nan
        metrics["size_correlation"] = np.nan
    
    return metrics

def compute_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule la courbe ROC.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_prob: Probabilités prédites
        
    Returns:
        Tuple contenant les faux positifs, vrais positifs et seuils
    """
    # Aplatir les tableaux
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()
    
    # Calculer la courbe ROC
    fpr, tpr, thresholds = skm.roc_curve(y_true_flat, y_prob_flat)
    
    return fpr, tpr, thresholds

def compute_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule la courbe précision-rappel.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_prob: Probabilités prédites
        
    Returns:
        Tuple contenant la précision, le rappel et les seuils
    """
    # Aplatir les tableaux
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()
    
    # Calculer la courbe précision-rappel
    precision, recall, thresholds = skm.precision_recall_curve(y_true_flat, y_prob_flat)
    
    return precision, recall, thresholds

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "f1",
    num_thresholds: int = 100
) -> Tuple[float, float]:
    """
    Trouve le seuil optimal selon un critère donné.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_prob: Probabilités prédites
        criterion: Critère d'optimisation ('f1', 'iou', 'accuracy', 'balanced_accuracy')
        num_thresholds: Nombre de seuils à tester
        
    Returns:
        Tuple contenant le seuil optimal et la valeur de la métrique correspondante
    """
    # Aplatir les tableaux
    y_true_flat = y_true.flatten()
    y_prob_flat = y_prob.flatten()
    
    # Définir la fonction de critère
    def get_criterion_value(threshold):
        y_pred = (y_prob_flat >= threshold).astype(int)
        
        if criterion == "f1":
            return skm.f1_score(y_true_flat, y_pred)
        elif criterion == "iou":
            return skm.jaccard_score(y_true_flat, y_pred)
        elif criterion == "accuracy":
            return skm.accuracy_score(y_true_flat, y_pred)
        elif criterion == "balanced_accuracy":
            return skm.balanced_accuracy_score(y_true_flat, y_pred)
        elif criterion == "mcc":
            return skm.matthews_corrcoef(y_true_flat, y_pred)
        else:
            raise ValueError(f"Critère non reconnu: {criterion}")
    
    # Tester différents seuils
    thresholds = np.linspace(0, 1, num_thresholds)[1:-1]  # Éviter 0 et 1
    scores = [get_criterion_value(t) for t in thresholds]
    
    # Trouver le seuil optimal
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return best_threshold, best_score

def create_metrics_report(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Crée un rapport de métriques sous forme de DataFrame.
    
    Args:
        all_metrics: Dictionnaire de dictionnaires de métriques
        
    Returns:
        DataFrame contenant toutes les métriques
    """
    # Créer un DataFrame vide
    df = pd.DataFrame()
    
    # Remplir le DataFrame
    for model_name, metrics in all_metrics.items():
        df[model_name] = pd.Series(metrics)
    
    # Trier les métriques par nom
    df = df.sort_index()
    
    return df

def create_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
    metrics: List[str] = ["accuracy", "precision", "recall", "f1_score", "iou"]
) -> pd.DataFrame:
    """
    Calcule les métriques pour différents seuils.
    
    Args:
        y_true: Vérité terrain (binaire)
        y_prob: Probabilités prédites
        thresholds: Liste des seuils à tester
        metrics: Liste des métriques à calculer
        
    Returns:
        DataFrame contenant les métriques pour chaque seuil
    """
    # Préparer le DataFrame
    results = {"threshold": thresholds}
    for metric in metrics:
        results[metric] = []
    
    # Calculer les métriques pour chaque seuil
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculer les métriques pour ce seuil
        for metric in metrics:
            if metric in ["accuracy", "precision", "recall", "f1_score"]:
                # Utiliser scikit-learn pour ces métriques
                func = getattr(skm, f"{metric}_score")
                value = func(y_true.flatten(), y_pred.flatten())
            elif metric == "iou":
                # Jaccard/IoU
                value = skm.jaccard_score(y_true.flatten(), y_pred.flatten())
            elif metric == "balanced_accuracy":
                value = skm.balanced_accuracy_score(y_true.flatten(), y_pred.flatten())
            elif metric == "mcc":
                value = skm.matthews_corrcoef(y_true.flatten(), y_pred.flatten())
            else:
                # Calculer d'autres métriques si nécessaire
                cm_metrics = confusion_matrix_metrics(y_true, y_pred)
                value = cm_metrics.get(metric, np.nan)
            
            results[metric].append(value)
    
    # Créer le DataFrame
    df = pd.DataFrame(results)
    
    return df 