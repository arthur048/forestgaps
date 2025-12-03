"""
Wrapper pour les métriques d'évaluation.

Ce module ré-exporte les fonctions de métriques depuis le module parent
pour maintenir la compatibilité avec les imports existants.
"""

from ..metrics import (
    compute_confusion_matrix,
    compute_all_metrics,
    create_threshold_metrics,
    confusion_matrix_metrics,
    probability_metrics,
    object_level_metrics,
    gap_size_metrics,
    compute_roc_curve,
    compute_precision_recall_curve,
    find_optimal_threshold,
    create_metrics_report
)

# Aliases pour compatibilité
calculate_confusion_matrix = compute_confusion_matrix
calculate_metrics = compute_all_metrics
calculate_threshold_metrics = create_threshold_metrics

__all__ = [
    # Fonctions originales
    "compute_confusion_matrix",
    "compute_all_metrics",
    "create_threshold_metrics",
    "confusion_matrix_metrics",
    "probability_metrics",
    "object_level_metrics",
    "gap_size_metrics",
    "compute_roc_curve",
    "compute_precision_recall_curve",
    "find_optimal_threshold",
    "create_metrics_report",

    # Aliases de compatibilité
    "calculate_confusion_matrix",
    "calculate_metrics",
    "calculate_threshold_metrics"
]
