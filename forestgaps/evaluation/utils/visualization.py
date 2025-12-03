"""
Module de visualisation pour l'évaluation.

Fournit des fonctions pour visualiser les résultats d'évaluation,
comparaisons de modèles, et métriques.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List


def visualize_metrics(
    metrics: Dict[str, float],
    title: str = "Métriques d'évaluation",
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Visualise les métriques d'évaluation sous forme de graphique.

    Args:
        metrics: Dictionnaire de métriques {nom: valeur}
        title: Titre du graphique
        save_path: Chemin pour sauvegarder la figure (optionnel)
        **kwargs: Arguments supplémentaires pour matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    ax.barh(metric_names, metric_values)
    ax.set_xlabel('Valeur')
    ax.set_title(title)
    ax.set_xlim(0, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_comparison(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    title: str = "Comparaison prédiction vs vérité terrain",
    save_path: Optional[str] = None,
    **kwargs
) -> None:
    """
    Visualise la comparaison entre prédictions et vérité terrain.

    Args:
        predictions: Prédictions du modèle
        ground_truth: Vérité terrain
        title: Titre de la visualisation
        save_path: Chemin pour sauvegarder la figure (optionnel)
        **kwargs: Arguments supplémentaires pour matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth
    axes[0].imshow(ground_truth, cmap='gray')
    axes[0].set_title('Vérité terrain')
    axes[0].axis('off')

    # Predictions
    axes[1].imshow(predictions, cmap='gray')
    axes[1].set_title('Prédictions')
    axes[1].axis('off')

    # Difference
    diff = np.abs(ground_truth.astype(float) - predictions.astype(float))
    axes[2].imshow(diff, cmap='Reds')
    axes[2].set_title('Différence')
    axes[2].axis('off')

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> str:
    """
    Crée un tableau formaté des métriques.

    Args:
        metrics_dict: Dictionnaire de métriques {modèle: {métrique: valeur}}
        save_path: Chemin pour sauvegarder le tableau (optionnel)

    Returns:
        Tableau formaté en markdown
    """
    import pandas as pd

    # Créer DataFrame
    df = pd.DataFrame(metrics_dict).T

    # Formater en markdown
    table_md = df.to_markdown()

    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_md)

    return table_md


__all__ = [
    "visualize_metrics",
    "visualize_comparison",
    "create_metrics_table"
]
