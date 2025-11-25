"""
Fonctions de création de graphiques pour ForestGaps.

Ce module fournit des fonctions pour créer des graphiques et visualisations
des données et résultats du workflow ForestGaps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path


def visualize_metrics_evolution(tracker, save_path=None):
    """
    Visualise l'évolution des métriques d'entraînement et de validation.
    
    Args:
        tracker: Objet LossTracker contenant l'historique des métriques.
        save_path (str, optional): Chemin pour sauvegarder la figure.
        
    Returns:
        matplotlib.figure.Figure: Figure matplotlib.
    """
    # Créer une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Définir les métriques à visualiser
    metrics_to_plot = [
        ('loss', 'Perte', 0),
        ('iou', 'IoU', 1),
        ('f1_score', 'F1-Score', 2),
        ('accuracy', 'Accuracy', 3)
    ]
    
    # Tracer chaque métrique
    for metric_name, metric_label, ax_idx in metrics_to_plot:
        ax = axes[ax_idx]
        
        # Récupérer les données
        train_values = getattr(tracker, f'train_{metric_name}', [])
        val_values = getattr(tracker, f'val_{metric_name}', [])
        epochs = list(range(1, len(train_values) + 1))
        
        # Tracer les courbes
        ax.plot(epochs, train_values, 'b-', label=f'Entraînement')
        ax.plot(epochs, val_values, 'r-', label=f'Validation')
        
        # Ajouter les annotations
        ax.set_title(f'Évolution de {metric_label}')
        ax.set_xlabel('Époque')
        ax.set_ylabel(metric_label)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Ajouter des annotations pour les meilleures valeurs
        if val_values:
            best_epoch = np.argmin(val_values) + 1 if metric_name == 'loss' else np.argmax(val_values) + 1
            best_value = np.min(val_values) if metric_name == 'loss' else np.max(val_values)
            
            ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
            ax.text(best_epoch + 0.1, best_value, f'Meilleur: {best_value:.4f} (époque {best_epoch})',
                    verticalalignment='center')
    
    plt.tight_layout()
    
    # Sauvegarder la figure si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique d'évolution des métriques sauvegardé dans {save_path}")
    
    return fig


def visualize_metrics_by_threshold(metrics, title="Métriques par seuil", save_path=None):
    """
    Visualise les métriques en fonction du seuil de hauteur.
    
    Args:
        metrics (dict): Dictionnaire de métriques retourné par SegmentationMetrics.compute()
        title (str): Titre de la figure
        save_path (str, optional): Chemin pour sauvegarder la figure
        
    Returns:
        matplotlib.figure.Figure: Figure matplotlib.
    """
    if 'threshold_metrics' not in metrics or not metrics['threshold_metrics']:
        print("Aucune métrique par seuil disponible")
        return None
    
    # Extraire les données
    thresholds = sorted(metrics['threshold_metrics'].keys())
    
    # Métriques à visualiser
    metric_configs = [
        ('accuracy', 'Accuracy', 'tab:blue'),
        ('precision', 'Precision', 'tab:orange'),
        ('recall', 'Recall', 'tab:green'),
        ('f1_score', 'F1-Score', 'tab:red'),
        ('iou', 'IoU', 'tab:purple')
    ]
    
    # Créer la figure
    fig = plt.figure(figsize=(14, 10))
    
    # Tracer chaque métrique
    for metric_key, metric_name, color in metric_configs:
        values = [metrics['threshold_metrics'][t][metric_key] for t in thresholds]
        plt.plot(thresholds, values, '-o', color=color, label=metric_name)
    
    plt.xlabel('Seuil de hauteur (m)')
    plt.ylabel('Valeur')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    # Ajouter des annotations pour chaque point
    for metric_key, metric_name, color in metric_configs:
        for i, threshold in enumerate(thresholds):
            value = metrics['threshold_metrics'][threshold][metric_key]
            plt.annotate(
                f'{value:.3f}',
                (threshold, value),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                color=color
            )
    
    # Ajuster les limites de l'axe y
    plt.ylim(0, 1.1)
    
    # Sauvegarder la figure si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique des métriques par seuil sauvegardé dans {save_path}")
    
    return fig


def visualize_confusion_matrix(confusion_data, title="Matrice de confusion", save_path=None):
    """
    Visualise la matrice de confusion.
    
    Args:
        confusion_data (dict): Dictionnaire contenant les données de confusion
        title (str): Titre de la figure
        save_path (str, optional): Chemin pour sauvegarder la figure
        
    Returns:
        matplotlib.figure.Figure: Figure matplotlib.
    """
    # Extraire les données
    matrix = confusion_data['matrix']
    percentages = confusion_data.get('percentages', {})
    total_pixels = confusion_data.get('total_pixels', 0)
    
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Matrice de confusion en valeurs absolues
    cm = np.array([
        [matrix['tn'], matrix['fp']],
        [matrix['fn'], matrix['tp']]
    ])
    
    # Matrice de confusion en pourcentages
    cm_percent = np.array([
        [percentages.get('tn', 0), percentages.get('fp', 0)],
        [percentages.get('fn', 0), percentages.get('tp', 0)]
    ]) * 100
    
    # Tracer la matrice en valeurs absolues
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1,
                xticklabels=["Non-trouée", "Trouée"],
                yticklabels=["Non-trouée", "Trouée"])
    ax1.set_title(f"{title} (valeurs absolues)")
    ax1.set_xlabel("Prédiction")
    ax1.set_ylabel("Réalité")
    
    # Tracer la matrice en pourcentages
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax2,
                xticklabels=["Non-trouée", "Trouée"],
                yticklabels=["Non-trouée", "Trouée"])
    ax2.set_title(f"{title} (pourcentages)")
    ax2.set_xlabel("Prédiction")
    ax2.set_ylabel("Réalité")
    
    # Ajouter des informations supplémentaires
    plt.figtext(0.5, 0.01, f"Total des pixels: {total_pixels:,}", ha="center", fontsize=12)
    
    plt.tight_layout()
    
    # Sauvegarder la figure si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matrice de confusion sauvegardée dans {save_path}")
    
    return fig


def create_metrics_tables(metrics, title="Métriques d'évaluation"):
    """
    Crée un tableau de métriques formaté pour l'affichage.
    
    Args:
        metrics (dict): Dictionnaire de métriques
        title (str): Titre du tableau
        
    Returns:
        str: Tableau formaté
    """
    def format_num(num):
        """Formate un nombre pour l'affichage."""
        if isinstance(num, (int, float)):
            return f"{num:.4f}"
        return str(num)
    
    # Créer l'en-tête du tableau
    table = f"\n{title}\n"
    table += "=" * len(title) + "\n\n"
    
    # Métriques globales
    table += "Métriques globales:\n"
    table += "-" * 20 + "\n"
    
    for metric_name, value in metrics.items():
        if metric_name != 'threshold_metrics' and metric_name != 'confusion_data':
            table += f"{metric_name.capitalize():15}: {format_num(value)}\n"
    
    # Métriques par seuil
    if 'threshold_metrics' in metrics and metrics['threshold_metrics']:
        table += "\nMétriques par seuil de hauteur:\n"
        table += "-" * 30 + "\n"
        
        # En-tête du tableau des seuils
        thresholds = sorted(metrics['threshold_metrics'].keys())
        header = f"{'Métrique':15} | " + " | ".join([f"{t:^8}" for t in thresholds]) + " |"
        table += header + "\n"
        table += "-" * len(header) + "\n"
        
        # Métriques à afficher
        metric_names = ['iou', 'accuracy', 'precision', 'recall', 'f1_score']
        
        # Lignes du tableau
        for metric_name in metric_names:
            row = f"{metric_name.capitalize():15} | "
            for threshold in thresholds:
                value = metrics['threshold_metrics'][threshold].get(metric_name, 0)
                row += f"{format_num(value):^8} | "
            table += row + "\n"
    
    return table


def plot_segmentation_results(input_dsm, target_mask, predicted_mask, 
                              threshold=None, save_path=None):
    """
    Visualise les résultats de segmentation.
    
    Args:
        input_dsm: DSM d'entrée [height, width] ou [channels, height, width].
        target_mask: Masque cible [height, width] ou [channels, height, width].
        predicted_mask: Masque prédit [height, width] ou [channels, height, width].
        threshold: Valeur de seuil utilisée (pour le titre).
        save_path: Chemin où sauvegarder la figure. Si None, affiche la figure.
    """
    # Normaliser les données pour l'affichage
    def _normalize_for_display(img):
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img[0]  # Prendre le premier canal si besoin
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    input_display = _normalize_for_display(input_dsm)
    target_display = _normalize_for_display(target_mask)
    pred_display = _normalize_for_display(predicted_mask)
    
    # Créer la figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Afficher les images
    axes[0].imshow(input_display, cmap='terrain')
    axes[0].set_title("DSM d'entrée")
    axes[0].axis('off')
    
    axes[1].imshow(target_display, cmap='gray')
    axes[1].set_title("Masque cible")
    axes[1].axis('off')
    
    axes[2].imshow(pred_display, cmap='gray')
    threshold_str = f" (seuil: {threshold:.1f}m)" if threshold is not None else ""
    axes[2].set_title(f"Masque prédit{threshold_str}")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder ou afficher
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_regression_results(input_dsm, target_chm, predicted_chm, 
                           threshold=None, save_path=None):
    """
    Visualise les résultats de régression.
    
    Args:
        input_dsm: DSM d'entrée [height, width] ou [channels, height, width].
        target_chm: CHM cible [height, width] ou [channels, height, width].
        predicted_chm: CHM prédit [height, width] ou [channels, height, width].
        threshold: Valeur de seuil utilisée (pour le titre).
        save_path: Chemin où sauvegarder la figure. Si None, affiche la figure.
    """
    # Normaliser les données pour l'affichage
    def _extract_2d(img):
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img[0]  # Prendre le premier canal si besoin
        return img
    
    input_display = _extract_2d(input_dsm)
    target_display = _extract_2d(target_chm)
    pred_display = _extract_2d(predicted_chm)
    
    # Calculer l'erreur
    error_display = np.abs(target_display - pred_display)
    
    # Créer la figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Afficher les images
    im0 = axes[0].imshow(input_display, cmap='terrain')
    axes[0].set_title("DSM d'entrée")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(target_display, cmap='viridis')
    axes[1].set_title("CHM cible")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(pred_display, cmap='viridis')
    threshold_str = f" (seuil: {threshold:.1f}m)" if threshold is not None else ""
    axes[2].set_title(f"CHM prédit{threshold_str}")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    im3 = axes[3].imshow(error_display, cmap='hot')
    axes[3].set_title("Erreur absolue")
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Sauvegarder ou afficher
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics_by_epoch(metrics_history, metrics_names=None, save_path=None):
    """
    Visualise l'évolution des métriques au cours des époques.
    
    Args:
        metrics_history: Dictionnaire contenant l'historique des métriques.
        metrics_names: Liste des noms des métriques à afficher. Si None, affiche toutes les métriques.
        save_path: Chemin où sauvegarder la figure. Si None, affiche la figure.
    """
    if metrics_names is None:
        metrics_names = list(metrics_history.keys())
    
    # Créer la figure
    fig, axes = plt.subplots(len(metrics_names), 1, figsize=(10, 3 * len(metrics_names)))
    
    # Si une seule métrique, axes n'est pas un itérable
    if len(metrics_names) == 1:
        axes = [axes]
    
    # Afficher chaque métrique
    for ax, metric_name in zip(axes, metrics_names):
        if metric_name in metrics_history:
            history = metrics_history[metric_name]
            epochs = range(1, len(history) + 1)
            
            ax.plot(epochs, history)
            ax.set_title(f"Évolution de {metric_name}")
            ax.set_xlabel("Époque")
            ax.set_ylabel(metric_name)
            ax.grid(True)
    
    plt.tight_layout()
    
    # Sauvegarder ou afficher
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
