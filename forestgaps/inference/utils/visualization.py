"""
Utilitaires de visualisation pour le module d'inférence.

Ce module fournit des fonctions pour visualiser les prédictions, 
les cartes de probabilité et les comparaisons avec les références.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import Dict, Any, Tuple, List, Optional, Union

# Configuration du logging
logger = logging.getLogger(__name__)

def create_colormap(
    name: str = "forestgaps", 
    background_color: Tuple[float, float, float] = (0.95, 0.95, 0.95),
    foreground_color: Tuple[float, float, float] = (0.2, 0.7, 0.3)
) -> mcolors.ListedColormap:
    """
    Crée une colormap personnalisée pour la visualisation des prédictions.
    
    Args:
        name: Nom de la colormap
        background_color: Couleur RGB pour l'arrière-plan (classe 0)
        foreground_color: Couleur RGB pour les trouées (classe 1)
        
    Returns:
        Colormap personnalisée
    """
    colors = [background_color, foreground_color]
    return mcolors.ListedColormap(colors, name=name)

def visualize_prediction(
    prediction: np.ndarray,
    original_data: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Prédiction",
    figsize: Tuple[int, int] = (12, 8),
    alpha: float = 0.7,
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show_colorbar: bool = True
) -> Figure:
    """
    Visualise une prédiction, optionnellement superposée sur les données originales
    ou comparée à la vérité terrain.
    
    Args:
        prediction: Prédiction à visualiser (binaire ou probabilité)
        original_data: Données originales sur lesquelles superposer la prédiction
        ground_truth: Vérité terrain pour comparaison
        title: Titre de la figure
        figsize: Taille de la figure
        alpha: Transparence pour la superposition
        cmap: Colormap à utiliser (si None, utilise la colormap par défaut du projet)
        save_path: Chemin où sauvegarder la figure (optionnel)
        dpi: Résolution de l'image sauvegardée
        show_colorbar: Afficher la barre de couleur
        
    Returns:
        Figure matplotlib
    """
    # Créer une figure
    fig, axes = plt.subplots(1, 1 if ground_truth is None else 3, figsize=figsize)
    
    # Utiliser la colormap par défaut si aucune n'est spécifiée
    if cmap is None:
        cmap = create_colormap()
    
    # Si la prédiction est une carte de probabilité, l'afficher comme telle
    is_probability = not np.array_equal(prediction, prediction.astype(bool))
    
    # Si ground_truth est None, afficher une seule image
    if ground_truth is None:
        ax = axes if isinstance(axes, plt.Axes) else axes[0]
        
        if original_data is not None:
            # Normaliser les données originales pour l'affichage
            if original_data.dtype != np.uint8:
                vmin, vmax = np.nanmin(original_data), np.nanmax(original_data)
                if vmin == vmax:
                    # Éviter la division par zéro
                    normalized = np.zeros_like(original_data)
                else:
                    normalized = (original_data - vmin) / (vmax - vmin)
            else:
                normalized = original_data / 255.0
            
            # Afficher les données originales
            ax.imshow(normalized, cmap='gray', interpolation='nearest')
            
            # Superposer la prédiction
            if is_probability:
                # Pour une carte de probabilité, utiliser une colormap 'hot' avec transparence
                im = ax.imshow(prediction, cmap='hot', alpha=alpha * prediction, 
                              vmin=0, vmax=1, interpolation='nearest')
                if show_colorbar:
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Probabilité de trouée')
            else:
                # Pour une prédiction binaire, utiliser la colormap personnalisée
                # avec masque pour que seules les trouées soient affichées
                mask = prediction.astype(bool)
                overlay = np.zeros((*prediction.shape, 4), dtype=np.float32)
                overlay[..., :3] = mcolors.to_rgb(cmap(1))  # Couleur de la trouée
                overlay[..., 3] = np.where(mask, alpha, 0)  # Alpha: transparent où il n'y a pas de trouée
                ax.imshow(overlay, interpolation='nearest')
        else:
            # Afficher uniquement la prédiction
            if is_probability:
                im = ax.imshow(prediction, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
                if show_colorbar:
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Probabilité de trouée')
            else:
                im = ax.imshow(prediction, cmap=cmap, interpolation='nearest')
                if show_colorbar:
                    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
                    cbar.set_ticklabels(['Non-trouée', 'Trouée'])
        
        ax.set_title(title)
        ax.axis('off')
    
    else:
        # Afficher trois images : données originales, prédiction, vérité terrain
        
        # Première image : données originales ou prédiction
        if original_data is not None:
            # Normaliser les données originales pour l'affichage
            if original_data.dtype != np.uint8:
                vmin, vmax = np.nanmin(original_data), np.nanmax(original_data)
                if vmin == vmax:
                    normalized = np.zeros_like(original_data)
                else:
                    normalized = (original_data - vmin) / (vmax - vmin)
            else:
                normalized = original_data / 255.0
            
            axes[0].imshow(normalized, cmap='gray', interpolation='nearest')
            axes[0].set_title("Données originales")
        else:
            if is_probability:
                im = axes[0].imshow(prediction, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
                if show_colorbar:
                    cbar = plt.colorbar(im, ax=axes[0])
                    cbar.set_label('Probabilité')
            else:
                axes[0].imshow(prediction, cmap=cmap, interpolation='nearest')
            axes[0].set_title("Prédiction")
        
        # Deuxième image : prédiction
        if original_data is not None:
            if is_probability:
                im = axes[1].imshow(prediction, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
                if show_colorbar:
                    cbar = plt.colorbar(im, ax=axes[1])
                    cbar.set_label('Probabilité')
            else:
                axes[1].imshow(prediction, cmap=cmap, interpolation='nearest')
            axes[1].set_title("Prédiction")
        
        # Troisième image : vérité terrain
        axes[-1].imshow(ground_truth, cmap=cmap, interpolation='nearest')
        axes[-1].set_title("Vérité terrain")
        
        # Désactiver les axes
        for ax in axes:
            ax.axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder si nécessaire
    if save_path is not None:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans: {save_path}")
    
    return fig

def visualize_comparison(
    predictions: List[np.ndarray],
    labels: List[str],
    original_data: Optional[np.ndarray] = None,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Comparaison des prédictions",
    figsize: Optional[Tuple[int, int]] = None,
    cmap: Optional[Union[str, mcolors.Colormap]] = None,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure:
    """
    Visualise une comparaison entre plusieurs prédictions.
    
    Args:
        predictions: Liste des prédictions à comparer
        labels: Liste des étiquettes pour chaque prédiction
        original_data: Données originales (optionnel)
        ground_truth: Vérité terrain (optionnel)
        title: Titre de la figure
        figsize: Taille de la figure
        cmap: Colormap à utiliser
        save_path: Chemin où sauvegarder la figure (optionnel)
        dpi: Résolution de l'image sauvegardée
        
    Returns:
        Figure matplotlib
    """
    # Déterminer le nombre d'images à afficher
    n_images = len(predictions)
    if original_data is not None:
        n_images += 1
    if ground_truth is not None:
        n_images += 1
    
    # Déterminer la disposition des sous-figures
    if n_images <= 3:
        n_rows, n_cols = 1, n_images
    else:
        n_cols = min(n_images, 3)
        n_rows = (n_images + n_cols - 1) // n_cols
    
    # Déterminer la taille de la figure si non spécifiée
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    # Créer la figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Convertir axes en tableau 1D pour simplifier l'indexation
    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Utiliser la colormap par défaut si aucune n'est spécifiée
    if cmap is None:
        cmap = create_colormap()
    
    # Remplir la figure
    img_idx = 0
    
    # Afficher les données originales si disponibles
    if original_data is not None:
        # Normaliser les données originales pour l'affichage
        if original_data.dtype != np.uint8:
            vmin, vmax = np.nanmin(original_data), np.nanmax(original_data)
            if vmin == vmax:
                normalized = np.zeros_like(original_data)
            else:
                normalized = (original_data - vmin) / (vmax - vmin)
        else:
            normalized = original_data / 255.0
        
        axes[img_idx].imshow(normalized, cmap='gray', interpolation='nearest')
        axes[img_idx].set_title("Données originales")
        axes[img_idx].axis('off')
        img_idx += 1
    
    # Afficher les prédictions
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        is_probability = not np.array_equal(pred, pred.astype(bool))
        
        if is_probability:
            im = axes[img_idx].imshow(pred, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
            plt.colorbar(im, ax=axes[img_idx])
        else:
            axes[img_idx].imshow(pred, cmap=cmap, interpolation='nearest')
        
        axes[img_idx].set_title(label)
        axes[img_idx].axis('off')
        img_idx += 1
    
    # Afficher la vérité terrain si disponible
    if ground_truth is not None:
        axes[img_idx].imshow(ground_truth, cmap=cmap, interpolation='nearest')
        axes[img_idx].set_title("Vérité terrain")
        axes[img_idx].axis('off')
        img_idx += 1
    
    # Désactiver les axes restants s'il y en a
    for j in range(img_idx, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Sauvegarder si nécessaire
    if save_path is not None:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans: {save_path}")
    
    return fig

def visualize_overlay(
    data: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    title: str = "Superposition DSM et prédiction",
    figsize: Tuple[int, int] = (10, 10),
    prediction_color: Tuple[float, float, float] = (0.2, 0.7, 0.3),
    ground_truth_color: Optional[Tuple[float, float, float]] = (0.9, 0.3, 0.3),
    alpha: float = 0.5,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure:
    """
    Visualise une superposition des données originales et de la prédiction,
    avec optionnellement la vérité terrain pour comparaison.
    
    Args:
        data: Données originales (DSM ou image)
        prediction: Prédiction binaire
        ground_truth: Vérité terrain (optionnel)
        title: Titre de la figure
        figsize: Taille de la figure
        prediction_color: Couleur RGB pour la prédiction
        ground_truth_color: Couleur RGB pour la vérité terrain
        alpha: Transparence des superpositions
        save_path: Chemin où sauvegarder la figure (optionnel)
        dpi: Résolution de l'image sauvegardée
        
    Returns:
        Figure matplotlib
    """
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normaliser les données originales pour l'affichage
    if data.dtype != np.uint8:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        if vmin == vmax:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - vmin) / (vmax - vmin)
    else:
        normalized = data / 255.0
    
    # Afficher les données originales
    ax.imshow(normalized, cmap='gray', interpolation='nearest')
    
    # Créer un masque pour la prédiction
    mask_prediction = np.zeros((*prediction.shape, 4), dtype=np.float32)
    mask_prediction[..., :3] = prediction_color
    mask_prediction[..., 3] = np.where(prediction.astype(bool), alpha, 0)
    
    # Superposer la prédiction
    ax.imshow(mask_prediction, interpolation='nearest')
    
    # Si la vérité terrain est fournie, la superposer également
    if ground_truth is not None and ground_truth_color is not None:
        # Créer un masque pour la vérité terrain
        mask_ground_truth = np.zeros((*ground_truth.shape, 4), dtype=np.float32)
        mask_ground_truth[..., :3] = ground_truth_color
        mask_ground_truth[..., 3] = np.where(ground_truth.astype(bool), alpha, 0)
        
        # Superposer la vérité terrain
        ax.imshow(mask_ground_truth, interpolation='nearest')
        
        # Ajouter une légende
        import matplotlib.patches as mpatches
        pred_patch = mpatches.Patch(color=prediction_color, alpha=alpha, label='Prédiction')
        gt_patch = mpatches.Patch(color=ground_truth_color, alpha=alpha, label='Vérité terrain')
        ax.legend(handles=[pred_patch, gt_patch], loc='lower right')
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder si nécessaire
    if save_path is not None:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans: {save_path}")
    
    return fig

def create_evaluation_plot(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Évaluation des performances",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure:
    """
    Crée un graphique pour visualiser les métriques d'évaluation.
    
    Args:
        metrics: Dictionnaire de métriques par modèle
        title: Titre du graphique
        figsize: Taille de la figure
        save_path: Chemin où sauvegarder la figure (optionnel)
        dpi: Résolution de l'image sauvegardée
        
    Returns:
        Figure matplotlib
    """
    # Extraire les noms des modèles et les métriques
    model_names = list(metrics.keys())
    metric_names = set()
    for model_metrics in metrics.values():
        metric_names.update(model_metrics.keys())
    metric_names = sorted(list(metric_names))
    
    # Nombre de métriques et de modèles
    n_metrics = len(metric_names)
    n_models = len(model_names)
    
    # Créer la figure
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    # Gérer le cas d'une seule métrique
    if n_metrics == 1:
        axes = [axes]
    
    # Couleurs pour les barres
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Tracer les barres pour chaque métrique
    for i, metric_name in enumerate(metric_names):
        values = []
        for model_name in model_names:
            values.append(metrics[model_name].get(metric_name, 0))
        
        bars = axes[i].bar(model_names, values, color=colors)
        
        # Ajouter les valeurs au-dessus des barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', rotation=45)
        
        axes[i].set_title(metric_name)
        axes[i].set_ylim(0, min(1, max(values) * 1.2))  # Ajuster l'échelle pour les métriques entre 0 et 1
        
        # Rotation des étiquettes
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Sauvegarder si nécessaire
    if save_path is not None:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans: {save_path}")
    
    return fig

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    model_name: str = "Modèle",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure:
    """
    Trace la courbe ROC (Receiver Operating Characteristic).
    
    Args:
        fpr: Taux de faux positifs
        tpr: Taux de vrais positifs
        auc: Aire sous la courbe
        model_name: Nom du modèle
        figsize: Taille de la figure
        save_path: Chemin où sauvegarder la figure (optionnel)
        dpi: Résolution de l'image sauvegardée
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC = 0.5)')
    
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title('Courbe ROC')
    ax.legend(loc='lower right')
    
    # Ajuster les limites et la grille
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Sauvegarder si nécessaire
    if save_path is not None:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans: {save_path}")
    
    return fig

def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    average_precision: float,
    model_name: str = "Modèle",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure:
    """
    Trace la courbe précision-rappel.
    
    Args:
        precision: Précision à différents seuils
        recall: Rappel à différents seuils
        average_precision: Précision moyenne
        model_name: Nom du modèle
        figsize: Taille de la figure
        save_path: Chemin où sauvegarder la figure (optionnel)
        dpi: Résolution de l'image sauvegardée
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, label=f'{model_name} (AP = {average_precision:.3f})')
    ax.axhline(y=sum(precision)/len(precision), color='r', linestyle='--', 
              label='Ligne de base')
    
    ax.set_xlabel('Rappel')
    ax.set_ylabel('Précision')
    ax.set_title('Courbe Précision-Rappel')
    ax.legend(loc='lower left')
    
    # Ajuster les limites et la grille
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Sauvegarder si nécessaire
    if save_path is not None:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans: {save_path}")
    
    return fig

def visualize_error_map(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    original_data: Optional[np.ndarray] = None,
    title: str = "Carte d'erreurs",
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
    dpi: int = 300
) -> Figure:
    """
    Visualise une carte des erreurs de prédiction.
    
    Args:
        prediction: Prédiction binaire
        ground_truth: Vérité terrain
        original_data: Données originales (optionnel)
        title: Titre de la figure
        figsize: Taille de la figure
        save_path: Chemin où sauvegarder la figure (optionnel)
        dpi: Résolution de l'image sauvegardée
        
    Returns:
        Figure matplotlib
    """
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculer la carte d'erreurs
    # 0: Vrai Négatif (blanc), 1: Faux Positif (rouge), 
    # 2: Faux Négatif (bleu), 3: Vrai Positif (vert)
    error_map = prediction.astype(int) + 2 * ground_truth.astype(int)
    
    # Créer une colormap personnalisée
    colors = [(1, 1, 1), (1, 0.3, 0.3), (0.3, 0.3, 1), (0.3, 0.8, 0.3)]
    error_cmap = mcolors.ListedColormap(colors)
    
    # Si les données originales sont fournies, les afficher en arrière-plan
    if original_data is not None:
        # Normaliser les données originales
        if original_data.dtype != np.uint8:
            vmin, vmax = np.nanmin(original_data), np.nanmax(original_data)
            if vmin == vmax:
                normalized = np.zeros_like(original_data)
            else:
                normalized = (original_data - vmin) / (vmax - vmin)
        else:
            normalized = original_data / 255.0
        
        # Afficher les données originales
        ax.imshow(normalized, cmap='gray', interpolation='nearest')
        
        # Superposer la carte d'erreurs avec transparence
        # Créer un masque RGBA pour les erreurs
        error_rgba = np.zeros((*error_map.shape, 4), dtype=np.float32)
        
        # Attribuer les couleurs selon le type d'erreur
        for i, color in enumerate(colors):
            mask = error_map == i
            error_rgba[mask, :3] = color
            
            # Définir la transparence (0 = transparent pour VN, 0.7 pour les autres)
            error_rgba[mask, 3] = 0 if i == 0 else 0.7
        
        # Afficher le masque d'erreur
        ax.imshow(error_rgba, interpolation='nearest')
    else:
        # Afficher directement la carte d'erreurs
        im = ax.imshow(error_map, cmap=error_cmap, interpolation='nearest')
        
        # Ajouter une barre de couleur
        cbar = plt.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5])
        cbar.set_ticklabels(['Vrai Négatif', 'Faux Positif', 'Faux Négatif', 'Vrai Positif'])
    
    # Ajouter une légende
    import matplotlib.patches as mpatches
    vn_patch = mpatches.Patch(color=colors[0], label='Vrai Négatif')
    fp_patch = mpatches.Patch(color=colors[1], label='Faux Positif')
    fn_patch = mpatches.Patch(color=colors[2], label='Faux Négatif')
    vp_patch = mpatches.Patch(color=colors[3], label='Vrai Positif')
    ax.legend(handles=[vn_patch, fp_patch, fn_patch, vp_patch], loc='lower right')
    
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder si nécessaire
    if save_path is not None:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure sauvegardée dans: {save_path}")
    
    return fig 

def visualize_predictions(
    prediction: np.ndarray,
    probability: Optional[np.ndarray] = None,
    output_dir: str = ".",
    base_name: str = "prediction",
    threshold: Optional[float] = None,
    **kwargs
) -> str:
    """
    Wrapper pour générer plusieurs visualisations d'une prédiction.

    Args:
        prediction: Prédiction binaire (2D array)
        probability: Prédiction de probabilité (2D array, optionnel)
        output_dir: Répertoire pour sauvegarder les visualisations
        base_name: Nom de base pour les fichiers de sortie
        threshold: Seuil utilisé pour la prédiction (pour le titre)
        **kwargs: Arguments additionnels pour visualize_prediction()

    Returns:
        Chemin du répertoire de visualisations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Générer titre avec seuil si disponible
    if threshold is not None:
        title = f"{base_name} (seuil: {threshold}m)"
    else:
        title = base_name

    # Visualisation de la prédiction binaire
    pred_path = os.path.join(output_dir, f"{base_name}_binary.png")
    visualize_prediction(
        prediction=prediction,
        title=f"Prédiction binaire - {title}",
        save_path=pred_path,
        **kwargs
    )

    # Visualisation de la probabilité si disponible
    if probability is not None:
        prob_path = os.path.join(output_dir, f"{base_name}_probability.png")
        visualize_prediction(
            prediction=probability,
            title=f"Probabilité - {title}",
            save_path=prob_path,
            **kwargs
        )

    return output_dir


# Alias pour compatibilité
create_comparison_figure = visualize_comparison
