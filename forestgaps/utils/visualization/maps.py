"""
Fonctions de visualisation de cartes pour ForestGaps.

Ce module fournit des fonctions pour visualiser les données géospatiales,
les prédictions et les résultats du workflow ForestGaps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Union, Any


def visualize_dsm_with_gaps(dsm, gaps_mask, title="DSM avec trouées", save_path=None, figsize=(10, 8)):
    """
    Visualise un DSM avec les trouées détectées superposées.
    
    Args:
        dsm (numpy.ndarray): Modèle numérique de surface (DSM).
        gaps_mask (numpy.ndarray): Masque binaire des trouées.
        title (str): Titre de la figure.
        save_path (str, optional): Chemin pour sauvegarder la figure.
        figsize (tuple): Taille de la figure.
        
    Returns:
        matplotlib.figure.Figure: Figure matplotlib.
    """
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Afficher le DSM avec une colormap terrain
    dsm_plot = ax.imshow(dsm, cmap='terrain', interpolation='nearest')
    
    # Créer un masque transparent pour les trouées
    gaps_overlay = np.zeros_like(dsm, dtype=np.float32)
    gaps_overlay[gaps_mask > 0] = 1.0
    
    # Superposer les trouées avec une colormap rouge transparente
    red_cmap = LinearSegmentedColormap.from_list('red_transparent', 
                                                [(0, 0, 0, 0), (1, 0, 0, 0.7)])
    gaps_plot = ax.imshow(gaps_overlay, cmap=red_cmap, interpolation='nearest')
    
    # Ajouter une barre de couleur pour le DSM
    cbar = plt.colorbar(dsm_plot, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Hauteur (m)')
    
    # Ajouter une légende pour les trouées
    gap_patch = mpatches.Patch(color='red', alpha=0.7, label='Trouées')
    ax.legend(handles=[gap_patch], loc='upper right')
    
    # Ajouter un titre et des étiquettes
    ax.set_title(title)
    ax.set_xlabel('Colonne')
    ax.set_ylabel('Ligne')
    
    # Sauvegarder la figure si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Carte DSM avec trouées sauvegardée dans {save_path}")
    
    return fig


def visualize_predictions_grid(model, data_loader, device, config, num_samples=5, threshold_idx=0):
    """
    Crée une grille de visualisations des prédictions sur plusieurs échantillons.
    
    Args:
        model: Modèle U-Net.
        data_loader: DataLoader contenant les données.
        device: Périphérique (GPU/CPU).
        config: Configuration.
        num_samples: Nombre d'échantillons à visualiser.
        threshold_idx: Index du seuil de hauteur à utiliser.
        
    Returns:
        matplotlib.figure.Figure: Figure matplotlib.
    """
    import torch
    
    model.eval()
    
    # Sélectionner le seuil
    threshold = config.THRESHOLDS[threshold_idx]
    normalized_threshold = threshold / max(config.THRESHOLDS)
    threshold_tensor = torch.tensor([[normalized_threshold]], dtype=torch.float32).to(device)
    
    # Collecter les échantillons
    samples = []
    with torch.no_grad():
        for dsm, _, target in data_loader:
            # Prendre seulement le nombre d'échantillons nécessaires
            for i in range(min(dsm.size(0), num_samples - len(samples))):
                dsm_i = dsm[i:i+1].to(device)
                target_i = target[i:i+1].to(device)
                
                # Prédiction
                output_i = model(dsm_i, threshold_tensor)
                pred_binary = (torch.sigmoid(output_i) > 0.5).float()
                
                # Calculer l'IoU pour cet échantillon
                from forestgaps.utils.visualization.plots import iou_metric
                iou = iou_metric(output_i, target_i).item()
                
                # Ajouter l'échantillon
                samples.append({
                    'dsm': dsm_i.cpu().numpy()[0, 0],
                    'target': target_i.cpu().numpy()[0, 0],
                    'output': torch.sigmoid(output_i).cpu().numpy()[0, 0],
                    'pred_binary': pred_binary.cpu().numpy()[0, 0],
                    'iou': iou
                })
                
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
    
    # Créer la grille de visualisations
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i, sample in enumerate(samples):
        # Afficher le DSM
        axes[i, 0].imshow(sample['dsm'], cmap='terrain')
        axes[i, 0].set_title(f"DSM")
        
        # Afficher le masque réel
        axes[i, 1].imshow(sample['target'], cmap='binary')
        axes[i, 1].set_title(f"Trouées réelles (seuil {threshold}m)")
        
        # Afficher la prédiction
        axes[i, 2].imshow(sample['pred_binary'], cmap='binary')
        axes[i, 2].set_title(f"Trouées prédites (IoU: {sample['iou']:.4f})")
    
    plt.tight_layout()
    
    return fig


def visualize_threshold_comparison(model, data_loader, device, config, sample_idx=0):
    """
    Visualise les prédictions du modèle pour différents seuils de hauteur sur un même échantillon.
    
    Args:
        model: Modèle U-Net.
        data_loader: DataLoader contenant les données.
        device: Périphérique (GPU/CPU).
        config: Configuration.
        sample_idx: Index de l'échantillon à visualiser.
        
    Returns:
        matplotlib.figure.Figure: Figure matplotlib.
    """
    import torch
    
    model.eval()
    
    # Obtenir un échantillon
    sample_found = False
    dsm_sample = None
    
    with torch.no_grad():
        for idx, (dsm, _, _) in enumerate(data_loader):
            if idx * data_loader.batch_size + sample_idx < len(data_loader.dataset):
                if sample_idx < dsm.size(0):
                    dsm_sample = dsm[sample_idx:sample_idx+1].to(device)
                    sample_found = True
                    break
    
    if not sample_found or dsm_sample is None:
        print(f"Échantillon à l'index {sample_idx} non trouvé")
        return None
    
    # Créer la figure
    rows = len(config.THRESHOLDS)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    
    # Afficher le DSM original seulement une fois
    dsm_np = dsm_sample.cpu().numpy()[0, 0]
    
    for i, threshold in enumerate(config.THRESHOLDS):
        # Normaliser le seuil
        normalized_threshold = threshold / max(config.THRESHOLDS)
        threshold_tensor = torch.tensor([[normalized_threshold]], dtype=torch.float32).to(device)
        
        # Faire la prédiction
        output = model(dsm_sample, threshold_tensor)
        pred_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred_binary = (pred_prob > 0.5).astype(np.float32)
        
        # Afficher le DSM
        axes[i, 0].imshow(dsm_np, cmap='terrain')
        axes[i, 0].set_title(f"DSM")
        
        # Afficher la carte de probabilités
        axes[i, 1].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Probabilités (seuil {threshold}m)")
        
        # Afficher la prédiction binaire
        axes[i, 2].imshow(pred_binary, cmap='binary')
        axes[i, 2].set_title(f"Trouées prédites (seuil {threshold}m)")
    
    plt.tight_layout()
    
    return fig


def visualize_predictions_with_actual(model, tile_info, threshold_value, device, config, num_samples=5):
    """
    Visualise les prédictions du modèle comparées aux masques réels.
    
    Args:
        model: Modèle PyTorch.
        tile_info: Liste d'informations sur les tuiles.
        threshold_value: Seuil de hauteur pour la segmentation.
        device: Périphérique (CPU ou GPU).
        config: Configuration.
        num_samples: Nombre d'exemples à visualiser.
        
    Returns:
        matplotlib.figure.Figure: Figure matplotlib.
    """
    import torch
    
    model.eval()
    
    # Sélectionner des tuiles aléatoires
    indices = np.random.choice(len(tile_info), num_samples, replace=False)
    
    # Normaliser le seuil
    normalized_threshold = threshold_value / max(config.THRESHOLDS)
    threshold_tensor = torch.tensor([[normalized_threshold]], dtype=torch.float32).to(device)
    
    # Créer la figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            info = tile_info[idx]
            
            # Charger la tuile DSM
            dsm_tile = np.load(info['dsm_path'])
            
            # Charger le masque réel
            mask_path = info['mask_paths'][threshold_value]
            mask_tile = np.load(mask_path)
            
            # Convertir le masque en binaire
            mask_valid = (mask_tile != 255)
            mask_binary = np.where(mask_valid, (mask_tile > 0).astype(np.float32), 0)
            
            # Normaliser la tuile DSM
            dsm_valid = ~np.isnan(dsm_tile)
            dsm_display = dsm_tile.copy()  # Pour l'affichage
            
            if np.any(dsm_valid):
                dsm_min = np.nanmin(dsm_tile)
                dsm_max = np.nanmax(dsm_tile)
                dsm_range = dsm_max - dsm_min
                if dsm_range > 0:
                    dsm_tile = np.where(dsm_valid, (dsm_tile - dsm_min) / dsm_range, 0)
                else:
                    dsm_tile = np.where(dsm_valid, 0, 0)
            
            # Convertir en tensor
            dsm_tensor = torch.tensor(dsm_tile, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Faire la prédiction
            output = model(dsm_tensor, threshold_tensor)
            
            # Convertir en NumPy
            prediction = output.squeeze().cpu().numpy()
            
            # Calculer l'IoU
            from forestgaps.utils.visualization.plots import iou_metric
            pred_tensor = torch.sigmoid(torch.tensor(prediction, dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.tensor(mask_binary, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            iou = iou_metric(pred_tensor, mask_tensor).item()
            
            # Afficher
            axes[i, 0].imshow(dsm_display, cmap='terrain')
            axes[i, 0].set_title(f"DSM - {info['site']}")
            
            axes[i, 1].imshow(mask_binary, cmap='binary')
            axes[i, 1].set_title(f"Trouées réelles (seuil {threshold_value}m)")
            
            axes[i, 2].imshow(torch.sigmoid(torch.tensor(prediction)).numpy() > 0.5, cmap='binary')
            axes[i, 2].set_title(f"Trouées prédites (IoU: {iou:.4f})")
    
    plt.tight_layout()
    
    return fig
