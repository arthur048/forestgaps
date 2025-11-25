"""
Utilitaires pour le traitement d'images dans le module d'inférence.

Ce module fournit des fonctions pour préparer les images avant l'inférence,
notamment le tuilage, le prétraitement et la normalisation des données.
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Any, Tuple, List, Optional, Union, Callable

# Configuration du logging
logger = logging.getLogger(__name__)

def normalize_data(
    data: np.ndarray,
    method: str = "min-max",
    stats: Optional[Dict[str, float]] = None,
    compute_stats: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalise les données selon la méthode spécifiée.
    
    Args:
        data: Données à normaliser
        method: Méthode de normalisation ('min-max', 'z-score', 'percentile')
        stats: Statistiques de normalisation prédéfinies
        compute_stats: Calculer et retourner les statistiques utilisées
        
    Returns:
        Tuple contenant les données normalisées et les statistiques utilisées
    """
    # Remplacer les NaN par 0
    if np.isnan(data).any():
        logger.warning("Les données contiennent des valeurs NaN qui seront remplacées par 0")
        data = np.nan_to_num(data, nan=0.0)
    
    # Initialiser les statistiques
    if stats is None:
        stats = {}
    
    # Appliquer la normalisation selon la méthode choisie
    if method == "min-max":
        # Utiliser les statistiques fournies ou calculer les nouvelles
        data_min = stats.get("min")
        data_max = stats.get("max")
        
        if data_min is None or data_max is None:
            data_min = np.min(data)
            data_max = np.max(data)
            if compute_stats:
                stats["min"] = float(data_min)
                stats["max"] = float(data_max)
        
        # Éviter la division par zéro
        if data_max == data_min:
            normalized = np.zeros_like(data, dtype=np.float32)
        else:
            normalized = (data - data_min) / (data_max - data_min)
    
    elif method == "z-score":
        # Utiliser les statistiques fournies ou calculer les nouvelles
        mean = stats.get("mean")
        std = stats.get("std")
        
        if mean is None or std is None:
            mean = np.mean(data)
            std = np.std(data)
            if compute_stats:
                stats["mean"] = float(mean)
                stats["std"] = float(std)
        
        # Éviter la division par zéro
        if std == 0:
            normalized = np.zeros_like(data, dtype=np.float32)
        else:
            normalized = (data - mean) / std
    
    elif method == "percentile":
        # Utiliser les statistiques fournies ou calculer les nouvelles
        p_min = stats.get("p_min", 1)
        p_max = stats.get("p_max", 99)
        
        if "min_value" not in stats or "max_value" not in stats:
            min_value = np.percentile(data, p_min)
            max_value = np.percentile(data, p_max)
            if compute_stats:
                stats["min_value"] = float(min_value)
                stats["max_value"] = float(max_value)
                stats["p_min"] = float(p_min)
                stats["p_max"] = float(p_max)
        else:
            min_value = stats["min_value"]
            max_value = stats["max_value"]
        
        # Éviter la division par zéro
        if max_value == min_value:
            normalized = np.zeros_like(data, dtype=np.float32)
        else:
            normalized = np.clip((data - min_value) / (max_value - min_value), 0, 1)
    
    else:
        raise ValueError(f"Méthode de normalisation '{method}' non reconnue")
    
    return normalized.astype(np.float32), stats

def preprocess_for_model(
    data: np.ndarray,
    normalize_method: str = "min-max",
    normalize_stats: Optional[Dict[str, float]] = None,
    add_batch_dim: bool = True,
    add_channel_dim: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Prétraite les données pour l'inférence avec un modèle PyTorch.
    
    Args:
        data: Données à prétraiter
        normalize_method: Méthode de normalisation
        normalize_stats: Statistiques de normalisation prédéfinies
        add_batch_dim: Ajouter une dimension pour le lot (batch)
        add_channel_dim: Ajouter une dimension pour le canal
        
    Returns:
        Tuple contenant le tenseur prétraité et les statistiques de normalisation
    """
    # Normaliser les données
    normalized_data, stats = normalize_data(data, normalize_method, normalize_stats)
    
    # Convertir en tenseur PyTorch
    tensor = torch.from_numpy(normalized_data)
    
    # Ajouter la dimension du canal si nécessaire
    if add_channel_dim and len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)  # Ajouter dim canal: (H, W) -> (1, H, W)
    
    # Ajouter la dimension du lot si nécessaire
    if add_batch_dim:
        tensor = tensor.unsqueeze(0)  # Ajouter dim batch: (C, H, W) -> (1, C, H, W)
    
    return tensor, stats

def create_tiles(
    data: np.ndarray,
    tile_size: int = 512,
    overlap: float = 0.25
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Découpe une image en tuiles avec chevauchement.
    
    Args:
        data: Image à découper en tuiles
        tile_size: Taille des tuiles (supposée carrée)
        overlap: Proportion de chevauchement entre les tuiles (0.0 à 1.0)
        
    Returns:
        Tuple contenant la liste des tuiles et la liste des coordonnées (x, y) de chaque tuile
    """
    if len(data.shape) == 3:
        h, w, c = data.shape
        is_multichannel = True
    else:
        h, w = data.shape
        is_multichannel = False
    
    # Calculer le pas entre les tuiles
    stride = int(tile_size * (1 - overlap))
    
    # Calculer le nombre de tuiles dans chaque dimension
    num_tiles_h = max(1, (h - tile_size + stride) // stride)
    num_tiles_w = max(1, (w - tile_size + stride) // stride)
    
    # Ajuster pour couvrir toute l'image si nécessaire
    if num_tiles_h * stride < h:
        num_tiles_h += 1
    if num_tiles_w * stride < w:
        num_tiles_w += 1
    
    tiles = []
    coords = []
    
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            # Calculer les coordonnées de la tuile
            x = min(j * stride, w - tile_size) if w > tile_size else 0
            y = min(i * stride, h - tile_size) if h > tile_size else 0
            
            # Extraire la tuile
            if is_multichannel:
                tile = data[y:y+tile_size, x:x+tile_size, :]
            else:
                tile = data[y:y+tile_size, x:x+tile_size]
            
            # Gérer le cas où la tuile est plus petite que tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                # Créer une tuile vide aux bonnes dimensions
                if is_multichannel:
                    padded_tile = np.zeros((tile_size, tile_size, c), dtype=tile.dtype)
                else:
                    padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                
                # Copier les données
                h_actual, w_actual = tile.shape[:2]
                padded_tile[:h_actual, :w_actual, ...] = tile
                tile = padded_tile
            
            tiles.append(tile)
            coords.append((x, y))
    
    return tiles, coords

def stitch_tiles(
    tiles: List[np.ndarray],
    coords: List[Tuple[int, int]],
    output_shape: Tuple[int, int],
    overlap: float = 0.25,
    blend_method: str = "linear"
) -> np.ndarray:
    """
    Recompose une image à partir de tuiles avec fusion des chevauchements.
    
    Args:
        tiles: Liste des tuiles
        coords: Liste des coordonnées (x, y) de chaque tuile
        output_shape: Dimensions (hauteur, largeur) de l'image de sortie
        overlap: Proportion de chevauchement entre les tuiles (0.0 à 1.0)
        blend_method: Méthode de fusion des zones de chevauchement ('linear', 'mean', 'max')
        
    Returns:
        Image recomposée
    """
    h, w = output_shape
    is_multichannel = len(tiles[0].shape) > 2
    
    if is_multichannel:
        c = tiles[0].shape[2]
        output = np.zeros((h, w, c), dtype=np.float32)
        weight = np.zeros((h, w, 1), dtype=np.float32)
    else:
        output = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
    
    tile_size = tiles[0].shape[0]
    
    # Créer un masque de poids pour le mélange linéaire
    if blend_method == "linear":
        # Créer une matrice de pondération qui diminue du centre vers les bords
        y, x = np.mgrid[0:tile_size, 0:tile_size]
        center = tile_size // 2
        if is_multichannel:
            weight_mask = np.minimum(
                np.minimum(x, tile_size - x - 1),
                np.minimum(y, tile_size - y - 1)
            ).astype(np.float32)[:, :, np.newaxis]
        else:
            weight_mask = np.minimum(
                np.minimum(x, tile_size - x - 1),
                np.minimum(y, tile_size - y - 1)
            ).astype(np.float32)
        
        # Normaliser
        weight_mask = weight_mask / np.max(weight_mask)
    else:
        # Pour 'mean' ou 'max', utiliser un poids uniforme
        if is_multichannel:
            weight_mask = np.ones((tile_size, tile_size, 1), dtype=np.float32)
        else:
            weight_mask = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Ajouter les tuiles à l'image de sortie
    for tile, (x, y) in zip(tiles, coords):
        # Calculer la zone effective de la tuile
        y_end = min(y + tile_size, h)
        x_end = min(x + tile_size, w)
        tile_h, tile_w = y_end - y, x_end - x
        
        # Extraire la partie de la tuile qui correspond à l'image de sortie
        tile_part = tile[:tile_h, :tile_w]
        mask_part = weight_mask[:tile_h, :tile_w]
        
        if blend_method == "max":
            # Utiliser le maximum pour chaque pixel
            if is_multichannel:
                mask = weight[y:y_end, x:x_end, 0] < mask_part[:, :, 0]
                for c_idx in range(c):
                    output[y:y_end, x:x_end, c_idx] = np.where(
                        mask, tile_part[:, :, c_idx], output[y:y_end, x:x_end, c_idx]
                    )
                weight[y:y_end, x:x_end] = np.maximum(weight[y:y_end, x:x_end], mask_part)
            else:
                mask = weight[y:y_end, x:x_end] < mask_part
                output[y:y_end, x:x_end] = np.where(
                    mask, tile_part, output[y:y_end, x:x_end]
                )
                weight[y:y_end, x:x_end] = np.maximum(weight[y:y_end, x:x_end], mask_part)
        else:
            # Méthode linéaire (pondérée) ou moyenne
            if is_multichannel:
                output[y:y_end, x:x_end] += tile_part * mask_part
                weight[y:y_end, x:x_end] += mask_part
            else:
                output[y:y_end, x:x_end] += tile_part * mask_part
                weight[y:y_end, x:x_end] += mask_part
    
    # Normaliser par le poids pour éviter les artefacts de chevauchement
    # Éviter la division par zéro
    mask = weight > 0
    if is_multichannel:
        for c_idx in range(c):
            output[:, :, c_idx] = np.where(
                mask[:, :, 0], output[:, :, c_idx] / weight[:, :, 0], 0
            )
    else:
        output = np.where(mask, output / weight, 0)
    
    return output

def apply_threshold(
    probability: np.ndarray,
    threshold: float = 0.5,
    as_binary: bool = True
) -> np.ndarray:
    """
    Applique un seuil aux probabilités pour obtenir une prédiction binaire.
    
    Args:
        probability: Matrice des probabilités (0.0 à 1.0)
        threshold: Seuil de décision (0.0 à 1.0)
        as_binary: Convertir en valeurs binaires (0 et 1) ou garder les probabilités filtrées
        
    Returns:
        Matrice des prédictions après application du seuil
    """
    if as_binary:
        return (probability >= threshold).astype(np.uint8)
    else:
        # Mettre à zéro les probabilités en dessous du seuil
        return np.where(probability >= threshold, probability, 0).astype(np.float32)

def apply_crf(
    probability: np.ndarray,
    reference_data: np.ndarray,
    theta_p: float = 1.0,
    theta_s: float = 1.0,
    num_iterations: int = 5
) -> np.ndarray:
    """
    Applique un Conditional Random Field (CRF) pour affiner les prédictions.
    
    Args:
        probability: Matrice des probabilités (0.0 à 1.0)
        reference_data: Données de référence pour les caractéristiques spatiales (généralement l'image d'entrée)
        theta_p: Paramètre de pondération des caractéristiques d'apparence
        theta_s: Paramètre de pondération des caractéristiques de lissage
        num_iterations: Nombre d'itérations pour l'inférence CRF
        
    Returns:
        Matrice des probabilités affinées après application du CRF
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        logger.error("Le package pydensecrf n'est pas installé. Impossible d'appliquer le CRF.")
        return probability
    
    # S'assurer que les dimensions sont correctes
    h, w = probability.shape
    
    # Préparer les probabilités pour le CRF (format attendu: [classes, height, width])
    # Pour la segmentation binaire, nous avons besoin de 2 canaux (foreground et background)
    prob = np.stack([1 - probability, probability], axis=0)
    
    # Créer le CRF
    d = dcrf.DenseCRF2D(w, h, 2)  # width, height, nlabels
    
    # Définir les potentiels unaires à partir des probabilités
    U = unary_from_softmax(prob)
    d.setUnaryEnergy(U)
    
    # Préparer l'image de référence pour les potentiels de paires
    if reference_data.ndim == 2:
        # Si l'image est en niveaux de gris, la convertir en RGB en la dupliquant
        reference_data = np.stack([reference_data] * 3, axis=2)
    
    # Normaliser l'image de référence
    reference_data = reference_data.astype(np.float32) / 255.0
    
    # Ajouter les potentiels de paires (apparence et lissage)
    d.addPairwiseGaussian(sxy=theta_s, compat=3)
    d.addPairwiseBilateral(sxy=theta_p, srgb=0.01, rgbim=reference_data, compat=10)
    
    # Effectuer l'inférence
    Q = d.inference(num_iterations)
    
    # Récupérer la carte de probabilité affinée
    refined_prob = np.array(Q)[1].reshape((h, w))
    
    return refined_prob

def post_process_prediction(
    prediction: np.ndarray,
    min_area: int = 10,
    close_kernel_size: int = 3,
    open_kernel_size: int = 3
) -> np.ndarray:
    """
    Applique des opérations morphologiques pour nettoyer les prédictions binaires.
    
    Args:
        prediction: Prédiction binaire (0 et 1)
        min_area: Taille minimale des objets à conserver
        close_kernel_size: Taille du noyau pour l'opération de fermeture
        open_kernel_size: Taille du noyau pour l'opération d'ouverture
        
    Returns:
        Prédiction nettoyée
    """
    try:
        from scipy import ndimage
        import cv2
    except ImportError:
        logger.error("Les packages scipy ou opencv-python ne sont pas installés. Impossible d'appliquer le post-traitement.")
        return prediction
    
    # Convertir en type binaire
    binary = prediction.astype(bool)
    
    # Appliquer la fermeture pour combler les petits trous
    if close_kernel_size > 0:
        kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    
    # Appliquer l'ouverture pour supprimer les petits objets isolés
    if open_kernel_size > 0:
        kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    
    # Supprimer les objets plus petits que min_area
    if min_area > 0:
        labeled, num_features = ndimage.label(binary)
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        too_small = sizes < min_area
        mask_too_small = too_small[labeled - 1]
        binary[mask_too_small] = False
    
    return binary.astype(np.uint8) 