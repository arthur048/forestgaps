# Fonctions d'opérations sur les rasters
"""
Fonctions d'opérations sur les rasters pour ForestGaps.

Ce module fournit des fonctions pour manipuler, charger et traiter
les données raster (DSM, CHM, etc.) utilisées dans le workflow ForestGaps.
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Dict, List, Tuple, Optional, Union, Any

from forestgaps.utils.errors import InvalidDataFormatError, DataProcessingError


def load_raster(file_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Charge un fichier raster et retourne les données et les métadonnées.
    
    Args:
        file_path (str): Chemin vers le fichier raster.
        
    Returns:
        tuple: (données raster, métadonnées)
        
    Raises:
        InvalidDataFormatError: Si le format du fichier est invalide.
        DataProcessingError: Si une erreur se produit lors du chargement.
    """
    try:
        with rasterio.open(file_path) as src:
            # Lire les données
            data = src.read(1)  # Lire la première bande
            
            # Extraire les métadonnées importantes
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'nodata': src.nodata
            }
            
            return data, metadata
    except rasterio.errors.RasterioIOError as e:
        raise InvalidDataFormatError(f"Erreur lors de l'ouverture du fichier raster {file_path}: {str(e)}")
    except Exception as e:
        raise DataProcessingError(f"Erreur lors du chargement du raster {file_path}: {str(e)}")


def save_raster(data: np.ndarray, metadata: Dict, file_path: str) -> None:
    """
    Sauvegarde des données raster dans un fichier.
    
    Args:
        data (numpy.ndarray): Données raster à sauvegarder.
        metadata (dict): Métadonnées du raster.
        file_path (str): Chemin où sauvegarder le fichier.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de la sauvegarde.
    """
    try:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Préparer les paramètres pour l'écriture
        kwargs = {
            'driver': 'GTiff',
            'height': metadata['height'],
            'width': metadata['width'],
            'count': 1,
            'dtype': data.dtype,
            'crs': metadata['crs'],
            'transform': metadata['transform']
        }
        
        # Ajouter la valeur nodata si présente
        if 'nodata' in metadata and metadata['nodata'] is not None:
            kwargs['nodata'] = metadata['nodata']
        
        # Écrire le fichier
        with rasterio.open(file_path, 'w', **kwargs) as dst:
            dst.write(data, 1)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de la sauvegarde du raster {file_path}: {str(e)}")


def get_raster_window(file_path: str, window: Window) -> Tuple[np.ndarray, Dict]:
    """
    Charge une fenêtre spécifique d'un raster.
    
    Args:
        file_path (str): Chemin vers le fichier raster.
        window (rasterio.windows.Window): Fenêtre à extraire.
        
    Returns:
        tuple: (données de la fenêtre, métadonnées)
        
    Raises:
        InvalidDataFormatError: Si le format du fichier est invalide.
        DataProcessingError: Si une erreur se produit lors du chargement.
    """
    try:
        with rasterio.open(file_path) as src:
            # Lire la fenêtre
            data = src.read(1, window=window)
            
            # Calculer la transformation pour cette fenêtre
            window_transform = rasterio.windows.transform(window, src.transform)
            
            # Extraire les métadonnées importantes
            metadata = {
                'crs': src.crs,
                'transform': window_transform,
                'width': window.width,
                'height': window.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'nodata': src.nodata,
                'window': window
            }
            
            return data, metadata
    except rasterio.errors.RasterioIOError as e:
        raise InvalidDataFormatError(f"Erreur lors de l'ouverture du fichier raster {file_path}: {str(e)}")
    except Exception as e:
        raise DataProcessingError(f"Erreur lors du chargement de la fenêtre du raster {file_path}: {str(e)}")


def normalize_raster(data: np.ndarray, method: str = 'minmax', 
                    min_val: Optional[float] = None, max_val: Optional[float] = None,
                    mean_val: Optional[float] = None, std_val: Optional[float] = None) -> np.ndarray:
    """
    Normalise les données raster selon différentes méthodes.
    
    Args:
        data (numpy.ndarray): Données raster à normaliser.
        method (str): Méthode de normalisation ('minmax' ou 'zscore').
        min_val (float, optional): Valeur minimale pour la normalisation minmax.
        max_val (float, optional): Valeur maximale pour la normalisation minmax.
        mean_val (float, optional): Valeur moyenne pour la normalisation zscore.
        std_val (float, optional): Écart-type pour la normalisation zscore.
        
    Returns:
        numpy.ndarray: Données normalisées.
        
    Raises:
        ValueError: Si la méthode de normalisation est invalide.
    """
    # Créer un masque pour les valeurs valides (non-NaN)
    valid_mask = ~np.isnan(data)
    
    # Copier les données pour éviter de modifier l'original
    normalized_data = data.copy()
    
    if method == 'minmax':
        # Calculer min et max si non fournis
        if min_val is None:
            min_val = np.nanmin(data) if np.any(valid_mask) else 0
        if max_val is None:
            max_val = np.nanmax(data) if np.any(valid_mask) else 1
        
        # Éviter la division par zéro
        range_val = max_val - min_val
        if range_val > 0:
            normalized_data = np.where(valid_mask, (data - min_val) / range_val, 0)
        else:
            normalized_data = np.where(valid_mask, 0, 0)
    
    elif method == 'zscore':
        # Calculer moyenne et écart-type si non fournis
        if mean_val is None:
            mean_val = np.nanmean(data) if np.any(valid_mask) else 0
        if std_val is None:
            std_val = np.nanstd(data) if np.any(valid_mask) else 1
        
        # Éviter la division par zéro
        if std_val > 0:
            normalized_data = np.where(valid_mask, (data - mean_val) / std_val, 0)
        else:
            normalized_data = np.where(valid_mask, 0, 0)
    
    else:
        raise ValueError(f"Méthode de normalisation '{method}' non supportée. Utilisez 'minmax' ou 'zscore'.")
    
    return normalized_data


def calculate_raster_statistics(file_path: str) -> Dict[str, float]:
    """
    Calcule les statistiques d'un raster (min, max, moyenne, écart-type).
    
    Args:
        file_path (str): Chemin vers le fichier raster.
        
    Returns:
        dict: Statistiques du raster.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors du calcul.
    """
    try:
        with rasterio.open(file_path) as src:
            # Lire les données
            data = src.read(1)
            
            # Créer un masque pour les valeurs valides (non-NaN et non-nodata)
            valid_mask = ~np.isnan(data)
            if src.nodata is not None:
                valid_mask &= (data != src.nodata)
            
            # Calculer les statistiques
            if np.any(valid_mask):
                min_val = float(np.min(data[valid_mask]))
                max_val = float(np.max(data[valid_mask]))
                mean_val = float(np.mean(data[valid_mask]))
                std_val = float(np.std(data[valid_mask]))
                median_val = float(np.median(data[valid_mask]))
                count = int(np.sum(valid_mask))
            else:
                min_val = max_val = mean_val = std_val = median_val = 0.0
                count = 0
            
            return {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'count': count,
                'total_pixels': data.size
            }
    except Exception as e:
        raise DataProcessingError(f"Erreur lors du calcul des statistiques du raster {file_path}: {str(e)}")


def align_rasters(source_path: str, target_path: str, output_path: str, 
                 resampling_method: str = 'bilinear') -> None:
    """
    Aligne un raster source sur un raster cible et sauvegarde le résultat.
    
    Args:
        source_path (str): Chemin vers le raster source à aligner.
        target_path (str): Chemin vers le raster cible (référence).
        output_path (str): Chemin où sauvegarder le raster aligné.
        resampling_method (str): Méthode de rééchantillonnage.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de l'alignement.
    """
    try:
        import rasterio.warp
        
        # Ouvrir le raster cible pour obtenir les métadonnées de référence
        with rasterio.open(target_path) as target:
            target_crs = target.crs
            target_transform = target.transform
            target_width = target.width
            target_height = target.height
            
            # Ouvrir le raster source
            with rasterio.open(source_path) as source:
                # Déterminer la méthode de rééchantillonnage
                resampling = getattr(rasterio.warp.Resampling, resampling_method)
                
                # Rééchantillonner le raster source pour l'aligner sur le raster cible
                data = rasterio.warp.reproject(
                    source=rasterio.band(source, 1),
                    destination=np.zeros((target_height, target_width), dtype=source.dtypes[0]),
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resampling
                )[0]
                
                # Sauvegarder le raster aligné
                metadata = {
                    'crs': target_crs,
                    'transform': target_transform,
                    'width': target_width,
                    'height': target_height,
                    'count': 1,
                    'dtype': source.dtypes[0],
                    'nodata': source.nodata
                }
                
                save_raster(data, metadata, output_path)
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de l'alignement du raster {source_path} sur {target_path}: {str(e)}")


def create_mask_from_raster(raster_path: str, threshold: float, output_path: str = None,
                           above_threshold: bool = True) -> np.ndarray:
    """
    Crée un masque binaire à partir d'un raster en appliquant un seuil.
    
    Args:
        raster_path (str): Chemin vers le fichier raster.
        threshold (float): Valeur seuil.
        output_path (str, optional): Chemin où sauvegarder le masque.
        above_threshold (bool): Si True, le masque est 1 pour les valeurs > seuil.
        
    Returns:
        numpy.ndarray: Masque binaire.
        
    Raises:
        DataProcessingError: Si une erreur se produit lors de la création du masque.
    """
    try:
        # Charger le raster
        data, metadata = load_raster(raster_path)
        
        # Créer un masque pour les valeurs valides (non-NaN et non-nodata)
        valid_mask = ~np.isnan(data)
        if metadata['nodata'] is not None:
            valid_mask &= (data != metadata['nodata'])
        
        # Créer le masque binaire
        if above_threshold:
            mask = np.where(valid_mask, (data > threshold).astype(np.uint8), 255)
        else:
            mask = np.where(valid_mask, (data <= threshold).astype(np.uint8), 255)
        
        # Sauvegarder le masque si un chemin de sortie est spécifié
        if output_path is not None:
            # Mettre à jour les métadonnées pour le masque
            mask_metadata = metadata.copy()
            mask_metadata['dtype'] = 'uint8'
            mask_metadata['nodata'] = 255
            
            save_raster(mask, mask_metadata, output_path)
        
        return mask
    except Exception as e:
        raise DataProcessingError(f"Erreur lors de la création du masque à partir du raster {raster_path}: {str(e)}")
