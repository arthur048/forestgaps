"""
Module de conversion des formats de données pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour convertir les données entre différents formats
(GeoTIFF, NumPy, etc.) et pour effectuer des transformations sur les données.
"""

import os
import logging
from typing import Dict, Any, Union, Optional, Tuple, List
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# Configuration du logger
logger = logging.getLogger(__name__)

def convert_to_numpy(
    raster_path: Union[str, Path],
    window: Optional[Window] = None,
    normalize: bool = False,
    fill_nodata: bool = True,
    fill_value: float = 0.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convertit un raster en tableau NumPy.
    
    Args:
        raster_path: Chemin vers le fichier raster
        window: Fenêtre de lecture (None = lire tout le raster)
        normalize: Si True, normalise les données entre 0 et 1
        fill_nodata: Si True, remplace les valeurs nodata par fill_value
        fill_value: Valeur à utiliser pour remplacer les valeurs nodata
        
    Returns:
        Tuple (données, métadonnées) où données est un tableau NumPy et métadonnées
        est un dictionnaire contenant les informations sur le raster
    """
    try:
        with rasterio.open(raster_path) as src:
            # Lire les données
            if window is None:
                data = src.read(1)
            else:
                data = src.read(1, window=window)
            
            # Créer un masque pour les données valides
            valid_mask = ~np.isnan(data)
            if src.nodata is not None:
                valid_mask &= (data != src.nodata)
            
            # Remplacer les valeurs nodata si demandé
            if fill_nodata:
                data = np.where(valid_mask, data, fill_value)
            
            # Normaliser les données si demandé
            if normalize and np.any(valid_mask):
                min_val = np.min(data[valid_mask])
                max_val = np.max(data[valid_mask])
                
                if max_val > min_val:
                    data = np.where(valid_mask, (data - min_val) / (max_val - min_val), fill_value)
                else:
                    # Si toutes les valeurs sont identiques, mettre à 0 ou 1 selon la valeur
                    if min_val > 0:
                        data = np.where(valid_mask, 1.0, fill_value)
                    else:
                        data = np.where(valid_mask, 0.0, fill_value)
            
            # Préparer les métadonnées
            metadata = {
                "original_path": str(raster_path),
                "shape": data.shape,
                "dtype": str(data.dtype),
                "valid_ratio": float(np.sum(valid_mask) / data.size),
                "window": window._asdict() if window is not None else None,
                "crs": str(src.crs),
                "transform": list(src.transform),
                "nodata": src.nodata,
                "normalized": normalize,
                "filled_nodata": fill_nodata,
                "fill_value": fill_value
            }
            
            if normalize and np.any(valid_mask):
                metadata["original_min"] = float(min_val)
                metadata["original_max"] = float(max_val)
            
            return data, metadata
    
    except Exception as e:
        error_msg = f"Erreur lors de la conversion du raster en NumPy: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def convert_to_geotiff(
    data: np.ndarray,
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    reference_path: Optional[Union[str, Path]] = None,
    nodata_value: Optional[float] = None,
    compress: bool = True
) -> str:
    """
    Convertit un tableau NumPy en fichier GeoTIFF.
    
    Args:
        data: Tableau NumPy à convertir
        output_path: Chemin du fichier de sortie
        metadata: Métadonnées à utiliser (si reference_path n'est pas fourni)
        reference_path: Chemin vers un fichier raster de référence pour les métadonnées
        nodata_value: Valeur à utiliser comme nodata (None = utiliser celle du fichier de référence)
        compress: Si True, compresse le fichier de sortie
        
    Returns:
        Chemin du fichier créé
    """
    try:
        # Préparer le profil
        if reference_path is not None:
            with rasterio.open(reference_path) as src:
                profile = src.profile.copy()
                
                # Mettre à jour le profil avec les nouvelles dimensions si nécessaires
                if data.shape != (src.height, src.width):
                    # Calculer la nouvelle transformation si les dimensions changent
                    # (cela nécessite des informations supplémentaires, comme l'origine)
                    # Pour simplifier, on suppose que les dimensions sont identiques
                    if data.shape != (src.height, src.width):
                        logger.warning(f"Les dimensions du tableau ({data.shape}) ne correspondent pas "
                                      f"à celles du fichier de référence ({src.height}, {src.width}). "
                                      f"La géoréférence peut être incorrecte.")
                
                # Mettre à jour le type de données
                profile.update(
                    dtype=data.dtype.name,
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1
                )
                
                # Mettre à jour la valeur nodata si spécifiée
                if nodata_value is not None:
                    profile.update(nodata=nodata_value)
        
        elif metadata is not None:
            # Créer un profil à partir des métadonnées
            profile = {
                'driver': 'GTiff',
                'height': data.shape[0],
                'width': data.shape[1],
                'count': 1,
                'dtype': data.dtype.name,
                'crs': metadata.get('crs', None),
                'transform': metadata.get('transform', None),
                'nodata': nodata_value if nodata_value is not None else metadata.get('nodata', None)
            }
        
        else:
            # Créer un profil par défaut
            profile = {
                'driver': 'GTiff',
                'height': data.shape[0],
                'width': data.shape[1],
                'count': 1,
                'dtype': data.dtype.name,
                'crs': None,
                'transform': None,
                'nodata': nodata_value
            }
        
        # Ajouter la compression si demandée
        if compress:
            profile.update(
                compress='lzw',
                predictor=2
            )
        
        # Écrire le fichier
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
        
        logger.info(f"Fichier GeoTIFF créé avec succès: {output_path}")
        return str(output_path)
    
    except Exception as e:
        error_msg = f"Erreur lors de la conversion en GeoTIFF: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def extract_raster_window(
    raster_path: Union[str, Path],
    bounds: Optional[Tuple[float, float, float, float]] = None,
    pixel_coords: Optional[Tuple[int, int, int, int]] = None,
    output_path: Optional[Union[str, Path]] = None,
    to_numpy: bool = False
) -> Union[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Extrait une fenêtre d'un raster, soit par coordonnées géographiques, soit par coordonnées en pixels.
    
    Args:
        raster_path: Chemin vers le fichier raster
        bounds: Tuple (left, bottom, right, top) des coordonnées géographiques
        pixel_coords: Tuple (col_start, row_start, width, height) des coordonnées en pixels
        output_path: Chemin du fichier de sortie (None = pas de sauvegarde)
        to_numpy: Si True, retourne un tableau NumPy au lieu d'un chemin de fichier
        
    Returns:
        Si to_numpy=True: Tuple (données, métadonnées)
        Sinon: Chemin du fichier créé
    """
    try:
        with rasterio.open(raster_path) as src:
            # Déterminer la fenêtre à extraire
            if bounds is not None:
                # Convertir les coordonnées géographiques en coordonnées en pixels
                left, bottom, right, top = bounds
                window = src.window(left, bottom, right, top)
            
            elif pixel_coords is not None:
                # Utiliser directement les coordonnées en pixels
                col_start, row_start, width, height = pixel_coords
                window = Window(col_start, row_start, width, height)
            
            else:
                raise ValueError("Vous devez spécifier soit bounds, soit pixel_coords")
            
            # Lire les données
            data = src.read(1, window=window)
            
            # Si on veut juste les données NumPy, les retourner
            if to_numpy:
                return convert_to_numpy(raster_path, window=window)
            
            # Sinon, sauvegarder dans un fichier
            if output_path is None:
                # Générer un nom de fichier par défaut
                base_name = os.path.splitext(os.path.basename(raster_path))[0]
                if bounds is not None:
                    suffix = f"_bounds_{left:.1f}_{bottom:.1f}_{right:.1f}_{top:.1f}"
                else:
                    suffix = f"_window_{col_start}_{row_start}_{width}_{height}"
                
                output_path = os.path.join(os.path.dirname(raster_path), f"{base_name}{suffix}.tif")
            
            # Calculer la transformation pour la fenêtre
            window_transform = src.window_transform(window)
            
            # Créer le profil pour le fichier de sortie
            profile = src.profile.copy()
            profile.update({
                'height': window.height,
                'width': window.width,
                'transform': window_transform
            })
            
            # Écrire le fichier
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            
            logger.info(f"Fenêtre extraite avec succès: {output_path}")
            return str(output_path)
    
    except Exception as e:
        error_msg = f"Erreur lors de l'extraction de la fenêtre: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def create_binary_mask(
    raster_path: Union[str, Path],
    threshold: float,
    output_path: Optional[Union[str, Path]] = None,
    below_threshold: bool = True,
    nodata_value: int = 255
) -> str:
    """
    Crée un masque binaire à partir d'un raster en fonction d'un seuil.
    
    Args:
        raster_path: Chemin vers le fichier raster
        threshold: Seuil pour la création du masque
        output_path: Chemin du fichier de sortie (None = générer automatiquement)
        below_threshold: Si True, les pixels < threshold sont à 1, sinon les pixels >= threshold sont à 1
        nodata_value: Valeur à utiliser pour les pixels nodata
        
    Returns:
        Chemin du fichier créé
    """
    try:
        # Générer un nom de fichier par défaut si nécessaire
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(raster_path))[0]
            direction = "below" if below_threshold else "above"
            output_path = os.path.join(os.path.dirname(raster_path), 
                                      f"{base_name}_mask_{direction}_{threshold}.tif")
        
        with rasterio.open(raster_path) as src:
            # Lire les données
            data = src.read(1)
            
            # Créer un masque pour les données valides
            valid_mask = ~np.isnan(data)
            if src.nodata is not None:
                valid_mask &= (data != src.nodata)
            
            # Créer le masque binaire
            binary_mask = np.full_like(data, nodata_value, dtype=np.uint8)
            
            if below_threshold:
                binary_mask[valid_mask & (data < threshold)] = 1
                binary_mask[valid_mask & (data >= threshold)] = 0
            else:
                binary_mask[valid_mask & (data >= threshold)] = 1
                binary_mask[valid_mask & (data < threshold)] = 0
            
            # Créer le profil pour le fichier de sortie
            profile = src.profile.copy()
            profile.update({
                'dtype': 'uint8',
                'nodata': nodata_value
            })
            
            # Écrire le fichier
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(binary_mask, 1)
            
            # Calculer quelques statistiques
            valid_pixels = np.sum(valid_mask)
            mask_pixels = np.sum(binary_mask[valid_mask] == 1)
            mask_ratio = mask_pixels / valid_pixels if valid_pixels > 0 else 0
            
            logger.info(f"Masque binaire créé avec succès: {output_path}")
            logger.info(f"Pixels valides: {valid_pixels}, Pixels masqués: {mask_pixels} ({mask_ratio:.2%})")
            
            return str(output_path)
    
    except Exception as e:
        error_msg = f"Erreur lors de la création du masque binaire: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
