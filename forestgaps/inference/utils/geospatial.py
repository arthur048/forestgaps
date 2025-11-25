"""
Utilitaires pour les opérations géospatiales.

Ce module fournit des fonctions pour charger, sauvegarder et manipuler
des données géospatiales, en préservant les métadonnées géographiques.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from typing import Dict, Any, Tuple, Optional, Union

# Configuration du logging
logger = logging.getLogger(__name__)

def load_raster(
    file_path: str,
    band: int = 1,
    as_array: bool = True,
    nodata_value: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Charge un fichier raster au format GeoTIFF.
    
    Args:
        file_path: Chemin vers le fichier raster
        band: Index de la bande à charger (1-indexed)
        as_array: Retourner un tableau NumPy si True, sinon un objet rasterio.DatasetReader
        nodata_value: Valeur à utiliser pour les pixels NoData (optionnel)
        
    Returns:
        Tuple contenant les données raster et les métadonnées géospatiales
    
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        ValueError: Si la bande spécifiée n'existe pas
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
    
    try:
        with rasterio.open(file_path) as src:
            # Vérifier que la bande demandée existe
            if band > src.count:
                raise ValueError(f"La bande {band} n'existe pas dans le fichier qui contient {src.count} bandes")
            
            # Extraire les métadonnées
            metadata = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "driver": src.driver,
                "bounds": src.bounds
            }
            
            # Charger les données
            if as_array:
                data = src.read(band)
                
                # Remplacer les valeurs NoData
                if src.nodata is not None or nodata_value is not None:
                    mask_value = src.nodata if src.nodata is not None else nodata_value
                    if mask_value is not None:
                        data = np.where(data == mask_value, np.nan, data)
                
                return data, metadata
            else:
                return src, metadata
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier {file_path}: {str(e)}")
        raise

def save_raster(
    data: np.ndarray,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    format: str = "GTiff",
    dtype: Optional[str] = None,
    nodata_value: Optional[float] = None
) -> str:
    """
    Sauvegarde un tableau NumPy en tant que fichier raster.
    
    Args:
        data: Données à sauvegarder
        output_path: Chemin de sortie pour le fichier raster
        metadata: Métadonnées géospatiales (optionnel)
        format: Format de sortie (par défaut: "GTiff")
        dtype: Type de données de sortie (optionnel)
        nodata_value: Valeur à utiliser pour les pixels NoData (optionnel)
        
    Returns:
        Chemin du fichier sauvegardé
    """
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Déterminer le type de données de sortie
    if dtype is None:
        if data.dtype == np.float32 or data.dtype == np.float64:
            output_dtype = rasterio.float32
        elif data.dtype == np.int32:
            output_dtype = rasterio.int32
        elif data.dtype == np.uint8:
            output_dtype = rasterio.uint8
        else:
            output_dtype = rasterio.float32
    else:
        output_dtype = dtype
    
    # Préparer les métadonnées
    if metadata is None:
        # Créer des métadonnées par défaut
        height, width = data.shape
        transform = rasterio.transform.from_origin(0, 0, 1, 1)
        crs = None
    else:
        height = metadata.get("height", data.shape[0])
        width = metadata.get("width", data.shape[1])
        transform = metadata.get("transform")
        crs = metadata.get("crs")
    
    # Déterminer la valeur NoData
    if nodata_value is None and metadata is not None:
        nodata_value = metadata.get("nodata")
    
    # Vérifier si les données sont 3D (avec canal) ou 2D
    if len(data.shape) == 3:
        count = data.shape[0]
        height, width = data.shape[1], data.shape[2]
    else:
        count = 1
        height, width = data.shape
    
    # Configurations pour la création du fichier
    raster_profile = {
        "driver": format,
        "height": height,
        "width": width,
        "count": count,
        "dtype": output_dtype
    }
    
    if transform is not None:
        raster_profile["transform"] = transform
    
    if crs is not None:
        raster_profile["crs"] = crs
    
    if nodata_value is not None:
        raster_profile["nodata"] = nodata_value
    
    # Sauvegarder les données
    try:
        with rasterio.open(output_path, 'w', **raster_profile) as dst:
            if len(data.shape) == 3:
                for i in range(count):
                    dst.write(data[i], i + 1)
            else:
                dst.write(data, 1)
        
        logger.info(f"Raster sauvegardé dans: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du raster {output_path}: {str(e)}")
        raise

def preserve_metadata(
    source_path: str,
    data: np.ndarray,
    output_path: str,
    update_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Sauvegarde des données en préservant les métadonnées d'un fichier source.
    
    Args:
        source_path: Chemin vers le fichier source contenant les métadonnées à préserver
        data: Données à sauvegarder
        output_path: Chemin de sortie pour le fichier raster
        update_metadata: Métadonnées à mettre à jour (optionnel)
        
    Returns:
        Chemin du fichier sauvegardé
    """
    # Charger les métadonnées du fichier source
    with rasterio.open(source_path) as src:
        metadata = {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "count": 1,  # Nous écrivons une seule bande
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "driver": src.driver
        }
    
    # Mettre à jour les métadonnées si nécessaire
    if update_metadata:
        metadata.update(update_metadata)
    
    # Sauvegarder les données avec les métadonnées préservées
    return save_raster(data, output_path, metadata)

def reproject_raster(
    source_path: str,
    destination_path: str,
    dst_crs: str,
    resampling_method: Resampling = Resampling.nearest
) -> str:
    """
    Reprojette un raster vers un système de coordonnées différent.
    
    Args:
        source_path: Chemin vers le fichier source
        destination_path: Chemin de sortie pour le fichier reprojetté
        dst_crs: Système de coordonnées de destination (format WKT, EPSG, etc.)
        resampling_method: Méthode de rééchantillonnage
        
    Returns:
        Chemin du fichier reprojetté
    """
    try:
        with rasterio.open(source_path) as src:
            # Calculer la transformation pour la reprojection
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            
            # Mettre à jour les métadonnées
            out_meta = src.meta.copy()
            out_meta.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            # Créer le fichier de sortie
            with rasterio.open(destination_path, 'w', **out_meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=resampling_method
                    )
        
        logger.info(f"Raster reprojetté sauvegardé dans: {destination_path}")
        return destination_path
    
    except Exception as e:
        logger.error(f"Erreur lors de la reprojection du raster {source_path}: {str(e)}")
        raise

def get_raster_info(file_path: str) -> Dict[str, Any]:
    """
    Récupère les informations d'un fichier raster.
    
    Args:
        file_path: Chemin vers le fichier raster
        
    Returns:
        Dictionnaire contenant les informations du raster
    """
    try:
        with rasterio.open(file_path) as src:
            info = {
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "crs": str(src.crs),
                "transform": list(src.transform),
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top
                },
                "resolution": {
                    "x": src.res[0],
                    "y": src.res[1]
                },
                "nodata": src.nodata
            }
            
            # Calculer les statistiques de base pour chaque bande
            stats = []
            for i in range(1, src.count + 1):
                band = src.read(i)
                if src.nodata is not None:
                    band = band[band != src.nodata]
                
                if len(band) > 0:
                    band_stats = {
                        "min": float(band.min()),
                        "max": float(band.max()),
                        "mean": float(band.mean()),
                        "std": float(band.std())
                    }
                else:
                    band_stats = {
                        "min": None,
                        "max": None,
                        "mean": None,
                        "std": None
                    }
                
                stats.append(band_stats)
            
            info["stats"] = stats
            
            return info
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations du raster {file_path}: {str(e)}")
        raise 