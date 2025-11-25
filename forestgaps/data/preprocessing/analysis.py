"""
Module d'analyse des rasters pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour analyser les rasters DSM et CHM,
vérifier leur intégrité et générer des rapports détaillés sur leurs caractéristiques.
"""

import os
import logging
from typing import Dict, Any, Union, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

# Configuration du logger
logger = logging.getLogger(__name__)

def verify_raster_integrity(
    raster_path: Union[str, Path],
    check_data: bool = True,
    sample_size: int = 10
) -> Dict[str, Any]:
    """
    Vérifie l'intégrité d'un fichier raster.
    
    Args:
        raster_path: Chemin vers le fichier raster
        check_data: Si True, vérifie également les données (pas seulement les métadonnées)
        sample_size: Taille de l'échantillon à lire pour vérifier les données
        
    Returns:
        Dictionnaire contenant les informations sur l'intégrité du raster:
            - valid (bool): True si le raster est valide
            - issues (list): Liste des problèmes détectés
            - metadata (dict): Métadonnées du raster si valide
    """
    result = {
        "valid": True,
        "issues": [],
        "metadata": {}
    }
    
    try:
        with rasterio.open(raster_path) as src:
            # Vérifier les métadonnées de base
            result["metadata"] = {
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs),
                "transform": list(src.transform),
                "nodata": src.nodata,
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top
                },
                "resolution": src.res
            }
            
            # Vérifier les dimensions
            if src.width <= 0 or src.height <= 0:
                result["valid"] = False
                result["issues"].append(f"Dimensions invalides: {src.width}x{src.height}")
            
            # Vérifier le nombre de bandes
            if src.count <= 0:
                result["valid"] = False
                result["issues"].append(f"Nombre de bandes invalide: {src.count}")
            
            # Vérifier les données si demandé
            if check_data:
                try:
                    # Lire un petit échantillon pour vérifier l'intégrité des données
                    window_size = min(sample_size, src.width, src.height)
                    sample = src.read(1, window=rasterio.windows.Window(0, 0, window_size, window_size))
                    
                    # Vérifier si l'échantillon contient uniquement des valeurs NaN ou nodata
                    valid_mask = ~np.isnan(sample)
                    if src.nodata is not None:
                        valid_mask &= (sample != src.nodata)
                    
                    if not np.any(valid_mask):
                        result["valid"] = False
                        result["issues"].append("Le raster ne contient aucune donnée valide dans l'échantillon")
                
                except Exception as e:
                    result["valid"] = False
                    result["issues"].append(f"Erreur lors de la lecture des données: {str(e)}")
    
    except rasterio.errors.RasterioIOError as e:
        result["valid"] = False
        result["issues"].append(f"Erreur d'ouverture du fichier: {str(e)}")
    
    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Erreur inattendue: {str(e)}")
    
    return result

def analyze_raster_pair(
    dsm_path: Union[str, Path],
    chm_path: Union[str, Path],
    prefix: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyse une paire de rasters DSM/CHM et génère un rapport détaillé.
    
    Args:
        dsm_path: Chemin vers le fichier DSM
        chm_path: Chemin vers le fichier CHM
        prefix: Préfixe du site (si None, extrait du nom de fichier)
        
    Returns:
        Dictionnaire contenant le rapport d'analyse
    """
    # Déterminer le préfixe si non fourni
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(dsm_path))[0].replace("_DSM", "")
    
    # Initialiser le rapport
    report = {
        'site_id': prefix,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'valid': True,
        'warnings': [],
        'errors': [],
        'metadata': {'dsm': {}, 'chm': {}},
        'statistics': {'dsm': {}, 'chm': {}},
        'alignment_status': 'aligned',
        'processing_needed': [],
        'processing_history': []
    }
    
    try:
        # Vérifier l'intégrité des fichiers
        dsm_integrity = verify_raster_integrity(dsm_path)
        chm_integrity = verify_raster_integrity(chm_path)
        
        if not dsm_integrity["valid"]:
            report['valid'] = False
            for issue in dsm_integrity["issues"]:
                report['errors'].append(f"DSM: {issue}")
        
        if not chm_integrity["valid"]:
            report['valid'] = False
            for issue in chm_integrity["issues"]:
                report['errors'].append(f"CHM: {issue}")
        
        # Si l'un des fichiers n'est pas valide, arrêter l'analyse
        if not report['valid']:
            return report
        
        # Analyse du DSM
        with rasterio.open(dsm_path) as dsm:
            dsm_data = dsm.read(1)
            
            # Métadonnées
            report['metadata']['dsm'] = {
                'path': str(dsm_path),
                'shape': (dsm.width, dsm.height),
                'resolution': dsm.res,
                'crs': str(dsm.crs),
                'bounds': {
                    'left': dsm.bounds.left,
                    'bottom': dsm.bounds.bottom,
                    'right': dsm.bounds.right,
                    'top': dsm.bounds.top
                },
                'transform': list(dsm.transform),
                'nodata': dsm.nodata,
                'driver': dsm.driver
            }
            
            # Statistiques basiques
            valid_mask = ~np.isnan(dsm_data)
            if dsm.nodata is not None:
                valid_mask &= (dsm_data != dsm.nodata)
            
            if np.any(valid_mask):
                report['statistics']['dsm'] = {
                    'min': float(np.min(dsm_data[valid_mask])),
                    'max': float(np.max(dsm_data[valid_mask])),
                    'mean': float(np.mean(dsm_data[valid_mask])),
                    'std': float(np.std(dsm_data[valid_mask])),
                    'valid_ratio': float(np.sum(valid_mask) / dsm_data.size)
                }
            else:
                report['statistics']['dsm'] = {
                    'valid_ratio': 0.0
                }
                report['errors'].append("Le DSM ne contient aucune donnée valide")
                report['valid'] = False
        
        # Analyse du CHM
        with rasterio.open(chm_path) as chm:
            chm_data = chm.read(1)
            
            # Métadonnées
            report['metadata']['chm'] = {
                'path': str(chm_path),
                'shape': (chm.width, chm.height),
                'resolution': chm.res,
                'crs': str(chm.crs),
                'bounds': {
                    'left': chm.bounds.left,
                    'bottom': chm.bounds.bottom,
                    'right': chm.bounds.right,
                    'top': chm.bounds.top
                },
                'transform': list(chm.transform),
                'nodata': chm.nodata,
                'driver': chm.driver
            }
            
            # Statistiques basiques
            valid_mask = ~np.isnan(chm_data)
            if chm.nodata is not None:
                valid_mask &= (chm_data != chm.nodata)
            
            if np.any(valid_mask):
                report['statistics']['chm'] = {
                    'min': float(np.min(chm_data[valid_mask])),
                    'max': float(np.max(chm_data[valid_mask])),
                    'mean': float(np.mean(chm_data[valid_mask])),
                    'std': float(np.std(chm_data[valid_mask])),
                    'valid_ratio': float(np.sum(valid_mask) / chm_data.size)
                }
            else:
                report['statistics']['chm'] = {
                    'valid_ratio': 0.0
                }
                report['errors'].append("Le CHM ne contient aucune donnée valide")
                report['valid'] = False
        
        # Vérification des projections
        with rasterio.open(dsm_path) as dsm, rasterio.open(chm_path) as chm:
            if dsm.crs != chm.crs:
                report['warnings'].append(f"Projections différentes: DSM={dsm.crs}, CHM={chm.crs}")
                report['alignment_status'] = 'misaligned'
                report['processing_needed'].append('reproject')
            
            # Vérifier les résolutions
            resolution_threshold = 0.01  # Tolérance pour les différences de résolution
            if (abs(dsm.res[0] - chm.res[0]) > resolution_threshold or
                abs(dsm.res[1] - chm.res[1]) > resolution_threshold):
                report['warnings'].append(f"Résolutions différentes: DSM={dsm.res}, CHM={chm.res}")
                report['alignment_status'] = 'misaligned'
                report['processing_needed'].append('resample')
            
            # Vérifier les emprises
            bounds_threshold = 1.0  # Tolérance pour les différences d'emprise en unités de carte
            if (abs(dsm.bounds.left - chm.bounds.left) > bounds_threshold or
                abs(dsm.bounds.bottom - chm.bounds.bottom) > bounds_threshold or
                abs(dsm.bounds.right - chm.bounds.right) > bounds_threshold or
                abs(dsm.bounds.top - chm.bounds.top) > bounds_threshold):
                report['warnings'].append("Emprises différentes")
                report['alignment_status'] = 'misaligned'
                report['processing_needed'].append('align_extent')
            
            # Vérifier des dimensions en pixels
            if dsm.width != chm.width or dsm.height != chm.height:
                report['warnings'].append(f"Dimensions différentes: DSM={dsm.width}x{dsm.height}, CHM={chm.width}x{chm.height}")
                if 'resample' not in report['processing_needed']:
                    report['processing_needed'].append('resample')
    
    except Exception as e:
        report['valid'] = False
        report['errors'].append(f"Erreur lors de l'analyse: {str(e)}")
        logger.error(f"Erreur lors de l'analyse de la paire {prefix}: {str(e)}")
    
    return report

def calculate_raster_statistics(
    raster_path: Union[str, Path],
    sample_size: Optional[int] = None,
    percentiles: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Calcule des statistiques détaillées sur un raster.
    
    Args:
        raster_path: Chemin vers le fichier raster
        sample_size: Taille de l'échantillon à utiliser (None = utiliser toutes les données)
        percentiles: Liste des percentiles à calculer (None = [5, 25, 50, 75, 95])
        
    Returns:
        Dictionnaire contenant les statistiques calculées
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]
    
    stats = {
        "basic": {},
        "percentiles": {},
        "histogram": {},
        "valid_data": {}
    }
    
    try:
        with rasterio.open(raster_path) as src:
            # Lire les données (échantillon ou complet)
            if sample_size is not None and (src.width > sample_size or src.height > sample_size):
                # Calculer un pas d'échantillonnage pour couvrir l'ensemble du raster
                row_step = max(1, src.height // sample_size)
                col_step = max(1, src.width // sample_size)
                
                # Lire un échantillon régulier
                indices = np.mgrid[0:src.height:row_step, 0:src.width:col_step]
                rows, cols = indices[0].flatten(), indices[1].flatten()
                
                # Limiter le nombre de points si nécessaire
                if len(rows) > sample_size * sample_size:
                    idx = np.random.choice(len(rows), sample_size * sample_size, replace=False)
                    rows, cols = rows[idx], cols[idx]
                
                # Lire les valeurs aux positions échantillonnées
                data = np.array([src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0] 
                                for row, col in zip(rows, cols)])
                
                stats["sampling"] = {
                    "method": "regular_grid",
                    "sample_size": len(data),
                    "original_size": src.width * src.height,
                    "sampling_ratio": len(data) / (src.width * src.height)
                }
            else:
                # Lire toutes les données
                data = src.read(1)
                stats["sampling"] = {
                    "method": "full",
                    "sample_size": data.size,
                    "original_size": data.size,
                    "sampling_ratio": 1.0
                }
            
            # Créer un masque pour les données valides
            valid_mask = ~np.isnan(data)
            if src.nodata is not None:
                valid_mask &= (data != src.nodata)
            
            valid_data = data[valid_mask]
            
            if len(valid_data) > 0:
                # Statistiques de base
                stats["basic"] = {
                    "min": float(np.min(valid_data)),
                    "max": float(np.max(valid_data)),
                    "mean": float(np.mean(valid_data)),
                    "median": float(np.median(valid_data)),
                    "std": float(np.std(valid_data)),
                    "var": float(np.var(valid_data))
                }
                
                # Percentiles
                stats["percentiles"] = {
                    f"p{p}": float(np.percentile(valid_data, p)) for p in percentiles
                }
                
                # Histogramme
                hist, bin_edges = np.histogram(valid_data, bins='auto')
                stats["histogram"] = {
                    "counts": hist.tolist(),
                    "bin_edges": bin_edges.tolist()
                }
                
                # Informations sur les données valides
                stats["valid_data"] = {
                    "count": int(np.sum(valid_mask)),
                    "ratio": float(np.sum(valid_mask) / data.size)
                }
            else:
                stats["error"] = "Aucune donnée valide trouvée"
    
    except Exception as e:
        stats["error"] = f"Erreur lors du calcul des statistiques: {str(e)}"
        logger.error(f"Erreur lors du calcul des statistiques pour {raster_path}: {str(e)}")
    
    return stats
