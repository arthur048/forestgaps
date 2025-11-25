"""
Module d'alignement des rasters pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour vérifier et corriger l'alignement
entre les rasters DSM (Digital Surface Model) et CHM (Canopy Height Model).
"""

import os
import logging
from typing import Dict, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform

# Configuration du logger
logger = logging.getLogger(__name__)

def check_alignment(
    dsm_path: Union[str, Path], 
    chm_path: Union[str, Path], 
    resolution_threshold: float = 0.01,
    bounds_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Vérifie l'alignement entre un DSM et un CHM.
    
    Args:
        dsm_path: Chemin vers le fichier DSM
        chm_path: Chemin vers le fichier CHM
        resolution_threshold: Tolérance pour les différences de résolution
        bounds_threshold: Tolérance pour les différences d'emprise en unités de carte
        
    Returns:
        Dictionnaire contenant les informations d'alignement:
            - aligned (bool): True si les rasters sont alignés
            - issues (list): Liste des problèmes détectés
            - processing_needed (list): Liste des opérations nécessaires pour aligner les rasters
    """
    result = {
        "aligned": True,
        "issues": [],
        "processing_needed": []
    }
    
    try:
        with rasterio.open(dsm_path) as dsm, rasterio.open(chm_path) as chm:
            # Vérification des projections
            if dsm.crs != chm.crs:
                result["aligned"] = False
                result["issues"].append(f"Projections différentes: DSM={dsm.crs}, CHM={chm.crs}")
                result["processing_needed"].append("reproject")
            
            # Vérification des résolutions
            if (abs(dsm.res[0] - chm.res[0]) > resolution_threshold or
                abs(dsm.res[1] - chm.res[1]) > resolution_threshold):
                result["aligned"] = False
                result["issues"].append(f"Résolutions différentes: DSM={dsm.res}, CHM={chm.res}")
                result["processing_needed"].append("resample")
            
            # Vérification des emprises
            if (abs(dsm.bounds.left - chm.bounds.left) > bounds_threshold or
                abs(dsm.bounds.bottom - chm.bounds.bottom) > bounds_threshold or
                abs(dsm.bounds.right - chm.bounds.right) > bounds_threshold or
                abs(dsm.bounds.top - chm.bounds.top) > bounds_threshold):
                result["aligned"] = False
                result["issues"].append("Emprises différentes")
                result["processing_needed"].append("align_extent")
            
            # Vérification des dimensions en pixels
            if dsm.width != chm.width or dsm.height != chm.height:
                result["aligned"] = False
                result["issues"].append(f"Dimensions différentes: DSM={dsm.width}x{dsm.height}, CHM={chm.width}x{chm.height}")
                if "resample" not in result["processing_needed"]:
                    result["processing_needed"].append("resample")
    
    except Exception as e:
        result["aligned"] = False
        result["issues"].append(f"Erreur lors de la vérification de l'alignement: {str(e)}")
    
    return result

def align_rasters(
    dsm_path: Union[str, Path], 
    chm_path: Union[str, Path], 
    output_dir: Optional[Union[str, Path]] = None,
    prefix: Optional[str] = None,
    resampling_method: Resampling = Resampling.bilinear
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Aligne les rasters DSM et CHM pour qu'ils aient la même projection, résolution et emprise.
    Utilise le DSM comme référence pour aligner le CHM.
    
    Args:
        dsm_path: Chemin vers le fichier DSM
        chm_path: Chemin vers le fichier CHM
        output_dir: Répertoire de sortie pour les fichiers alignés (si None, utilise le répertoire du DSM)
        prefix: Préfixe pour les fichiers de sortie (si None, utilise le nom de base du fichier CHM)
        resampling_method: Méthode de rééchantillonnage à utiliser
        
    Returns:
        Tuple (dsm_path_aligned, chm_path_aligned, report) où report est un dictionnaire
        contenant les informations sur l'alignement effectué
    """
    # Vérifier l'alignement actuel
    alignment_check = check_alignment(dsm_path, chm_path)
    
    # Si les rasters sont déjà alignés, rien à faire
    if alignment_check["aligned"]:
        logger.info("Les rasters sont déjà alignés, aucune modification nécessaire.")
        return str(dsm_path), str(chm_path), {"status": "already_aligned", "details": alignment_check}
    
    # Préparer les chemins de sortie
    if output_dir is None:
        output_dir = os.path.dirname(dsm_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(chm_path))[0].replace("_CHM", "")
    
    # Le DSM reste inchangé
    dsm_aligned_path = str(dsm_path)
    
    # Chemin pour le CHM aligné
    chm_aligned_path = os.path.join(output_dir, f"{prefix}_CHM_aligned.tif")
    
    # Rapport d'alignement
    report = {
        "status": "aligned",
        "original_check": alignment_check,
        "processing_applied": [],
        "dsm_path": dsm_aligned_path,
        "chm_path": chm_aligned_path
    }
    
    try:
        # Ouvrir les fichiers source
        with rasterio.open(dsm_path) as dsm_src, rasterio.open(chm_path) as chm_src:
            # Utiliser le DSM comme référence pour le nouveau CHM
            dst_crs = dsm_src.crs
            dst_transform = dsm_src.transform
            dst_width = dsm_src.width
            dst_height = dsm_src.height
            dst_nodata = chm_src.nodata if chm_src.nodata is not None else 0
            
            # Si les projections diffèrent, recalculer la transformation
            if "reproject" in alignment_check["processing_needed"] or "resample" in alignment_check["processing_needed"]:
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    chm_src.crs, dst_crs, dsm_src.width, dsm_src.height,
                    *dsm_src.bounds
                )
                report["processing_applied"].append("reproject_and_resample")
            
            # Préparer les métadonnées pour le CHM aligné
            dst_profile = chm_src.profile.copy()
            dst_profile.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'nodata': dst_nodata
            })
            
            # Créer le fichier CHM aligné
            with rasterio.open(chm_aligned_path, 'w', **dst_profile) as dst:
                # Reprojeter/Réechantillonner le CHM
                reproject(
                    source=rasterio.band(chm_src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=chm_src.transform,
                    src_crs=chm_src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    src_nodata=chm_src.nodata,
                    dst_nodata=dst_nodata,
                    resampling=resampling_method
                )
            
            # Ajouter des détails au rapport
            if "reproject" in alignment_check["processing_needed"]:
                report["processing_applied"].append(f"reprojection de {chm_src.crs} vers {dst_crs}")
            
            if "resample" in alignment_check["processing_needed"]:
                report["processing_applied"].append(f"rééchantillonnage de {chm_src.res} vers {dsm_src.res}")
            
            if "align_extent" in alignment_check["processing_needed"]:
                report["processing_applied"].append("alignement des emprises")
            
            logger.info(f"Alignement réussi: {', '.join(report['processing_applied'])}")
            
    except Exception as e:
        error_msg = f"Erreur lors de l'alignement des rasters: {str(e)}"
        logger.error(error_msg)
        report["status"] = "error"
        report["error"] = error_msg
        return str(dsm_path), str(chm_path), report
    
    return dsm_aligned_path, chm_aligned_path, report
