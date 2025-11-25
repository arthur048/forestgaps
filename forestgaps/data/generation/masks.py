"""
Module de génération des masques pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour générer des masques binaires
à partir des données raster (DSM/CHM) pour l'entraînement des modèles
de détection des trouées forestières.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, mapping
import cv2
from tqdm import tqdm
from skimage import morphology

# Configuration du logger
logger = logging.getLogger(__name__)

def create_binary_mask(
    chm_path: Union[str, Path],
    threshold: float = 5.0,
    output_path: Optional[Union[str, Path]] = None,
    below_threshold: bool = True,
    min_gap_size: int = 10,
    max_gap_size: Optional[int] = None,
    nodata_value: int = 255
) -> str:
    """
    Crée un masque binaire à partir d'un raster CHM en fonction d'un seuil de hauteur.
    
    Args:
        chm_path: Chemin vers le fichier raster CHM
        threshold: Seuil de hauteur pour la détection des trouées (en mètres)
        output_path: Chemin du fichier de sortie (None = générer automatiquement)
        below_threshold: Si True, les pixels < threshold sont considérés comme des trouées
        min_gap_size: Taille minimale des trouées en pixels
        max_gap_size: Taille maximale des trouées en pixels (None = pas de limite)
        nodata_value: Valeur à utiliser pour les pixels nodata
        
    Returns:
        Chemin du fichier créé
    """
    try:
        # Générer un nom de fichier par défaut si nécessaire
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(chm_path))[0]
            direction = "below" if below_threshold else "above"
            output_path = os.path.join(os.path.dirname(chm_path), 
                                      f"{base_name}_mask_{direction}_{threshold}.tif")
        
        with rasterio.open(chm_path) as src:
            # Lire les données
            chm_data = src.read(1)
            
            # Créer un masque pour les données valides
            valid_mask = ~np.isnan(chm_data)
            if src.nodata is not None:
                valid_mask &= (chm_data != src.nodata)
            
            # Créer le masque binaire initial
            if below_threshold:
                binary_mask = np.where(valid_mask & (chm_data < threshold), 1, 0).astype(np.uint8)
            else:
                binary_mask = np.where(valid_mask & (chm_data >= threshold), 1, 0).astype(np.uint8)
            
            # Appliquer un filtre pour éliminer les petites trouées
            if min_gap_size > 1:
                # Utiliser une ouverture morphologique pour éliminer les petits objets
                binary_mask = morphology.remove_small_objects(
                    binary_mask.astype(bool), 
                    min_size=min_gap_size
                ).astype(np.uint8)
            
            # Appliquer un filtre pour éliminer les grandes trouées si nécessaire
            if max_gap_size is not None and max_gap_size > 0:
                # Identifier les composantes connexes
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                
                # Créer un nouveau masque en excluant les grandes trouées
                filtered_mask = np.zeros_like(binary_mask)
                
                # Parcourir toutes les composantes (sauf le fond qui a l'indice 0)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area <= max_gap_size:
                        filtered_mask[labels == i] = 1
                
                binary_mask = filtered_mask
            
            # Créer le masque final avec les valeurs nodata
            final_mask = np.full_like(chm_data, nodata_value, dtype=np.uint8)
            final_mask[valid_mask] = binary_mask[valid_mask]
            
            # Créer le profil pour le fichier de sortie
            profile = src.profile.copy()
            profile.update({
                'dtype': 'uint8',
                'nodata': nodata_value
            })
            
            # Écrire le fichier
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(final_mask, 1)
            
            # Calculer quelques statistiques
            valid_pixels = np.sum(valid_mask)
            mask_pixels = np.sum(binary_mask[valid_mask])
            mask_ratio = mask_pixels / valid_pixels if valid_pixels > 0 else 0
            
            logger.info(f"Masque binaire créé avec succès: {output_path}")
            logger.info(f"Pixels valides: {valid_pixels}, Pixels masqués: {mask_pixels} ({mask_ratio:.2%})")
            
            return str(output_path)
    
    except Exception as e:
        error_msg = f"Erreur lors de la création du masque binaire: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def refine_mask(
    mask_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    smooth: bool = True,
    fill_holes: bool = True,
    min_hole_size: int = 10,
    erosion_size: int = 0,
    dilation_size: int = 0
) -> str:
    """
    Raffine un masque binaire en appliquant des opérations morphologiques.
    
    Args:
        mask_path: Chemin vers le fichier masque
        output_path: Chemin du fichier de sortie (None = générer automatiquement)
        smooth: Si True, applique un lissage au masque
        fill_holes: Si True, remplit les trous dans les trouées
        min_hole_size: Taille minimale des trous à remplir en pixels
        erosion_size: Taille du noyau pour l'érosion (0 = pas d'érosion)
        dilation_size: Taille du noyau pour la dilatation (0 = pas de dilatation)
        
    Returns:
        Chemin du fichier créé
    """
    try:
        # Générer un nom de fichier par défaut si nécessaire
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            output_path = os.path.join(os.path.dirname(mask_path), f"{base_name}_refined.tif")
        
        with rasterio.open(mask_path) as src:
            # Lire les données
            mask_data = src.read(1)
            
            # Créer un masque pour les données valides
            valid_mask = ~np.isnan(mask_data)
            if src.nodata is not None:
                valid_mask &= (mask_data != src.nodata)
            
            # Extraire le masque binaire (pixels à 1)
            binary_mask = np.where(valid_mask & (mask_data == 1), 1, 0).astype(np.uint8)
            
            # Appliquer un lissage si demandé
            if smooth:
                # Appliquer un filtre médian pour réduire le bruit
                binary_mask = cv2.medianBlur(binary_mask, 3)
            
            # Remplir les trous si demandé
            if fill_holes and min_hole_size > 0:
                # Identifier les composantes connexes du fond
                inverted_mask = 1 - binary_mask
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    inverted_mask, connectivity=8
                )
                
                # Créer un masque pour les trous à remplir
                holes_mask = np.zeros_like(binary_mask)
                
                # Parcourir toutes les composantes (sauf le fond principal qui a généralement la plus grande aire)
                largest_comp_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1 if num_labels > 1 else 0
                
                for i in range(1, num_labels):
                    if i != largest_comp_idx:  # Ne pas considérer le fond principal
                        area = stats[i, cv2.CC_STAT_AREA]
                        if area <= min_hole_size:
                            holes_mask[labels == i] = 1
                
                # Remplir les trous
                binary_mask = np.maximum(binary_mask, holes_mask)
            
            # Appliquer une érosion si demandée
            if erosion_size > 0:
                kernel = np.ones((erosion_size, erosion_size), np.uint8)
                binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
            
            # Appliquer une dilatation si demandée
            if dilation_size > 0:
                kernel = np.ones((dilation_size, dilation_size), np.uint8)
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            
            # Créer le masque final avec les valeurs nodata
            final_mask = np.full_like(mask_data, src.nodata if src.nodata is not None else 255, dtype=np.uint8)
            final_mask[valid_mask] = binary_mask[valid_mask]
            
            # Créer le profil pour le fichier de sortie
            profile = src.profile.copy()
            
            # Écrire le fichier
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(final_mask, 1)
            
            logger.info(f"Masque raffiné créé avec succès: {output_path}")
            return str(output_path)
    
    except Exception as e:
        error_msg = f"Erreur lors du raffinement du masque: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def calculate_mask_statistics(
    mask_path: Union[str, Path],
    output_json: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Calcule des statistiques sur un masque binaire.
    
    Args:
        mask_path: Chemin vers le fichier masque
        output_json: Chemin du fichier JSON de sortie (None = pas de sauvegarde)
        
    Returns:
        Dictionnaire contenant les statistiques du masque
    """
    try:
        with rasterio.open(mask_path) as src:
            # Lire les données
            mask_data = src.read(1)
            
            # Créer un masque pour les données valides
            valid_mask = ~np.isnan(mask_data)
            if src.nodata is not None:
                valid_mask &= (mask_data != src.nodata)
            
            # Extraire le masque binaire (pixels à 1)
            binary_mask = np.where(valid_mask & (mask_data == 1), 1, 0).astype(np.uint8)
            
            # Calculer les statistiques de base
            valid_pixels = np.sum(valid_mask)
            mask_pixels = np.sum(binary_mask)
            mask_ratio = mask_pixels / valid_pixels if valid_pixels > 0 else 0
            
            # Identifier les composantes connexes
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )
            
            # Calculer les statistiques des composantes
            gap_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else []
            
            # Préparer les statistiques
            statistics = {
                'mask_path': str(mask_path),
                'valid_pixels': int(valid_pixels),
                'mask_pixels': int(mask_pixels),
                'mask_ratio': float(mask_ratio),
                'num_gaps': int(num_labels - 1),
                'gap_statistics': {
                    'min_area': int(np.min(gap_areas)) if len(gap_areas) > 0 else 0,
                    'max_area': int(np.max(gap_areas)) if len(gap_areas) > 0 else 0,
                    'mean_area': float(np.mean(gap_areas)) if len(gap_areas) > 0 else 0,
                    'median_area': float(np.median(gap_areas)) if len(gap_areas) > 0 else 0,
                    'total_area': int(np.sum(gap_areas)) if len(gap_areas) > 0 else 0
                },
                'gap_size_distribution': {
                    '0-10': int(np.sum(gap_areas < 10)) if len(gap_areas) > 0 else 0,
                    '10-50': int(np.sum((gap_areas >= 10) & (gap_areas < 50))) if len(gap_areas) > 0 else 0,
                    '50-100': int(np.sum((gap_areas >= 50) & (gap_areas < 100))) if len(gap_areas) > 0 else 0,
                    '100-500': int(np.sum((gap_areas >= 100) & (gap_areas < 500))) if len(gap_areas) > 0 else 0,
                    '500+': int(np.sum(gap_areas >= 500)) if len(gap_areas) > 0 else 0
                }
            }
            
            # Sauvegarder les statistiques si demandé
            if output_json is not None:
                with open(output_json, 'w') as f:
                    json.dump(statistics, f, indent=2)
                logger.info(f"Statistiques du masque sauvegardées: {output_json}")
            
            return statistics
    
    except Exception as e:
        error_msg = f"Erreur lors du calcul des statistiques du masque: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def mask_to_vector(
    mask_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    min_gap_size: int = 10,
    simplify_tolerance: float = 0.5
) -> str:
    """
    Convertit un masque binaire en fichier vectoriel (GeoJSON ou Shapefile).
    
    Args:
        mask_path: Chemin vers le fichier masque
        output_path: Chemin du fichier de sortie (None = générer automatiquement)
        min_gap_size: Taille minimale des trouées en pixels
        simplify_tolerance: Tolérance pour la simplification des polygones
        
    Returns:
        Chemin du fichier créé
    """
    try:
        # Générer un nom de fichier par défaut si nécessaire
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(mask_path))[0]
            output_path = os.path.join(os.path.dirname(mask_path), f"{base_name}_vector.geojson")
        
        with rasterio.open(mask_path) as src:
            # Lire les données
            mask_data = src.read(1)
            
            # Créer un masque pour les données valides
            valid_mask = ~np.isnan(mask_data)
            if src.nodata is not None:
                valid_mask &= (mask_data != src.nodata)
            
            # Extraire le masque binaire (pixels à 1)
            binary_mask = np.where(valid_mask & (mask_data == 1), 1, 0).astype(np.uint8)
            
            # Vectoriser le masque
            results = []
            for geom, value in shapes(binary_mask, mask=binary_mask == 1, transform=src.transform):
                if value == 1:
                    # Convertir en objet Shapely
                    polygon = shape(geom)
                    
                    # Calculer l'aire en pixels
                    area_pixels = polygon.area / (src.transform[0] * src.transform[4])
                    
                    # Filtrer par taille
                    if area_pixels >= min_gap_size:
                        # Simplifier le polygone si nécessaire
                        if simplify_tolerance > 0:
                            polygon = polygon.simplify(simplify_tolerance)
                        
                        # Ajouter à la liste des résultats
                        results.append({
                            'geometry': mapping(polygon),
                            'properties': {
                                'value': int(value),
                                'area_pixels': float(area_pixels),
                                'area_m2': float(polygon.area)
                            }
                        })
            
            # Créer un GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(results, crs=src.crs)
            
            # Sauvegarder le fichier
            if output_path.endswith('.geojson'):
                gdf.to_file(output_path, driver='GeoJSON')
            elif output_path.endswith('.shp'):
                gdf.to_file(output_path)
            else:
                # Par défaut, utiliser GeoJSON
                gdf.to_file(output_path, driver='GeoJSON')
            
            logger.info(f"Masque vectorisé créé avec succès: {output_path}")
            logger.info(f"Nombre de polygones: {len(gdf)}")
            
            return str(output_path)
    
    except Exception as e:
        error_msg = f"Erreur lors de la vectorisation du masque: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def generate_gap_masks(
    chm_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    threshold: float = 5.0,
    min_gap_size: int = 10,
    max_gap_size: Optional[int] = None,
    refine: bool = True,
    vectorize: bool = False,
    save_statistics: bool = True
) -> Dict[str, List[str]]:
    """
    Génère des masques de trouées à partir d'une liste de fichiers CHM.
    
    Args:
        chm_paths: Liste des chemins vers les fichiers CHM
        output_dir: Répertoire de sortie
        threshold: Seuil de hauteur pour la détection des trouées (en mètres)
        min_gap_size: Taille minimale des trouées en pixels
        max_gap_size: Taille maximale des trouées en pixels (None = pas de limite)
        refine: Si True, applique un raffinement aux masques
        vectorize: Si True, convertit les masques en fichiers vectoriels
        save_statistics: Si True, sauvegarde les statistiques des masques
        
    Returns:
        Dictionnaire contenant les chemins des fichiers créés
    """
    try:
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Préparer les listes pour les chemins des fichiers
        mask_paths = []
        refined_paths = []
        vector_paths = []
        stats_paths = []
        
        # Traiter chaque fichier CHM
        for chm_path in tqdm(chm_paths, desc="Génération des masques"):
            # Générer le nom de base pour les fichiers de sortie
            base_name = os.path.splitext(os.path.basename(chm_path))[0]
            
            # Créer le masque binaire
            mask_path = os.path.join(output_dir, f"{base_name}_mask.tif")
            mask_path = create_binary_mask(
                chm_path=chm_path,
                threshold=threshold,
                output_path=mask_path,
                below_threshold=True,
                min_gap_size=min_gap_size,
                max_gap_size=max_gap_size
            )
            mask_paths.append(mask_path)
            
            # Raffiner le masque si demandé
            if refine:
                refined_path = os.path.join(output_dir, f"{base_name}_mask_refined.tif")
                refined_path = refine_mask(
                    mask_path=mask_path,
                    output_path=refined_path,
                    smooth=True,
                    fill_holes=True,
                    min_hole_size=min_gap_size // 2
                )
                refined_paths.append(refined_path)
                
                # Utiliser le masque raffiné pour les étapes suivantes
                current_mask = refined_path
            else:
                current_mask = mask_path
            
            # Vectoriser le masque si demandé
            if vectorize:
                vector_path = os.path.join(output_dir, f"{base_name}_vector.geojson")
                vector_path = mask_to_vector(
                    mask_path=current_mask,
                    output_path=vector_path,
                    min_gap_size=min_gap_size
                )
                vector_paths.append(vector_path)
            
            # Calculer les statistiques si demandé
            if save_statistics:
                stats_path = os.path.join(output_dir, f"{base_name}_stats.json")
                calculate_mask_statistics(
                    mask_path=current_mask,
                    output_json=stats_path
                )
                stats_paths.append(stats_path)
        
        # Préparer le dictionnaire de résultats
        results = {
            'mask_paths': mask_paths,
            'refined_paths': refined_paths,
            'vector_paths': vector_paths,
            'stats_paths': stats_paths
        }
        
        logger.info(f"Génération des masques terminée: {len(mask_paths)} masques créés")
        return results
    
    except Exception as e:
        error_msg = f"Erreur lors de la génération des masques: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
