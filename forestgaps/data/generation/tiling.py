"""
Module de tuilage des rasters pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour découper les rasters en tuiles
de taille fixe pour l'entraînement des modèles de détection des trouées forestières.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import uuid

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Configuration du logger
logger = logging.getLogger(__name__)

def create_tile_grid(
    raster_path: Union[str, Path],
    tile_size: int = 256,
    overlap: int = 0,
    min_valid_ratio: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Crée une grille de tuiles pour un raster.
    
    Args:
        raster_path: Chemin vers le fichier raster
        tile_size: Taille des tuiles en pixels
        overlap: Chevauchement entre les tuiles en pixels
        min_valid_ratio: Ratio minimum de pixels valides pour qu'une tuile soit conservée
        
    Returns:
        Liste de dictionnaires contenant les informations sur chaque tuile
    """
    try:
        with rasterio.open(raster_path) as src:
            # Récupérer les dimensions du raster
            height, width = src.height, src.width
            
            # Calculer le pas entre les tuiles
            step = tile_size - overlap
            
            # Créer la liste des tuiles
            tiles = []
            
            # Parcourir le raster par pas de 'step'
            for y in range(0, height, step):
                for x in range(0, width, step):
                    # Ajuster la taille de la tuile si elle dépasse les limites du raster
                    actual_width = min(tile_size, width - x)
                    actual_height = min(tile_size, height - y)
                    
                    # Ignorer les tuiles trop petites
                    if actual_width < tile_size / 2 or actual_height < tile_size / 2:
                        continue
                    
                    # Créer la fenêtre pour la tuile
                    window = Window(x, y, actual_width, actual_height)
                    
                    # Lire les données pour vérifier la validité
                    data = src.read(1, window=window)
                    
                    # Créer un masque pour les données valides
                    valid_mask = ~np.isnan(data)
                    if src.nodata is not None:
                        valid_mask &= (data != src.nodata)
                    
                    # Calculer le ratio de pixels valides
                    valid_ratio = np.sum(valid_mask) / (actual_width * actual_height)
                    
                    # Ignorer les tuiles avec trop peu de pixels valides
                    if valid_ratio < min_valid_ratio:
                        continue
                    
                    # Calculer les coordonnées géographiques de la tuile
                    bounds = rasterio.windows.bounds(window, src.transform)
                    
                    # Ajouter la tuile à la liste
                    tiles.append({
                        'id': str(uuid.uuid4()),
                        'raster_path': str(raster_path),
                        'window': {
                            'col_off': int(window.col_off),
                            'row_off': int(window.row_off),
                            'width': int(window.width),
                            'height': int(window.height)
                        },
                        'bounds': {
                            'left': bounds[0],
                            'bottom': bounds[1],
                            'right': bounds[2],
                            'top': bounds[3]
                        },
                        'valid_ratio': float(valid_ratio),
                        'tile_size': tile_size,
                        'actual_size': (int(actual_width), int(actual_height))
                    })
            
            logger.info(f"Grille de tuiles créée pour {raster_path}: {len(tiles)} tuiles")
            return tiles
    
    except Exception as e:
        error_msg = f"Erreur lors de la création de la grille de tuiles: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def extract_tile(
    raster_path: Union[str, Path],
    window: Dict[str, int],
    output_path: Optional[Union[str, Path]] = None,
    to_numpy: bool = False
) -> Union[str, np.ndarray]:
    """
    Extrait une tuile d'un raster.
    
    Args:
        raster_path: Chemin vers le fichier raster
        window: Dictionnaire contenant les informations sur la fenêtre à extraire
        output_path: Chemin du fichier de sortie (None = pas de sauvegarde)
        to_numpy: Si True, retourne un tableau NumPy au lieu d'un chemin de fichier
        
    Returns:
        Si to_numpy=True: Tableau NumPy
        Sinon: Chemin du fichier créé
    """
    try:
        # Créer la fenêtre
        win = Window(
            window['col_off'],
            window['row_off'],
            window['width'],
            window['height']
        )
        
        with rasterio.open(raster_path) as src:
            # Lire les données
            data = src.read(1, window=win)
            
            # Si on veut juste les données NumPy, les retourner
            if to_numpy:
                return data
            
            # Sinon, sauvegarder dans un fichier
            if output_path is None:
                # Générer un nom de fichier par défaut
                base_name = os.path.splitext(os.path.basename(raster_path))[0]
                suffix = f"_tile_{window['col_off']}_{window['row_off']}_{window['width']}_{window['height']}"
                output_path = os.path.join(os.path.dirname(raster_path), f"{base_name}{suffix}.tif")
            
            # Calculer la transformation pour la fenêtre
            window_transform = src.window_transform(win)
            
            # Créer le profil pour le fichier de sortie
            profile = src.profile.copy()
            profile.update({
                'height': win.height,
                'width': win.width,
                'transform': window_transform
            })
            
            # Écrire le fichier
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
            
            return str(output_path)
    
    except Exception as e:
        error_msg = f"Erreur lors de l'extraction de la tuile: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def generate_tiles(
    raster_path: Union[str, Path],
    output_dir: Union[str, Path],
    tile_size: int = 256,
    overlap: int = 0,
    min_valid_ratio: float = 0.5,
    prefix: Optional[str] = None,
    save_metadata: bool = True
) -> Tuple[List[str], str]:
    """
    Génère des tuiles à partir d'un raster.
    
    Args:
        raster_path: Chemin vers le fichier raster
        output_dir: Répertoire de sortie
        tile_size: Taille des tuiles en pixels
        overlap: Chevauchement entre les tuiles en pixels
        min_valid_ratio: Ratio minimum de pixels valides pour qu'une tuile soit conservée
        prefix: Préfixe pour les noms de fichiers
        save_metadata: Si True, sauvegarde les métadonnées des tuiles
        
    Returns:
        Tuple (liste des chemins des tuiles, chemin du fichier de métadonnées)
    """
    try:
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Créer la grille de tuiles
        tiles = create_tile_grid(
            raster_path=raster_path,
            tile_size=tile_size,
            overlap=overlap,
            min_valid_ratio=min_valid_ratio
        )
        
        # Générer le préfixe si nécessaire
        if prefix is None:
            prefix = os.path.splitext(os.path.basename(raster_path))[0]
        
        # Extraire les tuiles
        tile_paths = []
        
        for i, tile in enumerate(tqdm(tiles, desc="Génération des tuiles")):
            # Générer le nom du fichier
            output_path = os.path.join(
                output_dir,
                f"{prefix}_tile_{i:04d}.tif"
            )
            
            # Extraire la tuile
            tile_path = extract_tile(
                raster_path=raster_path,
                window=tile['window'],
                output_path=output_path
            )
            
            # Ajouter le chemin à la liste
            tile_paths.append(tile_path)
            
            # Mettre à jour les métadonnées
            tile['output_path'] = tile_path
        
        # Sauvegarder les métadonnées si demandé
        metadata_path = ""
        if save_metadata:
            metadata_path = save_tile_metadata(
                tiles=tiles,
                output_dir=output_dir,
                prefix=prefix
            )
        
        logger.info(f"Génération des tuiles terminée: {len(tile_paths)} tuiles créées")
        return tile_paths, metadata_path
    
    except Exception as e:
        error_msg = f"Erreur lors de la génération des tuiles: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def save_tile_metadata(
    tiles: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    prefix: Optional[str] = None
) -> str:
    """
    Sauvegarde les métadonnées des tuiles dans un fichier JSON.
    
    Args:
        tiles: Liste des métadonnées des tuiles
        output_dir: Répertoire de sortie
        prefix: Préfixe pour le nom du fichier
        
    Returns:
        Chemin du fichier de métadonnées
    """
    try:
        # Générer le préfixe si nécessaire
        if prefix is None:
            prefix = "tiles"
        
        # Générer le nom du fichier
        output_path = os.path.join(output_dir, f"{prefix}_metadata.json")
        
        # Sauvegarder les métadonnées
        with open(output_path, 'w') as f:
            json.dump({
                'tiles': tiles,
                'count': len(tiles),
                'metadata_version': '1.0'
            }, f, indent=2)
        
        logger.info(f"Métadonnées des tuiles sauvegardées: {output_path}")
        return str(output_path)
    
    except Exception as e:
        error_msg = f"Erreur lors de la sauvegarde des métadonnées des tuiles: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def load_tile_metadata(
    metadata_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Charge les métadonnées des tuiles depuis un fichier JSON.
    
    Args:
        metadata_path: Chemin du fichier de métadonnées
        
    Returns:
        Dictionnaire contenant les métadonnées des tuiles
    """
    try:
        # Charger les métadonnées
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Métadonnées des tuiles chargées: {metadata_path}")
        return metadata
    
    except Exception as e:
        error_msg = f"Erreur lors du chargement des métadonnées des tuiles: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def merge_tiles(
    tile_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    reference_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Fusionne des tuiles en un seul raster.
    
    Args:
        tile_paths: Liste des chemins des tuiles
        output_path: Chemin du fichier de sortie
        reference_path: Chemin vers un fichier raster de référence pour les métadonnées
        
    Returns:
        Chemin du fichier créé
    """
    # Cette fonction est plus complexe et nécessite l'utilisation de rasterio.merge
    # Pour simplifier, on ne l'implémente pas ici
    logger.warning("La fonction merge_tiles n'est pas encore implémentée")
    return ""
