"""
Module de création de DataLoaders pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour créer des DataLoaders et des Datasets
pour l'entraînement des modèles de détection des trouées forestières.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import rasterio
from tqdm import tqdm

# Configuration du logger
logger = logging.getLogger(__name__)

class ForestGapDataset(Dataset):
    """
    Dataset pour la détection des trouées forestières.
    
    Ce dataset charge des paires d'images (DSM/CHM) et de masques pour l'entraînement
    de modèles de segmentation de trouées forestières.
    """
    
    def __init__(
        self,
        input_paths: List[str],
        mask_paths: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_data: bool = False,
        normalize: bool = True,
        input_bands: List[int] = [0],  # Par défaut, utilise seulement la première bande
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise le dataset.
        
        Args:
            input_paths: Liste des chemins vers les fichiers d'entrée (DSM/CHM)
            mask_paths: Liste des chemins vers les fichiers de masque
            transform: Transformation à appliquer aux images d'entrée
            target_transform: Transformation à appliquer aux masques
            cache_data: Si True, charge toutes les données en mémoire
            normalize: Si True, normalise les données entre 0 et 1
            input_bands: Liste des bandes à utiliser pour l'entrée
            metadata: Métadonnées supplémentaires pour le dataset
        """
        self.input_paths = input_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data
        self.normalize = normalize
        self.input_bands = input_bands
        self.metadata = metadata or {}
        
        # Vérifier que les listes ont la même longueur
        if len(input_paths) != len(mask_paths):
            raise ValueError(f"Les listes input_paths ({len(input_paths)}) et mask_paths ({len(mask_paths)}) "
                            f"doivent avoir la même longueur")
        
        # Initialiser le cache
        self.cached_inputs = {}
        self.cached_masks = {}
        
        # Charger les données en mémoire si demandé
        if cache_data:
            logger.info("Chargement des données en mémoire...")
            for i, (input_path, mask_path) in enumerate(tqdm(zip(input_paths, mask_paths), total=len(input_paths))):
                self.cached_inputs[i] = self._load_input(input_path)
                self.cached_masks[i] = self._load_mask(mask_path)
            logger.info(f"Données chargées en mémoire: {len(self.cached_inputs)} images")
    
    def __len__(self) -> int:
        """Retourne le nombre d'échantillons dans le dataset."""
        return len(self.input_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne un échantillon du dataset.
        
        Args:
            idx: Indice de l'échantillon
            
        Returns:
            Tuple (image d'entrée, masque)
        """
        # Charger les données depuis le cache ou depuis le disque
        if self.cache_data and idx in self.cached_inputs:
            input_data = self.cached_inputs[idx]
            mask_data = self.cached_masks[idx]
        else:
            input_data = self._load_input(self.input_paths[idx])
            mask_data = self._load_mask(self.mask_paths[idx])
        
        # Appliquer les transformations si nécessaire
        if self.transform:
            input_data = self.transform(input_data)
        
        if self.target_transform:
            mask_data = self.target_transform(mask_data)
        
        return input_data, mask_data
    
    def _load_input(self, path: str) -> torch.Tensor:
        """
        Charge une image d'entrée depuis un fichier.
        
        Args:
            path: Chemin vers le fichier
            
        Returns:
            Tensor contenant l'image d'entrée
        """
        try:
            with rasterio.open(path) as src:
                # Lire les bandes spécifiées
                if len(self.input_bands) == 1:
                    # Cas d'une seule bande (CHM ou DSM)
                    data = src.read(self.input_bands[0] + 1)  # rasterio utilise des indices 1-based
                    data = data[np.newaxis, :, :]  # Ajouter une dimension pour les canaux
                else:
                    # Cas multi-bandes (DSM + CHM par exemple)
                    data = np.stack([src.read(b + 1) for b in self.input_bands])
                
                # Gérer les valeurs nodata
                if src.nodata is not None:
                    data = np.where(data == src.nodata, 0, data)
                
                # Normaliser les données si demandé
                if self.normalize:
                    for i in range(data.shape[0]):
                        band = data[i]
                        if np.any(band):  # Éviter la division par zéro
                            min_val = np.min(band[band > 0])
                            max_val = np.max(band)
                            if max_val > min_val:
                                data[i] = (band - min_val) / (max_val - min_val)
                            else:
                                data[i] = np.where(band > 0, 1.0, 0.0)
                
                # Convertir en tensor PyTorch
                tensor = torch.from_numpy(data.astype(np.float32))
                
                return tensor
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'image {path}: {str(e)}")
            # Retourner un tensor vide en cas d'erreur
            return torch.zeros((len(self.input_bands), 256, 256), dtype=torch.float32)
    
    def _load_mask(self, path: str) -> torch.Tensor:
        """
        Charge un masque depuis un fichier.
        
        Args:
            path: Chemin vers le fichier
            
        Returns:
            Tensor contenant le masque
        """
        try:
            with rasterio.open(path) as src:
                # Lire les données
                data = src.read(1)
                
                # Gérer les valeurs nodata
                if src.nodata is not None:
                    data = np.where(data == src.nodata, 0, data)
                
                # Convertir en tensor PyTorch
                tensor = torch.from_numpy(data.astype(np.float32))
                
                # Ajouter une dimension pour les canaux si nécessaire
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0)
                
                return tensor
        
        except Exception as e:
            logger.error(f"Erreur lors du chargement du masque {path}: {str(e)}")
            # Retourner un tensor vide en cas d'erreur
            return torch.zeros((1, 256, 256), dtype=torch.float32)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Retourne les métadonnées du dataset.
        
        Returns:
            Dictionnaire contenant les métadonnées
        """
        metadata = self.metadata.copy()
        metadata.update({
            'num_samples': len(self),
            'input_bands': self.input_bands,
            'normalize': self.normalize,
            'cache_data': self.cache_data
        })
        return metadata

def create_dataset(
    input_paths: List[str],
    mask_paths: List[str],
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    cache_data: bool = False,
    normalize: bool = True,
    input_bands: List[int] = [0],
    metadata: Optional[Dict[str, Any]] = None
) -> ForestGapDataset:
    """
    Crée un dataset pour la détection des trouées forestières.
    
    Args:
        input_paths: Liste des chemins vers les fichiers d'entrée (DSM/CHM)
        mask_paths: Liste des chemins vers les fichiers de masque
        transform: Transformation à appliquer aux images d'entrée
        target_transform: Transformation à appliquer aux masques
        cache_data: Si True, charge toutes les données en mémoire
        normalize: Si True, normalise les données entre 0 et 1
        input_bands: Liste des bandes à utiliser pour l'entrée
        metadata: Métadonnées supplémentaires pour le dataset
        
    Returns:
        Dataset pour la détection des trouées forestières
    """
    return ForestGapDataset(
        input_paths=input_paths,
        mask_paths=mask_paths,
        transform=transform,
        target_transform=target_transform,
        cache_data=cache_data,
        normalize=normalize,
        input_bands=input_bands,
        metadata=metadata
    )

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = False
) -> DataLoader:
    """
    Crée un DataLoader pour un dataset.
    
    Args:
        dataset: Dataset à utiliser
        batch_size: Taille des batchs
        shuffle: Si True, mélange les données
        num_workers: Nombre de workers pour le chargement des données
        pin_memory: Si True, utilise la mémoire épinglée pour accélérer les transferts vers le GPU
        drop_last: Si True, ignore le dernier batch s'il est incomplet
        prefetch_factor: Nombre de batchs à précharger par worker
        persistent_workers: Si True, garde les workers en vie entre les époques
        
    Returns:
        DataLoader pour le dataset
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )

def create_train_val_dataloaders(
    input_paths: List[str],
    mask_paths: List[str],
    val_ratio: float = 0.2,
    batch_size: int = 8,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    cache_data: bool = False,
    normalize: bool = True,
    input_bands: List[int] = [0],
    seed: int = 42,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée des DataLoaders pour l'entraînement et la validation.
    
    Args:
        input_paths: Liste des chemins vers les fichiers d'entrée (DSM/CHM)
        mask_paths: Liste des chemins vers les fichiers de masque
        val_ratio: Ratio des données à utiliser pour la validation
        batch_size: Taille des batchs
        num_workers: Nombre de workers pour le chargement des données
        transform: Transformation à appliquer aux images d'entrée
        target_transform: Transformation à appliquer aux masques
        cache_data: Si True, charge toutes les données en mémoire
        normalize: Si True, normalise les données entre 0 et 1
        input_bands: Liste des bandes à utiliser pour l'entrée
        seed: Graine pour la reproductibilité
        pin_memory: Si True, utilise la mémoire épinglée pour accélérer les transferts vers le GPU
        
    Returns:
        Tuple (DataLoader d'entraînement, DataLoader de validation)
    """
    # Créer le dataset complet
    dataset = create_dataset(
        input_paths=input_paths,
        mask_paths=mask_paths,
        transform=transform,
        target_transform=target_transform,
        cache_data=cache_data,
        normalize=normalize,
        input_bands=input_bands
    )
    
    # Calculer les tailles des ensembles d'entraînement et de validation
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    # Diviser le dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Créer les DataLoaders
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"DataLoaders créés: {train_size} échantillons d'entraînement, {val_size} échantillons de validation")
    return train_loader, val_loader

def load_dataset_from_metadata(
    metadata_path: Union[str, Path],
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    cache_data: bool = False,
    normalize: bool = True,
    input_bands: List[int] = [0]
) -> ForestGapDataset:
    """
    Charge un dataset à partir d'un fichier de métadonnées.
    
    Args:
        metadata_path: Chemin vers le fichier de métadonnées
        transform: Transformation à appliquer aux images d'entrée
        target_transform: Transformation à appliquer aux masques
        cache_data: Si True, charge toutes les données en mémoire
        normalize: Si True, normalise les données entre 0 et 1
        input_bands: Liste des bandes à utiliser pour l'entrée
        
    Returns:
        Dataset pour la détection des trouées forestières
    """
    try:
        # Charger les métadonnées
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extraire les chemins des fichiers
        input_paths = metadata.get('input_paths', [])
        mask_paths = metadata.get('mask_paths', [])
        
        # Vérifier que les listes ne sont pas vides
        if not input_paths or not mask_paths:
            raise ValueError("Les listes input_paths et mask_paths ne peuvent pas être vides")
        
        # Créer le dataset
        return create_dataset(
            input_paths=input_paths,
            mask_paths=mask_paths,
            transform=transform,
            target_transform=target_transform,
            cache_data=cache_data,
            normalize=normalize,
            input_bands=input_bands,
            metadata=metadata
        )
    
    except Exception as e:
        error_msg = f"Erreur lors du chargement du dataset à partir des métadonnées: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
