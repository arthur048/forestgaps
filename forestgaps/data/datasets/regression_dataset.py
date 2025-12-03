"""
Module pour la gestion des datasets de régression forestière.

Ce module fournit des classes et des fonctions pour créer et manipuler
des datasets pour les tâches de régression sur des données forestières.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# from forestgaps.inference.utils.image_processing import normalize_data  # Module inference pose problème
from forestgaps.data.datasets.transforms import create_transform_pipeline
from forestgaps.utils.errors import DataError

# Fonction normalize_data simplifiée pour éviter les imports problématiques
def normalize_data(data, stats):
    """Normalisation simple min-max."""
    if not stats:
        return data
    min_val = stats.get('min', data.min())
    max_val = stats.get('max', data.max())
    if max_val - min_val > 0:
        return (data - min_val) / (max_val - min_val)
    return data

logger = logging.getLogger(__name__)


class ForestRegressionDataset(Dataset):
    """
    Dataset pour les tâches de régression sur des données forestières.
    
    Ce dataset est conçu pour prédire des valeurs continues (comme le CHM)
    à partir du DSM, plutôt que des masques binaires de trouées.
    """
    
    def __init__(
        self,
        dsm_files: List[str],
        chm_files: List[str],
        thresholds: Optional[List[float]] = None,
        transform=None,
        normalize: bool = True,
        stats: Optional[Dict[str, float]] = None
    ):
        """
        Initialise le dataset de régression forestière.
        
        Args:
            dsm_files: Liste des chemins vers les fichiers DSM.
            chm_files: Liste des chemins vers les fichiers CHM.
            thresholds: Liste des valeurs de seuil (pour le conditionnement).
            transform: Transformations à appliquer aux données.
            normalize: Si True, normalise les données.
            stats: Statistiques pour la normalisation (si None, calculées à partir des données).
        """
        if len(dsm_files) != len(chm_files):
            raise DataError(f"Le nombre de fichiers DSM ({len(dsm_files)}) ne correspond pas "
                           f"au nombre de fichiers CHM ({len(chm_files)})")
        
        self.dsm_files = dsm_files
        self.chm_files = chm_files
        self.thresholds = thresholds
        self.transform = transform
        self.normalize = normalize
        self.stats = stats
        
        # Si pas de statistiques fournies et normalisation activée, calculer les stats
        if self.normalize and self.stats is None:
            self.stats = self._compute_statistics()
            logger.info(f"Statistiques de normalisation calculées: {self.stats}")
    
    def __len__(self):
        """
        Retourne le nombre d'éléments dans le dataset.
        
        Returns:
            Nombre d'éléments.
        """
        return len(self.dsm_files)
    
    def __getitem__(self, idx):
        """
        Retourne un élément du dataset.
        
        Args:
            idx: Indice de l'élément à retourner.
            
        Returns:
            Tuple contenant DSM, CHM et éventuellement le seuil.
        """
        # Charger le DSM
        dsm_path = self.dsm_files[idx]
        dsm = np.load(dsm_path).astype(np.float32)
        
        # Charger le CHM
        chm_path = self.chm_files[idx]
        chm = np.load(chm_path).astype(np.float32)
        
        # Déterminer le seuil (si fourni)
        threshold = None
        if self.thresholds is not None:
            threshold = np.array([self.thresholds[idx]], dtype=np.float32)
        
        # Normaliser si demandé
        if self.normalize and self.stats is not None:
            dsm = normalize_data(dsm, self.stats.get('dsm', {}))
            chm = normalize_data(chm, self.stats.get('chm', {}))
        
        # Appliquer les transformations si définies
        if self.transform:
            dsm, chm = self.transform(dsm, chm)
        
        # Convertir en tenseurs PyTorch
        dsm_tensor = torch.from_numpy(dsm).unsqueeze(0)  # Ajouter dimension des canaux
        chm_tensor = torch.from_numpy(chm).unsqueeze(0)
        
        # Retourner les données
        if threshold is not None:
            threshold_tensor = torch.from_numpy(threshold)
            return dsm_tensor, threshold_tensor, chm_tensor
        else:
            return dsm_tensor, chm_tensor
    
    def _compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calcule les statistiques pour la normalisation.
        
        Returns:
            Dictionnaire contenant les statistiques pour DSM et CHM.
        """
        logger.info("Calcul des statistiques pour la normalisation...")
        
        # Initialiser les variables pour les stats
        dsm_min, dsm_max = float('inf'), float('-inf')
        dsm_sum, dsm_sum_sq = 0, 0
        chm_min, chm_max = float('inf'), float('-inf')
        chm_sum, chm_sum_sq = 0, 0
        total_pixels = 0
        
        # Parcourir une partie du dataset pour calculer les stats
        sample_size = min(len(self), 100)  # Limiter à 100 échantillons pour la performance
        sample_indices = np.random.choice(len(self), sample_size, replace=False)
        
        for idx in sample_indices:
            # Charger les données
            dsm = np.load(self.dsm_files[idx]).astype(np.float32)
            chm = np.load(self.chm_files[idx]).astype(np.float32)
            
            # Mettre à jour les stats pour DSM
            dsm_min = min(dsm_min, np.min(dsm))
            dsm_max = max(dsm_max, np.max(dsm))
            dsm_sum += np.sum(dsm)
            dsm_sum_sq += np.sum(dsm ** 2)
            
            # Mettre à jour les stats pour CHM
            chm_min = min(chm_min, np.min(chm))
            chm_max = max(chm_max, np.max(chm))
            chm_sum += np.sum(chm)
            chm_sum_sq += np.sum(chm ** 2)
            
            # Mettre à jour le nombre total de pixels
            total_pixels += dsm.size
        
        # Calculer les moyennes et écarts-types
        dsm_mean = dsm_sum / total_pixels
        dsm_std = np.sqrt(dsm_sum_sq / total_pixels - dsm_mean ** 2)
        
        chm_mean = chm_sum / total_pixels
        chm_std = np.sqrt(chm_sum_sq / total_pixels - chm_mean ** 2)
        
        # Retourner les statistiques
        return {
            'dsm': {
                'min': float(dsm_min),
                'max': float(dsm_max),
                'mean': float(dsm_mean),
                'std': float(dsm_std)
            },
            'chm': {
                'min': float(chm_min),
                'max': float(chm_max),
                'mean': float(chm_mean),
                'std': float(chm_std)
            }
        }


def create_regression_dataset(
    dsm_files: List[str],
    chm_files: List[str],
    thresholds: Optional[List[float]] = None,
    transform_config: Optional[Dict[str, Any]] = None,
    normalize: bool = True,
    stats: Optional[Dict[str, Dict[str, float]]] = None
) -> ForestRegressionDataset:
    """
    Crée un dataset de régression forestière.
    
    Args:
        dsm_files: Liste des chemins vers les fichiers DSM.
        chm_files: Liste des chemins vers les fichiers CHM.
        thresholds: Liste des valeurs de seuil (pour le conditionnement).
        transform_config: Configuration des transformations.
        normalize: Si True, normalise les données.
        stats: Statistiques pour la normalisation.
        
    Returns:
        Dataset de régression forestière.
    """
    # Créer les transformations
    transform = None
    if transform_config:
        is_train = transform_config.get('is_train', True)
        transform = create_transform_pipeline(
            prob=transform_config.get('prob', 0.5),
            is_train=is_train,
            advanced_aug_prob=transform_config.get('advanced_aug_prob', 0.3),
            enable_elastic=transform_config.get('enable_elastic', False)
        )
    
    # Créer le dataset
    dataset = ForestRegressionDataset(
        dsm_files=dsm_files,
        chm_files=chm_files,
        thresholds=thresholds,
        transform=transform,
        normalize=normalize,
        stats=stats
    )
    
    return dataset


def create_regression_dataloader(
    dataset: ForestRegressionDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Crée un DataLoader pour un dataset de régression forestière.
    
    Args:
        dataset: Dataset de régression forestière.
        batch_size: Taille des batchs.
        shuffle: Si True, mélange les données.
        num_workers: Nombre de workers pour le chargement des données.
        pin_memory: Si True, épingle la mémoire pour transfert plus rapide vers GPU.
        
    Returns:
        DataLoader pour le dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def split_regression_dataset(
    dsm_files: List[str],
    chm_files: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Divise les fichiers DSM et CHM en ensembles d'entraînement, validation et test.
    
    Args:
        dsm_files: Liste des chemins vers les fichiers DSM.
        chm_files: Liste des chemins vers les fichiers CHM.
        train_ratio: Proportion de données pour l'entraînement.
        val_ratio: Proportion de données pour la validation.
        test_ratio: Proportion de données pour le test.
        seed: Graine pour la reproductibilité.
        
    Returns:
        Tuple contenant les listes de fichiers pour train, val et test (DSM puis CHM).
    """
    if len(dsm_files) != len(chm_files):
        raise DataError(f"Le nombre de fichiers DSM ({len(dsm_files)}) ne correspond pas "
                       f"au nombre de fichiers CHM ({len(chm_files)})")
    
    # Vérifier que les ratios somment à 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise DataError(f"Les ratios de division doivent sommer à 1, "
                       f"mais somment à {train_ratio + val_ratio + test_ratio}")
    
    # Mélanger les indices avec une graine fixe pour la reproductibilité
    np.random.seed(seed)
    indices = np.random.permutation(len(dsm_files))
    
    # Calculer les tailles de chaque ensemble
    n_train = int(len(dsm_files) * train_ratio)
    n_val = int(len(dsm_files) * val_ratio)
    
    # Diviser les indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Sélectionner les fichiers
    train_dsm = [dsm_files[i] for i in train_indices]
    train_chm = [chm_files[i] for i in train_indices]
    
    val_dsm = [dsm_files[i] for i in val_indices]
    val_chm = [chm_files[i] for i in val_indices]
    
    test_dsm = [dsm_files[i] for i in test_indices]
    test_chm = [chm_files[i] for i in test_indices]
    
    return train_dsm, train_chm, val_dsm, val_chm, test_dsm, test_chm 