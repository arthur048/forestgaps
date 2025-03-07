"""
Module de calibration dynamique des DataLoaders pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour calibrer automatiquement les paramètres
des DataLoaders en fonction des ressources disponibles et des caractéristiques des données.
"""

import os
import time
import logging
import platform
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.loaders.optimization import benchmark_dataloader, optimize_batch_size, optimize_num_workers

# Configuration du logger
logger = logging.getLogger(__name__)


class DataLoaderCalibrator:
    """
    Calibrateur automatique pour les DataLoaders.
    
    Cette classe permet de calibrer automatiquement les paramètres des DataLoaders
    en fonction des ressources disponibles et des caractéristiques des données.
    
    Attributes:
        device (torch.device): Périphérique sur lequel les données seront utilisées.
        max_workers (int): Nombre maximum de workers à tester.
        max_batch_size (int): Taille de batch maximale à tester.
        num_batches (int): Nombre de batchs à traiter par test.
        cache_calibration (bool): Si True, met en cache les résultats de calibration.
        cache_dir (str): Répertoire pour le cache de calibration.
        environment_info (Dict): Informations sur l'environnement d'exécution.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        max_workers: int = 16,
        max_batch_size: int = 64,
        num_batches: int = 50,
        cache_calibration: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialise le calibrateur de DataLoader.
        
        Args:
            device: Périphérique sur lequel les données seront utilisées.
            max_workers: Nombre maximum de workers à tester.
            max_batch_size: Taille de batch maximale à tester.
            num_batches: Nombre de batchs à traiter par test.
            cache_calibration: Si True, met en cache les résultats de calibration.
            cache_dir: Répertoire pour le cache de calibration.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_workers = self._adjust_max_workers(max_workers)
        self.max_batch_size = max_batch_size
        self.num_batches = num_batches
        self.cache_calibration = cache_calibration
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".forestgaps_dl", "calibration")
        
        # Créer le répertoire de cache si nécessaire
        if self.cache_calibration:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Collecter les informations sur l'environnement
        self.environment_info = self._collect_environment_info()
        
        logger.info(f"Calibrateur initialisé pour {self.device} avec max_workers={self.max_workers}, "
                   f"max_batch_size={self.max_batch_size}")
    
    def _adjust_max_workers(self, max_workers: int) -> int:
        """
        Ajuste le nombre maximum de workers en fonction de l'environnement.
        
        Args:
            max_workers: Nombre maximum de workers demandé.
            
        Returns:
            Nombre maximum de workers ajusté.
        """
        # Déterminer le nombre de CPU disponibles
        cpu_count = os.cpu_count() or 1
        
        # Limiter le nombre de workers en fonction de l'environnement
        if self._is_colab_environment():
            # Dans Colab, limiter à 2-4 workers pour éviter les problèmes de mémoire
            adjusted = min(max_workers, 4, cpu_count)
            logger.info(f"Environnement Colab détecté: limitation à {adjusted} workers")
            return adjusted
        else:
            # En général, ne pas dépasser le nombre de CPU - 1
            adjusted = min(max_workers, cpu_count - 1) if cpu_count > 1 else 1
            return adjusted
    
    def _is_colab_environment(self) -> bool:
        """
        Détecte si le code s'exécute dans Google Colab.
        
        Returns:
            True si l'environnement est Google Colab, False sinon.
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """
        Collecte des informations sur l'environnement d'exécution.
        
        Returns:
            Dictionnaire contenant les informations sur l'environnement.
        """
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "is_colab": self._is_colab_environment()
        }
        
        # Ajouter des informations sur le GPU si disponible
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory
            })
        
        return info
    
    def _get_cache_key(self, dataset: Dataset) -> str:
        """
        Génère une clé de cache pour un dataset.
        
        Args:
            dataset: Dataset pour lequel générer une clé.
            
        Returns:
            Clé de cache unique pour le dataset.
        """
        # Créer une empreinte basée sur les caractéristiques du dataset et de l'environnement
        dataset_info = {
            "dataset_class": dataset.__class__.__name__,
            "dataset_length": len(dataset),
            "sample_shape": self._get_sample_shape(dataset)
        }
        
        # Combiner avec les informations d'environnement
        cache_info = {
            "dataset": dataset_info,
            "environment": {
                "device": str(self.device),
                "cpu_count": os.cpu_count(),
                "is_colab": self._is_colab_environment()
            }
        }
        
        # Créer une clé de hachage
        import hashlib
        import json
        
        # Convertir en chaîne JSON triée pour assurer la cohérence
        cache_str = json.dumps(cache_info, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_sample_shape(self, dataset: Dataset) -> Tuple:
        """
        Obtient la forme d'un échantillon du dataset.
        
        Args:
            dataset: Dataset à analyser.
            
        Returns:
            Tuple contenant la forme de l'échantillon.
        """
        try:
            sample = dataset[0]
            if isinstance(sample, tuple):
                return tuple(x.shape if hasattr(x, 'shape') else None for x in sample)
            else:
                return sample.shape if hasattr(sample, 'shape') else None
        except Exception as e:
            logger.warning(f"Impossible de déterminer la forme de l'échantillon: {e}")
            return None
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Obtient le chemin du fichier de cache pour une clé donnée.
        
        Args:
            cache_key: Clé de cache.
            
        Returns:
            Chemin du fichier de cache.
        """
        return os.path.join(self.cache_dir, f"calibration_{cache_key}.pt")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Charge les résultats de calibration depuis le cache.
        
        Args:
            cache_key: Clé de cache.
            
        Returns:
            Résultats de calibration ou None si le cache n'existe pas.
        """
        if not self.cache_calibration:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                logger.info(f"Chargement des résultats de calibration depuis {cache_path}")
                return torch.load(cache_path)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du cache: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, results: Dict[str, Any]) -> None:
        """
        Sauvegarde les résultats de calibration dans le cache.
        
        Args:
            cache_key: Clé de cache.
            results: Résultats de calibration.
        """
        if not self.cache_calibration:
            return
        
        cache_path = self._get_cache_path(cache_key)
        try:
            logger.info(f"Sauvegarde des résultats de calibration dans {cache_path}")
            torch.save(results, cache_path)
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde du cache: {e}")
    
    def calibrate(
        self,
        dataset: Dataset,
        plot_results: bool = False,
        force_recalibration: bool = False
    ) -> Dict[str, Any]:
        """
        Calibre les paramètres du DataLoader pour un dataset.
        
        Args:
            dataset: Dataset à calibrer.
            plot_results: Si True, affiche des graphiques des résultats.
            force_recalibration: Si True, force la recalibration même si un cache existe.
            
        Returns:
            Dictionnaire contenant les paramètres optimaux.
        """
        # Vérifier si les résultats sont en cache
        cache_key = self._get_cache_key(dataset)
        if not force_recalibration:
            cached_results = self._load_from_cache(cache_key)
            if cached_results is not None:
                logger.info(f"Utilisation des résultats de calibration en cache")
                return cached_results
        
        logger.info(f"Calibration des paramètres du DataLoader pour {dataset.__class__.__name__}")
        
        # Optimiser le nombre de workers avec une taille de batch moyenne
        initial_batch_size = min(16, self.max_batch_size)
        optimal_workers, worker_results = optimize_num_workers(
            dataset=dataset,
            batch_size=initial_batch_size,
            max_workers=self.max_workers,
            num_batches=self.num_batches,
            pin_memory=True,
            device=self.device,
            plot_results=plot_results
        )
        
        # Optimiser la taille de batch avec le nombre optimal de workers
        optimal_batch_size, batch_results = optimize_batch_size(
            dataset=dataset,
            num_workers=optimal_workers,
            min_batch_size=1,
            max_batch_size=self.max_batch_size,
            num_batches=self.num_batches,
            pin_memory=True,
            device=self.device,
            plot_results=plot_results
        )
        
        # Déterminer les paramètres optimaux
        optimal_params = {
            'batch_size': optimal_batch_size,
            'num_workers': optimal_workers,
            'pin_memory': True,
            'prefetch_factor': 2 if optimal_workers > 0 else None,
            'persistent_workers': True if optimal_workers > 0 else False,
            'device': str(self.device),
            'throughput': optimal_batch_size / batch_results[optimal_batch_size]['avg_batch_time'],
            'calibration_timestamp': time.time(),
            'environment_info': self.environment_info
        }
        
        logger.info(f"Paramètres optimaux: batch_size={optimal_batch_size}, "
                   f"num_workers={optimal_workers}, "
                   f"throughput={optimal_params['throughput']:.1f} échantillons/s")
        
        # Sauvegarder les résultats dans le cache
        self._save_to_cache(cache_key, optimal_params)
        
        return optimal_params
    
    def create_calibrated_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        drop_last: bool = False,
        force_recalibration: bool = False,
        **kwargs
    ) -> DataLoader:
        """
        Crée un DataLoader calibré pour un dataset.
        
        Args:
            dataset: Dataset à utiliser.
            shuffle: Si True, mélange les données.
            drop_last: Si True, ignore le dernier batch s'il est incomplet.
            force_recalibration: Si True, force la recalibration même si un cache existe.
            **kwargs: Arguments supplémentaires pour le DataLoader.
            
        Returns:
            DataLoader calibré pour le dataset.
        """
        # Calibrer les paramètres
        params = self.calibrate(dataset, force_recalibration=force_recalibration)
        
        # Créer le DataLoader avec les paramètres calibrés
        dataloader_params = {
            'batch_size': params['batch_size'],
            'num_workers': params['num_workers'],
            'pin_memory': params['pin_memory'],
            'shuffle': shuffle,
            'drop_last': drop_last
        }
        
        # Ajouter les paramètres spécifiques aux workers si nécessaire
        if params['num_workers'] > 0:
            dataloader_params['prefetch_factor'] = params['prefetch_factor']
            dataloader_params['persistent_workers'] = params['persistent_workers']
        
        # Ajouter les arguments supplémentaires
        dataloader_params.update(kwargs)
        
        # Créer et retourner le DataLoader
        return DataLoader(dataset, **dataloader_params)


def create_calibrated_dataloader(
    dataset: Dataset,
    device: Optional[torch.device] = None,
    max_workers: int = 16,
    max_batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    force_recalibration: bool = False,
    cache_calibration: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Crée un DataLoader calibré pour un dataset.
    
    Args:
        dataset: Dataset à utiliser.
        device: Périphérique sur lequel les données seront utilisées.
        max_workers: Nombre maximum de workers à tester.
        max_batch_size: Taille de batch maximale à tester.
        shuffle: Si True, mélange les données.
        drop_last: Si True, ignore le dernier batch s'il est incomplet.
        force_recalibration: Si True, force la recalibration même si un cache existe.
        cache_calibration: Si True, met en cache les résultats de calibration.
        cache_dir: Répertoire pour le cache de calibration.
        **kwargs: Arguments supplémentaires pour le DataLoader.
        
    Returns:
        DataLoader calibré pour le dataset.
    """
    calibrator = DataLoaderCalibrator(
        device=device,
        max_workers=max_workers,
        max_batch_size=max_batch_size,
        cache_calibration=cache_calibration,
        cache_dir=cache_dir
    )
    
    return calibrator.create_calibrated_dataloader(
        dataset=dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        force_recalibration=force_recalibration,
        **kwargs
    )


def create_calibrated_train_val_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: Optional[torch.device] = None,
    max_workers: int = 16,
    max_batch_size: int = 64,
    force_recalibration: bool = False,
    cache_calibration: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée des DataLoaders calibrés pour l'entraînement et la validation.
    
    Args:
        train_dataset: Dataset d'entraînement.
        val_dataset: Dataset de validation.
        device: Périphérique sur lequel les données seront utilisées.
        max_workers: Nombre maximum de workers à tester.
        max_batch_size: Taille de batch maximale à tester.
        force_recalibration: Si True, force la recalibration même si un cache existe.
        cache_calibration: Si True, met en cache les résultats de calibration.
        cache_dir: Répertoire pour le cache de calibration.
        **kwargs: Arguments supplémentaires pour les DataLoaders.
        
    Returns:
        Tuple (DataLoader d'entraînement, DataLoader de validation).
    """
    calibrator = DataLoaderCalibrator(
        device=device,
        max_workers=max_workers,
        max_batch_size=max_batch_size,
        cache_calibration=cache_calibration,
        cache_dir=cache_dir
    )
    
    # Calibrer pour le dataset d'entraînement
    train_params = calibrator.calibrate(train_dataset, force_recalibration=force_recalibration)
    
    # Créer le DataLoader d'entraînement
    train_loader_params = {
        'batch_size': train_params['batch_size'],
        'num_workers': train_params['num_workers'],
        'pin_memory': train_params['pin_memory'],
        'shuffle': True,
        'drop_last': True
    }
    
    # Ajouter les paramètres spécifiques aux workers si nécessaire
    if train_params['num_workers'] > 0:
        train_loader_params['prefetch_factor'] = train_params['prefetch_factor']
        train_loader_params['persistent_workers'] = train_params['persistent_workers']
    
    # Ajouter les arguments supplémentaires
    train_loader_params.update(kwargs)
    
    # Créer le DataLoader de validation avec les mêmes paramètres mais sans shuffle
    val_loader_params = train_loader_params.copy()
    val_loader_params['shuffle'] = False
    val_loader_params['drop_last'] = False
    
    # Créer et retourner les DataLoaders
    train_loader = DataLoader(train_dataset, **train_loader_params)
    val_loader = DataLoader(val_dataset, **val_loader_params)
    
    return train_loader, val_loader 