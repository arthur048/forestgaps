"""
Module d'optimisation des DataLoaders pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour optimiser les performances des DataLoaders
pour l'entraînement des modèles de détection des trouées forestières.
"""

import time
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration du logger
logger = logging.getLogger(__name__)

def benchmark_dataloader(
    dataloader: DataLoader,
    num_epochs: int = 2,
    num_batches: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Mesure les performances d'un DataLoader.
    
    Args:
        dataloader: DataLoader à évaluer
        num_epochs: Nombre d'époques à exécuter
        num_batches: Nombre de batchs à traiter par époque (None = tous)
        device: Périphérique sur lequel transférer les données (None = CPU)
        
    Returns:
        Dictionnaire contenant les résultats du benchmark
    """
    # Déterminer le périphérique
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialiser les métriques
    total_time = 0
    data_loading_time = 0
    processing_time = 0
    total_batches = 0
    
    # Exécuter le benchmark
    logger.info(f"Benchmark du DataLoader sur {device}...")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        for i, (inputs, targets) in enumerate(tqdm(dataloader, desc=f"Époque {epoch+1}/{num_epochs}")):
            # Mesurer le temps de chargement des données
            data_loading_end = time.time()
            data_loading_time += data_loading_end - epoch_start
            
            # Transférer les données sur le périphérique
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Simuler un traitement
            _ = inputs + 0.1
            _ = targets + 0.1
            
            # Mesurer le temps de traitement
            processing_end = time.time()
            processing_time += processing_end - data_loading_end
            
            # Mettre à jour le temps de début pour le prochain batch
            epoch_start = time.time()
            
            # Incrémenter le compteur de batchs
            total_batches += 1
            
            # Arrêter si on a atteint le nombre de batchs demandé
            if num_batches is not None and i >= num_batches - 1:
                break
    
    # Calculer les métriques
    total_time = data_loading_time + processing_time
    avg_batch_time = total_time / total_batches if total_batches > 0 else 0
    avg_loading_time = data_loading_time / total_batches if total_batches > 0 else 0
    avg_processing_time = processing_time / total_batches if total_batches > 0 else 0
    
    # Préparer les résultats
    results = {
        'total_time': total_time,
        'data_loading_time': data_loading_time,
        'processing_time': processing_time,
        'total_batches': total_batches,
        'avg_batch_time': avg_batch_time,
        'avg_loading_time': avg_loading_time,
        'avg_processing_time': avg_processing_time,
        'loading_ratio': data_loading_time / total_time if total_time > 0 else 0,
        'processing_ratio': processing_time / total_time if total_time > 0 else 0,
        'dataloader_config': {
            'batch_size': dataloader.batch_size,
            'num_workers': dataloader.num_workers,
            'pin_memory': dataloader.pin_memory,
            'prefetch_factor': getattr(dataloader, 'prefetch_factor', None),
            'persistent_workers': getattr(dataloader, 'persistent_workers', False)
        }
    }
    
    # Afficher les résultats
    logger.info(f"Benchmark terminé: {results['total_batches']} batchs en {results['total_time']:.2f}s")
    logger.info(f"Temps moyen par batch: {results['avg_batch_time']:.4f}s")
    logger.info(f"Temps de chargement: {results['data_loading_time']:.2f}s ({results['loading_ratio']:.1%})")
    logger.info(f"Temps de traitement: {results['processing_time']:.2f}s ({results['processing_ratio']:.1%})")
    
    return results

def optimize_num_workers(
    dataset: Dataset,
    batch_size: int = 8,
    max_workers: int = 16,
    num_batches: int = 50,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
    plot_results: bool = False
) -> Tuple[int, Dict[str, Any]]:
    """
    Détermine le nombre optimal de workers pour un DataLoader.
    
    Args:
        dataset: Dataset à utiliser
        batch_size: Taille des batchs
        max_workers: Nombre maximum de workers à tester
        num_batches: Nombre de batchs à traiter par test
        pin_memory: Si True, utilise la mémoire épinglée
        device: Périphérique sur lequel transférer les données (None = CPU)
        plot_results: Si True, affiche un graphique des résultats
        
    Returns:
        Tuple (nombre optimal de workers, résultats des tests)
    """
    # Déterminer le périphérique
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialiser les résultats
    results = {}
    
    # Tester différents nombres de workers
    worker_options = [0, 1, 2, 4, 6, 8, 12, 16]
    worker_options = [w for w in worker_options if w <= max_workers]
    
    logger.info(f"Optimisation du nombre de workers pour batch_size={batch_size}...")
    
    for num_workers in worker_options:
        # Créer le DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=False
        )
        
        # Mesurer les performances
        logger.info(f"Test avec num_workers={num_workers}...")
        result = benchmark_dataloader(
            dataloader=dataloader,
            num_epochs=1,
            num_batches=num_batches,
            device=device
        )
        
        # Stocker les résultats
        results[num_workers] = result
    
    # Déterminer le nombre optimal de workers
    avg_batch_times = {w: results[w]['avg_batch_time'] for w in worker_options}
    optimal_workers = min(avg_batch_times, key=avg_batch_times.get)
    
    logger.info(f"Nombre optimal de workers: {optimal_workers}")
    logger.info(f"Temps moyen par batch: {avg_batch_times[optimal_workers]:.4f}s")
    
    # Afficher un graphique si demandé
    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(worker_options, [avg_batch_times[w] for w in worker_options], 'o-')
        plt.axvline(x=optimal_workers, color='r', linestyle='--', label=f'Optimal: {optimal_workers}')
        plt.xlabel('Nombre de workers')
        plt.ylabel('Temps moyen par batch (s)')
        plt.title(f'Optimisation du nombre de workers (batch_size={batch_size})')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return optimal_workers, results

def optimize_batch_size(
    dataset: Dataset,
    num_workers: int = 4,
    min_batch_size: int = 1,
    max_batch_size: int = 64,
    num_batches: int = 50,
    pin_memory: bool = True,
    device: Optional[torch.device] = None,
    plot_results: bool = False
) -> Tuple[int, Dict[str, Any]]:
    """
    Détermine la taille de batch optimale pour un DataLoader.
    
    Args:
        dataset: Dataset à utiliser
        num_workers: Nombre de workers à utiliser
        min_batch_size: Taille de batch minimale à tester
        max_batch_size: Taille de batch maximale à tester
        num_batches: Nombre de batchs à traiter par test
        pin_memory: Si True, utilise la mémoire épinglée
        device: Périphérique sur lequel transférer les données (None = CPU)
        plot_results: Si True, affiche un graphique des résultats
        
    Returns:
        Tuple (taille de batch optimale, résultats des tests)
    """
    # Déterminer le périphérique
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialiser les résultats
    results = {}
    
    # Tester différentes tailles de batch
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_sizes = [bs for bs in batch_sizes if min_batch_size <= bs <= max_batch_size]
    
    logger.info(f"Optimisation de la taille de batch pour num_workers={num_workers}...")
    
    for batch_size in batch_sizes:
        # Créer le DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=False
        )
        
        # Mesurer les performances
        logger.info(f"Test avec batch_size={batch_size}...")
        result = benchmark_dataloader(
            dataloader=dataloader,
            num_epochs=1,
            num_batches=num_batches,
            device=device
        )
        
        # Stocker les résultats
        results[batch_size] = result
    
    # Calculer le débit en échantillons par seconde
    throughputs = {bs: bs / results[bs]['avg_batch_time'] for bs in batch_sizes}
    optimal_batch_size = max(throughputs, key=throughputs.get)
    
    logger.info(f"Taille de batch optimale: {optimal_batch_size}")
    logger.info(f"Débit: {throughputs[optimal_batch_size]:.1f} échantillons/s")
    
    # Afficher un graphique si demandé
    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, [throughputs[bs] for bs in batch_sizes], 'o-')
        plt.axvline(x=optimal_batch_size, color='r', linestyle='--', label=f'Optimal: {optimal_batch_size}')
        plt.xlabel('Taille de batch')
        plt.ylabel('Débit (échantillons/s)')
        plt.title(f'Optimisation de la taille de batch (num_workers={num_workers})')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return optimal_batch_size, results

def prefetch_data(
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_batches: Optional[int] = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Précharge les données d'un DataLoader en mémoire.
    
    Args:
        dataloader: DataLoader à précharger
        device: Périphérique sur lequel transférer les données (None = CPU)
        num_batches: Nombre de batchs à précharger (None = tous)
        
    Returns:
        Liste des batchs préchargés
    """
    # Déterminer le périphérique
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Précharger les données
    logger.info(f"Préchargement des données sur {device}...")
    
    prefetched_data = []
    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Préchargement")):
        # Transférer les données sur le périphérique
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Stocker les données
        prefetched_data.append((inputs, targets))
        
        # Arrêter si on a atteint le nombre de batchs demandé
        if num_batches is not None and i >= num_batches - 1:
            break
    
    logger.info(f"Préchargement terminé: {len(prefetched_data)} batchs")
    return prefetched_data

def optimize_dataloader(
    dataset: Dataset,
    device: Optional[torch.device] = None,
    max_workers: int = 16,
    max_batch_size: int = 64,
    num_batches: int = 50,
    plot_results: bool = False
) -> Dict[str, Any]:
    """
    Optimise les paramètres d'un DataLoader.
    
    Args:
        dataset: Dataset à utiliser
        device: Périphérique sur lequel transférer les données (None = CPU)
        max_workers: Nombre maximum de workers à tester
        max_batch_size: Taille de batch maximale à tester
        num_batches: Nombre de batchs à traiter par test
        plot_results: Si True, affiche des graphiques des résultats
        
    Returns:
        Dictionnaire contenant les paramètres optimaux
    """
    # Déterminer le périphérique
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Optimiser le nombre de workers avec une taille de batch moyenne
    initial_batch_size = min(16, max_batch_size)
    optimal_workers, worker_results = optimize_num_workers(
        dataset=dataset,
        batch_size=initial_batch_size,
        max_workers=max_workers,
        num_batches=num_batches,
        pin_memory=True,
        device=device,
        plot_results=plot_results
    )
    
    # Optimiser la taille de batch avec le nombre optimal de workers
    optimal_batch_size, batch_results = optimize_batch_size(
        dataset=dataset,
        num_workers=optimal_workers,
        min_batch_size=1,
        max_batch_size=max_batch_size,
        num_batches=num_batches,
        pin_memory=True,
        device=device,
        plot_results=plot_results
    )
    
    # Préparer les résultats
    optimal_params = {
        'batch_size': optimal_batch_size,
        'num_workers': optimal_workers,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True,
        'device': str(device),
        'throughput': optimal_batch_size / batch_results[optimal_batch_size]['avg_batch_time']
    }
    
    logger.info(f"Paramètres optimaux: {optimal_params}")
    return optimal_params
