"""
Outils de benchmarking pour ForestGaps.

Ce module fournit des fonctions pour mesurer et analyser les performances
des différentes parties du workflow ForestGaps.
"""

import time
import functools
import torch
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Callable, Any, Optional, Union, Tuple


class Timer:
    """Classe utilitaire pour mesurer le temps d'exécution."""
    
    def __init__(self, name: str = ""):
        """
        Initialise un timer.
        
        Args:
            name (str): Nom du timer pour l'identification.
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
        
    def start(self) -> None:
        """Démarre le timer."""
        self.start_time = time.time()
        
    def stop(self) -> float:
        """
        Arrête le timer et retourne le temps écoulé.
        
        Returns:
            float: Temps écoulé en secondes.
        """
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
        
    def __enter__(self) -> 'Timer':
        """Support pour l'utilisation comme context manager."""
        self.start()
        return self
        
    def __exit__(self, *args) -> None:
        """Support pour l'utilisation comme context manager."""
        self.stop()


def timeit(func: Callable) -> Callable:
    """
    Décorateur pour mesurer le temps d'exécution d'une fonction.
    
    Args:
        func (Callable): Fonction à mesurer.
        
    Returns:
        Callable: Fonction décorée.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer(func.__name__)
        timer.start()
        result = func(*args, **kwargs)
        elapsed = timer.stop()
        print(f"Fonction {func.__name__} exécutée en {elapsed:.4f} secondes")
        return result
    return wrapper


@contextmanager
def profile_section(name: str) -> None:
    """
    Context manager pour profiler une section de code.
    
    Args:
        name (str): Nom de la section à profiler.
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"Section '{name}' exécutée en {elapsed:.4f} secondes")


def benchmark_transfers(tensor_size: Tuple[int, ...] = (1, 3, 256, 256), 
                        repetitions: int = 100) -> Dict[str, float]:
    """
    Compare différentes méthodes de transfert CPU→GPU.
    
    Args:
        tensor_size (tuple): Taille du tenseur à transférer.
        repetitions (int): Nombre de répétitions pour le benchmark.
        
    Returns:
        dict: Résultats du benchmark pour chaque méthode.
    """
    if not torch.cuda.is_available():
        print("CUDA n'est pas disponible, benchmark de transfert ignoré.")
        return {}
    
    results = {}
    
    # Méthode 1: Standard
    timer = Timer("standard")
    timer.start()
    for _ in range(repetitions):
        tensor = torch.randn(tensor_size)
        tensor_gpu = tensor.to('cuda')
        _ = tensor_gpu.sum().item()  # Force synchronization
    results["standard"] = timer.stop() / repetitions
    
    # Méthode 2: pin_memory
    timer = Timer("pin_memory")
    timer.start()
    for _ in range(repetitions):
        tensor = torch.randn(tensor_size, pin_memory=True)
        tensor_gpu = tensor.to('cuda')
        _ = tensor_gpu.sum().item()  # Force synchronization
    results["pin_memory"] = timer.stop() / repetitions
    
    # Méthode 3: pin_memory + non_blocking
    timer = Timer("pin_memory_non_blocking")
    timer.start()
    for _ in range(repetitions):
        tensor = torch.randn(tensor_size, pin_memory=True)
        tensor_gpu = tensor.to('cuda', non_blocking=True)
        _ = tensor_gpu.sum().item()  # Force synchronization
    results["pin_memory_non_blocking"] = timer.stop() / repetitions
    
    # Méthode 4: Création directe sur GPU
    timer = Timer("direct_gpu")
    timer.start()
    for _ in range(repetitions):
        tensor_gpu = torch.randn(tensor_size, device='cuda')
        _ = tensor_gpu.sum().item()  # Force synchronization
    results["direct_gpu"] = timer.stop() / repetitions
    
    # Afficher les résultats
    print("\nRésultats du benchmark de transfert CPU→GPU:")
    print("-" * 50)
    for method, time_per_transfer in sorted(results.items(), key=lambda x: x[1]):
        print(f"{method:25}: {time_per_transfer*1000:.3f} ms par transfert")
    
    return results


def profile_training_step(model: torch.nn.Module, 
                          criterion: torch.nn.Module, 
                          optimizer: torch.optim.Optimizer, 
                          data_loader: torch.utils.data.DataLoader, 
                          n_steps: int = 5, 
                          warmup: int = 2, 
                          trace_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Effectue un profiling détaillé d'étapes d'entraînement.
    
    Args:
        model: Modèle PyTorch.
        criterion: Fonction de perte.
        optimizer: Optimiseur.
        data_loader: DataLoader pour les données d'entraînement.
        n_steps: Nombre d'étapes à profiler.
        warmup: Nombre d'étapes de préchauffage.
        trace_path: Chemin pour sauvegarder la trace (optionnel).
        
    Returns:
        dict: Résultats du profiling.
    """
    if not torch.cuda.is_available():
        print("CUDA n'est pas disponible, profiling ignoré.")
        return {}
    
    # Importer le profiler PyTorch
    from torch.profiler import profile, record_function, ProfilerActivity
    
    # Définir les activités à profiler
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    # Préchauffage
    model.train()
    device = next(model.parameters()).device
    
    # Obtenir un batch pour le préchauffage
    data_iter = iter(data_loader)
    for _ in range(warmup):
        try:
            batch = next(data_iter)
            if isinstance(batch, (list, tuple)):
                inputs = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            else:
                inputs = batch.to(device)
            
            # Forward et backward pass
            optimizer.zero_grad()
            if isinstance(inputs, (list, tuple)):
                outputs = model(*inputs)
                loss = criterion(outputs, inputs[1])  # Supposer que la cible est le deuxième élément
            else:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        except StopIteration:
            data_iter = iter(data_loader)
            continue
    
    # Profiling
    results = {}
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        data_iter = iter(data_loader)
        for step in range(n_steps):
            with record_function(f"training_step_{step}"):
                try:
                    batch = next(data_iter)
                    if isinstance(batch, (list, tuple)):
                        inputs = [b.to(device) if torch.is_tensor(b) else b for b in batch]
                    else:
                        inputs = batch.to(device)
                    
                    # Forward pass
                    with record_function("forward"):
                        optimizer.zero_grad()
                        if isinstance(inputs, (list, tuple)):
                            outputs = model(*inputs)
                            loss = criterion(outputs, inputs[1])
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, inputs)
                    
                    # Backward pass
                    with record_function("backward"):
                        loss.backward()
                    
                    # Optimizer step
                    with record_function("optimizer"):
                        optimizer.step()
                except StopIteration:
                    data_iter = iter(data_loader)
                    continue
    
    # Analyser les résultats
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    # Sauvegarder la trace si un chemin est spécifié
    if trace_path:
        prof.export_chrome_trace(trace_path)
        print(f"Trace sauvegardée dans {trace_path}")
    
    # Extraire les statistiques clés
    results["total_cpu_time"] = prof.total_average().cpu_time
    if torch.cuda.is_available():
        results["total_cuda_time"] = prof.total_average().cuda_time
    
    # Extraire les temps par opération
    op_times = {}
    for item in prof.key_averages():
        op_times[item.key] = {
            "cpu_time": item.cpu_time,
            "cuda_time": item.cuda_time if torch.cuda.is_available() else 0,
            "self_cpu_time": item.self_cpu_time,
            "self_cuda_time": item.self_cuda_time if torch.cuda.is_available() else 0,
        }
    results["op_times"] = op_times
    
    return results


def optimize_dataloader_params(sample_dataset: torch.utils.data.Dataset, 
                              batch_size: int, 
                              max_workers: int = 16) -> Dict[str, int]:
    """
    Teste différentes configurations de workers et prefetch_factor pour trouver l'optimum.
    
    Args:
        sample_dataset: Dataset à tester.
        batch_size: Taille du batch.
        max_workers: Nombre maximum de workers à tester.
        
    Returns:
        dict: Configuration optimale (num_workers, prefetch_factor).
    """
    import multiprocessing
    
    # Déterminer le nombre maximum de workers basé sur les CPU disponibles
    cpu_count = multiprocessing.cpu_count()
    max_workers = min(max_workers, cpu_count * 2)
    
    # Configurations à tester
    worker_options = [0, 1, 2, 4, 8, 12, 16]
    worker_options = [w for w in worker_options if w <= max_workers]
    
    prefetch_options = [2, 4, 8, 16, 32]
    
    results = {}
    best_time = float('inf')
    best_config = {"num_workers": 0, "prefetch_factor": 2}
    
    print(f"Optimisation des paramètres du DataLoader (max_workers={max_workers})...")
    print("-" * 60)
    print(f"{'Workers':^10} | {'Prefetch':^10} | {'Temps (s)':^15} | {'Mémoire (MB)':^15}")
    print("-" * 60)
    
    for num_workers in worker_options:
        for prefetch_factor in prefetch_options:
            # Ignorer les combinaisons invalides
            if num_workers == 0 and prefetch_factor > 2:
                continue
                
            # Créer un DataLoader avec cette configuration
            dataloader = torch.utils.data.DataLoader(
                sample_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor if num_workers > 0 else 2,
                pin_memory=True
            )
            
            # Mesurer le temps pour parcourir le dataset
            start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            timer = Timer()
            timer.start()
            
            # Parcourir le dataset
            for _ in dataloader:
                pass
                
            elapsed = timer.stop()
            end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            mem_used = (end_mem - start_mem) / (1024 * 1024)  # En MB
            
            # Enregistrer les résultats
            config_key = f"w{num_workers}_p{prefetch_factor}"
            results[config_key] = {"time": elapsed, "memory": mem_used}
            
            print(f"{num_workers:^10} | {prefetch_factor:^10} | {elapsed:^15.4f} | {mem_used:^15.2f}")
            
            # Mettre à jour la meilleure configuration
            if elapsed < best_time:
                best_time = elapsed
                best_config = {"num_workers": num_workers, "prefetch_factor": prefetch_factor}
    
    print("-" * 60)
    print(f"Meilleure configuration: {best_config['num_workers']} workers, "
          f"prefetch_factor={best_config['prefetch_factor']}")
    print(f"Temps: {best_time:.4f}s")
    
    return best_config


def benchmark_model_architectures(model_factories: Dict[str, Callable], 
                                 sample_input: torch.Tensor,
                                 batch_sizes: List[int] = [1, 4, 16, 32, 64],
                                 repetitions: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Compare les performances d'inférence de différentes architectures de modèles.
    
    Args:
        model_factories: Dictionnaire de fonctions factory pour créer les modèles.
        sample_input: Tenseur d'entrée d'exemple.
        batch_sizes: Liste des tailles de batch à tester.
        repetitions: Nombre de répétitions pour chaque test.
        
    Returns:
        dict: Résultats du benchmark pour chaque modèle et taille de batch.
    """
    if not torch.cuda.is_available():
        print("CUDA n'est pas disponible, benchmark exécuté sur CPU.")
        device = 'cpu'
    else:
        device = 'cuda'
    
    results = {}
    
    print(f"Benchmark des architectures de modèles sur {device}...")
    print("-" * 80)
    print(f"{'Architecture':^20} | {'Batch Size':^10} | {'Latence (ms)':^15} | {'Mémoire (MB)':^15}")
    print("-" * 80)
    
    for model_name, model_factory in model_factories.items():
        results[model_name] = {}
        
        # Créer le modèle
        model = model_factory().to(device)
        model.eval()
        
        for batch_size in batch_sizes:
            # Créer un input de la bonne taille
            if batch_size == 1:
                input_tensor = sample_input.unsqueeze(0).to(device)
            else:
                # Répliquer l'input pour atteindre la taille de batch souhaitée
                input_tensor = sample_input.repeat(batch_size, 1, 1, 1).to(device)
            
            # Préchauffage
            with torch.no_grad():
                for _ in range(10):
                    _ = model(input_tensor)
            
            # Mesurer le temps d'inférence
            torch.cuda.synchronize() if device == 'cuda' else None
            start_mem = torch.cuda.memory_allocated() if device == 'cuda' else 0
            
            timer = Timer()
            timer.start()
            
            with torch.no_grad():
                for _ in range(repetitions):
                    _ = model(input_tensor)
                    torch.cuda.synchronize() if device == 'cuda' else None
            
            elapsed = timer.stop()
            end_mem = torch.cuda.memory_allocated() if device == 'cuda' else 0
            mem_used = (end_mem - start_mem) / (1024 * 1024)  # En MB
            
            # Calculer la latence moyenne par batch
            latency_ms = (elapsed / repetitions) * 1000
            
            # Enregistrer les résultats
            results[model_name][batch_size] = {
                "latency_ms": latency_ms,
                "memory_mb": mem_used
            }
            
            print(f"{model_name:^20} | {batch_size:^10} | {latency_ms:^15.2f} | {mem_used:^15.2f}")
    
    print("-" * 80)
    return results
