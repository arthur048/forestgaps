"""
Module pour le calcul et stockage des statistiques de normalisation.

Ce module fournit des fonctionnalités pour calculer et stocker des statistiques
de normalisation pour les données de détection de trouées forestières.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import rasterio
from tqdm import tqdm

from data.preprocessing.conversion import convert_to_numpy

# Configuration du logger
logger = logging.getLogger(__name__)

class NormalizationStatistics:
    """
    Classe pour le calcul et la gestion des statistiques de normalisation.
    
    Cette classe permet de calculer, stocker et récupérer des statistiques
    de normalisation (min, max, moyenne, écart-type, etc.) pour un ensemble
    de données raster.
    """
    
    def __init__(
        self,
        stats: Optional[Dict[str, Any]] = None,
        method: str = "minmax",
        percentile_range: Tuple[float, float] = (1.0, 99.0)
    ):
        """
        Initialise l'objet NormalizationStatistics.
        
        Args:
            stats: Dictionnaire de statistiques préexistant (optionnel)
            method: Méthode de normalisation ('minmax', 'zscore', 'robust', 'percentile')
            percentile_range: Plage de percentiles à utiliser pour la méthode 'percentile'
        """
        self.stats = stats or {}
        self.method = method
        self.percentile_range = percentile_range
        
        # Initialise les champs de statistiques nécessaires
        if not self.stats:
            self.stats = {
                'global': {
                    'min': None,
                    'max': None,
                    'mean': None,
                    'std': None,
                    'median': None,
                    'p1': None,
                    'p99': None,
                    'hist_bins': None,
                    'hist_values': None,
                    'sample_count': 0,
                    'pixel_count': 0,
                },
                'method': method,
                'percentile_range': percentile_range,
                'per_file': {},
                'version': '1.0'
            }
    
    def compute_from_paths(
        self,
        file_paths: List[str],
        max_files: Optional[int] = None,
        sample_ratio: float = 1.0,
        compute_per_file: bool = True,
        compute_histogram: bool = True,
        hist_bins: int = 100,
        save_path: Optional[str] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Calcule les statistiques de normalisation à partir d'une liste de chemins de fichiers.
        
        Args:
            file_paths: Liste des chemins vers les fichiers raster
            max_files: Nombre maximum de fichiers à traiter (None = tous)
            sample_ratio: Ratio d'échantillonnage des pixels (1.0 = tous)
            compute_per_file: Calcule les statistiques pour chaque fichier individuellement
            compute_histogram: Calcule l'histogramme des valeurs
            hist_bins: Nombre de bins pour l'histogramme
            save_path: Chemin pour sauvegarder les statistiques calculées (optionnel)
            force_recompute: Force le recalcul même si les statistiques existent déjà
            
        Returns:
            Dictionnaire contenant les statistiques calculées
        """
        # Si les statistiques existent déjà et force_recompute est False, on retourne
        if self.stats['global']['mean'] is not None and not force_recompute:
            logger.info("Les statistiques existent déjà. Utilisez force_recompute=True pour recalculer.")
            return self.stats
        
        # Limite le nombre de fichiers si spécifié
        if max_files and max_files < len(file_paths):
            file_paths = file_paths[:max_files]
        
        logger.info(f"Calcul des statistiques sur {len(file_paths)} fichiers (échantillonnage: {sample_ratio*100:.1f}%)")
        
        all_values = []
        file_stats = {}
        
        for file_path in tqdm(file_paths, desc="Calcul des statistiques"):
            try:
                # Charge les données avec convert_to_numpy
                data, _ = convert_to_numpy(file_path, normalize=False)
                
                # Échantillonne les données si nécessaire
                if sample_ratio < 1.0:
                    mask = np.random.choice([True, False], size=data.shape, p=[sample_ratio, 1-sample_ratio])
                    sampled_data = data[mask]
                else:
                    sampled_data = data.flatten()
                
                # Filtre les valeurs invalides (NaN, Inf)
                valid_mask = np.isfinite(sampled_data)
                valid_data = sampled_data[valid_mask]
                
                # Ajoute les valeurs au tableau global
                all_values.append(valid_data)
                
                # Calcule les statistiques par fichier si nécessaire
                if compute_per_file:
                    file_stats[os.path.basename(file_path)] = self._compute_stats_from_array(
                        valid_data, compute_histogram, hist_bins
                    )
                    
                    # Mise à jour du compteur global
                    self.stats['global']['sample_count'] += 1
                    self.stats['global']['pixel_count'] += len(valid_data)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement du fichier {file_path}: {str(e)}")
        
        # Concatène toutes les valeurs pour calculer les statistiques globales
        if all_values:
            all_values_array = np.concatenate(all_values)
            self.stats['global'] = self._compute_stats_from_array(all_values_array, compute_histogram, hist_bins)
            self.stats['global']['sample_count'] = len(file_paths)
            self.stats['global']['pixel_count'] = len(all_values_array)
            
        # Stocke les statistiques par fichier
        if compute_per_file:
            self.stats['per_file'] = file_stats
        
        # Sauvegarde les statistiques si un chemin est spécifié
        if save_path:
            self.save(save_path)
        
        return self.stats
    
    def compute_from_dataset(
        self,
        dataset,
        max_samples: Optional[int] = None,
        compute_histogram: bool = True,
        hist_bins: int = 100,
        save_path: Optional[str] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Calcule les statistiques de normalisation à partir d'un dataset PyTorch.
        
        Args:
            dataset: Dataset PyTorch (doit implémenter __getitem__ retournant (input, _))
            max_samples: Nombre maximum d'échantillons à traiter (None = tous)
            compute_histogram: Calcule l'histogramme des valeurs
            hist_bins: Nombre de bins pour l'histogramme
            save_path: Chemin pour sauvegarder les statistiques calculées (optionnel)
            force_recompute: Force le recalcul même si les statistiques existent déjà
            
        Returns:
            Dictionnaire contenant les statistiques calculées
        """
        # Si les statistiques existent déjà et force_recompute est False, on retourne
        if self.stats['global']['mean'] is not None and not force_recompute:
            logger.info("Les statistiques existent déjà. Utilisez force_recompute=True pour recalculer.")
            return self.stats
        
        # Détermine le nombre d'échantillons à traiter
        num_samples = len(dataset)
        if max_samples and max_samples < num_samples:
            indices = np.random.choice(range(num_samples), max_samples, replace=False)
        else:
            indices = range(num_samples)
        
        logger.info(f"Calcul des statistiques sur {len(indices)} échantillons du dataset")
        
        all_values = []
        
        for idx in tqdm(indices, desc="Calcul des statistiques"):
            try:
                # Récupère les données d'entrée (pas les masques)
                data, _ = dataset[idx]
                
                # Convertit en NumPy si nécessaire
                if isinstance(data, torch.Tensor):
                    data = data.cpu().numpy()
                
                # Filtre les valeurs invalides (NaN, Inf)
                data_flat = data.flatten()
                valid_mask = np.isfinite(data_flat)
                valid_data = data_flat[valid_mask]
                
                # Ajoute les valeurs au tableau global
                all_values.append(valid_data)
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de l'échantillon {idx}: {str(e)}")
        
        # Concatène toutes les valeurs pour calculer les statistiques globales
        if all_values:
            all_values_array = np.concatenate(all_values)
            self.stats['global'] = self._compute_stats_from_array(all_values_array, compute_histogram, hist_bins)
            self.stats['global']['sample_count'] = len(indices)
            self.stats['global']['pixel_count'] = len(all_values_array)
        
        # Sauvegarde les statistiques si un chemin est spécifié
        if save_path:
            self.save(save_path)
        
        return self.stats
    
    def _compute_stats_from_array(
        self,
        data: np.ndarray,
        compute_histogram: bool = True,
        hist_bins: int = 100
    ) -> Dict[str, Any]:
        """
        Calcule les statistiques à partir d'un tableau NumPy.
        
        Args:
            data: Tableau NumPy contenant les données
            compute_histogram: Calcule l'histogramme des valeurs
            hist_bins: Nombre de bins pour l'histogramme
            
        Returns:
            Dictionnaire contenant les statistiques calculées
        """
        stats = {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'p1': float(np.percentile(data, 1)),
            'p99': float(np.percentile(data, 99)),
            'pixel_count': len(data)
        }
        
        # Calcule l'histogramme si demandé
        if compute_histogram:
            hist_values, hist_edges = np.histogram(data, bins=hist_bins)
            stats['hist_bins'] = hist_edges.tolist()
            stats['hist_values'] = hist_values.tolist()
        
        return stats
    
    def save(self, file_path: str) -> None:
        """
        Sauvegarde les statistiques dans un fichier JSON.
        
        Args:
            file_path: Chemin où sauvegarder le fichier JSON
        """
        try:
            # Crée le répertoire si nécessaire
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Sauvegarde les statistiques au format JSON
            with open(file_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
                
            logger.info(f"Statistiques sauvegardées dans {file_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des statistiques: {str(e)}")
    
    @classmethod
    def load(cls, file_path: str) -> 'NormalizationStatistics':
        """
        Charge des statistiques à partir d'un fichier JSON.
        
        Args:
            file_path: Chemin du fichier JSON contenant les statistiques
            
        Returns:
            Instance de NormalizationStatistics avec les statistiques chargées
        """
        try:
            with open(file_path, 'r') as f:
                stats = json.load(f)
            
            # Récupère la méthode et le percentile_range des statistiques chargées
            method = stats.get('method', 'minmax')
            percentile_range = tuple(stats.get('percentile_range', (1.0, 99.0)))
            
            logger.info(f"Statistiques chargées depuis {file_path}")
            
            return cls(stats=stats, method=method, percentile_range=percentile_range)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des statistiques: {str(e)}")
            raise
    
    def get_normalization_params(self, method: Optional[str] = None) -> Dict[str, float]:
        """
        Retourne les paramètres nécessaires pour la normalisation selon la méthode spécifiée.
        
        Args:
            method: Méthode de normalisation ('minmax', 'zscore', 'robust', 'percentile')
                   Si None, utilise la méthode par défaut de l'instance
                   
        Returns:
            Dictionnaire contenant les paramètres de normalisation
        """
        method = method or self.method
        global_stats = self.stats['global']
        
        if method == 'minmax':
            return {
                'min': global_stats['min'],
                'max': global_stats['max']
            }
        elif method == 'zscore':
            return {
                'mean': global_stats['mean'],
                'std': global_stats['std']
            }
        elif method == 'robust':
            return {
                'median': global_stats['median'],
                'p1': global_stats['p1'],
                'p99': global_stats['p99']
            }
        elif method == 'percentile':
            p_min, p_max = self.percentile_range
            return {
                f'p{p_min}': float(np.percentile(global_stats.get('p1', 0), p_min)),
                f'p{p_max}': float(np.percentile(global_stats.get('p99', 1), p_max))
            }
        else:
            raise ValueError(f"Méthode de normalisation '{method}' non supportée")


def compute_normalization_statistics(
    file_paths: List[str],
    output_path: str,
    method: str = "minmax",
    percentile_range: Tuple[float, float] = (1.0, 99.0),
    max_files: Optional[int] = None,
    sample_ratio: float = 1.0,
    compute_per_file: bool = True,
    compute_histogram: bool = True,
    hist_bins: int = 100,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Fonction utilitaire pour calculer et sauvegarder des statistiques de normalisation.
    
    Args:
        file_paths: Liste des chemins vers les fichiers raster
        output_path: Chemin où sauvegarder les statistiques calculées
        method: Méthode de normalisation ('minmax', 'zscore', 'robust', 'percentile')
        percentile_range: Plage de percentiles à utiliser pour la méthode 'percentile'
        max_files: Nombre maximum de fichiers à traiter (None = tous)
        sample_ratio: Ratio d'échantillonnage des pixels (1.0 = tous)
        compute_per_file: Calcule les statistiques pour chaque fichier individuellement
        compute_histogram: Calcule l'histogramme des valeurs
        hist_bins: Nombre de bins pour l'histogramme
        force_recompute: Force le recalcul même si les statistiques existent déjà
        
    Returns:
        Dictionnaire contenant les statistiques calculées
    """
    # Crée une instance de NormalizationStatistics
    norm_stats = NormalizationStatistics(method=method, percentile_range=percentile_range)
    
    # Calcule les statistiques
    stats = norm_stats.compute_from_paths(
        file_paths=file_paths,
        max_files=max_files,
        sample_ratio=sample_ratio,
        compute_per_file=compute_per_file,
        compute_histogram=compute_histogram,
        hist_bins=hist_bins,
        save_path=output_path,
        force_recompute=force_recompute
    )
    
    return stats


def batch_compute_statistics(
    input_dirs: List[str],
    output_dir: str,
    file_pattern: str = "*.tif",
    method: str = "minmax",
    max_files_per_dir: Optional[int] = None,
    sample_ratio: float = 0.1,
    compute_per_file: bool = False
) -> Dict[str, str]:
    """
    Calcule des statistiques de normalisation pour plusieurs répertoires.
    
    Args:
        input_dirs: Liste des répertoires contenant les fichiers raster
        output_dir: Répertoire où sauvegarder les fichiers de statistiques
        file_pattern: Motif pour filtrer les fichiers (ex: "*.tif")
        method: Méthode de normalisation
        max_files_per_dir: Nombre maximum de fichiers à traiter par répertoire
        sample_ratio: Ratio d'échantillonnage des pixels
        compute_per_file: Calcule les statistiques pour chaque fichier individuellement
        
    Returns:
        Dictionnaire associant les noms des répertoires aux chemins des fichiers de statistiques
    """
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    result_paths = {}
    
    for input_dir in input_dirs:
        dir_name = os.path.basename(input_dir)
        
        # Recherche les fichiers correspondant au motif
        file_paths = glob.glob(os.path.join(input_dir, file_pattern))
        
        if not file_paths:
            logger.warning(f"Aucun fichier trouvé dans {input_dir} avec le motif {file_pattern}")
            continue
        
        # Chemin de sortie pour les statistiques
        output_path = os.path.join(output_dir, f"{dir_name}_stats.json")
        
        # Calcule les statistiques
        compute_normalization_statistics(
            file_paths=file_paths,
            output_path=output_path,
            method=method,
            max_files=max_files_per_dir,
            sample_ratio=sample_ratio,
            compute_per_file=compute_per_file
        )
        
        result_paths[dir_name] = output_path
    
    return result_paths 