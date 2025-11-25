"""
Module d'échantillonnage pour la détection des trouées forestières.

Ce module fournit des classes et fonctions d'échantillonnage pour équilibrer
les datasets de trouées forestières en fonction des ratios de trouées, des sites, etc.
"""

import logging
import json
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
import rasterio
from tqdm import tqdm

# Configuration du logger
logger = logging.getLogger(__name__)


def calculate_gap_ratios(
    mask_paths: List[str],
    threshold_values: Optional[List[float]] = None,
    sample_size: Optional[int] = None,
    output_json: Optional[str] = None
) -> Dict[float, Dict[str, float]]:
    """
    Calcule les ratios de trouées pour chaque masque et chaque seuil.

    Args:
        mask_paths: Liste des chemins vers les masques.
        threshold_values: Liste des valeurs de seuil utilisées.
        sample_size: Nombre de masques à échantillonner pour le calcul (pour de grands datasets).
        output_json: Chemin pour sauvegarder les ratios calculés au format JSON.

    Returns:
        Un dictionnaire avec les ratios de trouées pour chaque seuil.
    """
    threshold_values = threshold_values or [5.0]
    
    # Limite l'échantillon si nécessaire
    if sample_size and len(mask_paths) > sample_size:
        indices = np.random.choice(len(mask_paths), sample_size, replace=False)
        sampled_paths = [mask_paths[i] for i in indices]
        logger.info(f"Calcul des ratios de trouées sur un échantillon de {sample_size} masques")
    else:
        sampled_paths = mask_paths
        logger.info(f"Calcul des ratios de trouées sur l'ensemble des {len(mask_paths)} masques")
    
    # Calcul des ratios pour chaque seuil
    result = {}
    
    # Pour chaque seuil, on calcule différentes statistiques
    for threshold in threshold_values:
        stats = {
            'gap_ratios': [],
            'mean_ratio': 0.0,
            'median_ratio': 0.0,
            'min_ratio': 1.0,
            'max_ratio': 0.0,
            'std_ratio': 0.0,
            'num_zero_gaps': 0,
            'num_high_gaps': 0
        }
        
        # Calcul des ratios pour chaque masque
        for mask_path in tqdm(sampled_paths, desc=f"Analyse des masques (seuil {threshold}m)"):
            try:
                # Chargement du masque
                if mask_path.endswith(('.npy', '.npz')):
                    mask = np.load(mask_path)
                    if isinstance(mask, np.lib.npyio.NpzFile):
                        mask = mask['arr_0'] if 'arr_0' in mask.files else next(iter(mask.values()))
                else:
                    with rasterio.open(mask_path) as src:
                        mask = src.read(1)
                
                # Calcul du ratio de trouées (proportion de pixels > 0)
                total_pixels = mask.size
                gap_pixels = np.sum(mask > 0)
                
                if total_pixels > 0:
                    ratio = gap_pixels / total_pixels
                    stats['gap_ratios'].append(ratio)
                    
                    # Mise à jour des statistiques
                    stats['min_ratio'] = min(stats['min_ratio'], ratio)
                    stats['max_ratio'] = max(stats['max_ratio'], ratio)
                    
                    # Comptage des masques sans trouées ou avec beaucoup de trouées
                    if ratio == 0:
                        stats['num_zero_gaps'] += 1
                    if ratio > 0.5:
                        stats['num_high_gaps'] += 1
            except Exception as e:
                logger.warning(f"Erreur lors du calcul du ratio pour {mask_path}: {str(e)}")
        
        # Calcul des statistiques globales
        if stats['gap_ratios']:
            stats['mean_ratio'] = np.mean(stats['gap_ratios'])
            stats['median_ratio'] = np.median(stats['gap_ratios'])
            stats['std_ratio'] = np.std(stats['gap_ratios'])
        
        # Conversion en type Python standard pour JSON
        stats['gap_ratios'] = [float(r) for r in stats['gap_ratios']]
        
        # Ajout au résultat
        result[float(threshold)] = stats
        
        logger.info(f"Seuil {threshold}m: ratio moyen={stats['mean_ratio']:.4f}, médian={stats['median_ratio']:.4f}, "
                   f"min={stats['min_ratio']:.4f}, max={stats['max_ratio']:.4f}")
    
    # Sauvegarde des résultats si demandé
    if output_json:
        try:
            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Ratios de trouées sauvegardés dans {output_json}")
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde des ratios: {str(e)}")
    
    return result


def create_weighted_sampler(
    dataset: Dataset,
    mask_paths: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    gap_ratios: Optional[Dict[int, float]] = None,
    balance_strategy: str = 'inverse',
    replacement: bool = True
) -> WeightedRandomSampler:
    """
    Crée un échantillonneur pondéré pour équilibrer les classes dans un dataset.

    Args:
        dataset: Dataset à échantillonner.
        mask_paths: Liste des chemins vers les masques (si gap_ratios n'est pas fourni).
        weights: Poids prédéfinis pour chaque exemple (prioritaire si fourni).
        gap_ratios: Dictionnaire des ratios de trouées précalculés pour chaque exemple.
        balance_strategy: Stratégie d'équilibrage ('inverse', 'sqrt_inverse', 'equal').
        replacement: Si True, échantillonne avec remplacement.

    Returns:
        Un échantillonneur pondéré pour équilibrer les classes.
    """
    if weights is not None:
        logger.info("Utilisation des poids prédéfinis pour l'échantillonnage")
        if len(weights) != len(dataset):
            raise ValueError(f"Le nombre de poids ({len(weights)}) ne correspond pas au nombre d'exemples ({len(dataset)})")
        sample_weights = torch.tensor(weights, dtype=torch.float)
    
    elif gap_ratios is not None:
        logger.info("Utilisation des ratios de trouées précalculés pour l'échantillonnage")
        sample_weights = torch.ones(len(dataset), dtype=torch.float)
        
        for idx, ratio in gap_ratios.items():
            idx = int(idx)
            if idx < len(sample_weights):
                if balance_strategy == 'inverse':
                    # Plus le ratio est faible, plus le poids est élevé
                    sample_weights[idx] = 1.0 / (ratio + 0.01)
                elif balance_strategy == 'sqrt_inverse':
                    # Version plus douce de l'inverse
                    sample_weights[idx] = 1.0 / np.sqrt(ratio + 0.01)
                elif balance_strategy == 'equal':
                    # Suréchantillonnage des exemples avec peu de trouées
                    sample_weights[idx] = 1.0 if ratio > 0.1 else 10.0
    
    elif mask_paths is not None:
        logger.info("Calcul des poids basés sur les ratios de trouées")
        sample_weights = torch.ones(len(dataset), dtype=torch.float)
        
        for i, mask_path in enumerate(tqdm(mask_paths, desc="Calcul des poids d'échantillonnage")):
            try:
                # Chargement du masque
                if mask_path.endswith(('.npy', '.npz')):
                    mask = np.load(mask_path)
                    if isinstance(mask, np.lib.npyio.NpzFile):
                        mask = mask['arr_0'] if 'arr_0' in mask.files else next(iter(mask.values()))
                else:
                    with rasterio.open(mask_path) as src:
                        mask = src.read(1)
                
                # Calcul du ratio de trouées
                ratio = np.sum(mask > 0) / mask.size
                
                # Attribution du poids selon la stratégie
                if balance_strategy == 'inverse':
                    sample_weights[i] = 1.0 / (ratio + 0.01)
                elif balance_strategy == 'sqrt_inverse':
                    sample_weights[i] = 1.0 / np.sqrt(ratio + 0.01)
                elif balance_strategy == 'equal':
                    sample_weights[i] = 1.0 if ratio > 0.1 else 10.0
            
            except Exception as e:
                logger.warning(f"Erreur lors du calcul du poids pour {mask_path}: {str(e)}")
                sample_weights[i] = 1.0
    
    else:
        logger.warning("Aucune source de poids fournie. Utilisation de poids uniformes.")
        sample_weights = torch.ones(len(dataset), dtype=torch.float)
    
    # Normalisation des poids pour éviter les valeurs extrêmes
    if torch.max(sample_weights) > 100:
        sample_weights = 100 * sample_weights / torch.max(sample_weights)
    
    # Création de l'échantillonneur
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=replacement
    )
    
    logger.info(f"Échantillonneur créé avec {len(dataset)} exemples, "
               f"poids min={torch.min(sample_weights):.2f}, max={torch.max(sample_weights):.2f}")
    
    return sampler


class BalancedGapSampler(Sampler):
    """
    Sampler personnalisé pour équilibrer les exemples en fonction du ratio de trouées.
    
    Cette classe permet un contrôle plus fin que WeightedRandomSampler, avec des
    stratégies d'échantillonnage adaptées spécifiquement aux trouées forestières.
    
    Attributes:
        dataset: Dataset à échantillonner.
        mask_paths: Liste des chemins vers les masques.
        threshold_value: Valeur de seuil de hauteur.
        target_ratio: Ratio cible de trouées pour l'équilibrage.
        batch_size: Taille des batchs pour l'échantillonnage.
        indices: Indices des exemples dans le dataset.
        weights: Poids d'échantillonnage pour chaque exemple.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        mask_paths: List[str],
        threshold_value: float = 5.0,
        target_ratio: float = 0.5,
        batch_size: int = 8,
        balance_mode: str = 'batch',  # 'batch', 'global', 'alternate'
        precomputed_ratios: Optional[Dict[int, float]] = None
    ):
        """
        Initialise le sampler.
        
        Args:
            dataset: Dataset à échantillonner.
            mask_paths: Liste des chemins vers les masques.
            threshold_value: Valeur de seuil de hauteur.
            target_ratio: Ratio cible de trouées pour l'équilibrage.
            batch_size: Taille des batchs pour l'échantillonnage.
            balance_mode: Mode d'équilibrage ('batch', 'global', 'alternate').
            precomputed_ratios: Ratios précomputés (optionnel, pour éviter le recalcul).
        """
        self.dataset = dataset
        self.mask_paths = mask_paths
        self.threshold_value = threshold_value
        self.target_ratio = target_ratio
        self.batch_size = batch_size
        self.balance_mode = balance_mode
        
        # Indices de tous les exemples
        self.indices = list(range(len(dataset)))
        
        # Calcul ou récupération des ratios de trouées
        self.gap_ratios = {}
        if precomputed_ratios:
            self.gap_ratios = precomputed_ratios
            logger.info(f"Utilisation de ratios précomputés pour {len(precomputed_ratios)} exemples")
        else:
            self._compute_gap_ratios()
        
        # Classification des exemples selon leur ratio de trouées
        self.high_gap_indices = []  # Exemples avec beaucoup de trouées
        self.low_gap_indices = []   # Exemples avec peu de trouées
        self.balanced_indices = []  # Exemples avec un ratio proche de la cible
        
        for idx, ratio in self.gap_ratios.items():
            idx = int(idx)
            if ratio > self.target_ratio + 0.1:
                self.high_gap_indices.append(idx)
            elif ratio < self.target_ratio - 0.1:
                self.low_gap_indices.append(idx)
            else:
                self.balanced_indices.append(idx)
        
        logger.info(f"Exemples avec beaucoup de trouées: {len(self.high_gap_indices)}")
        logger.info(f"Exemples avec peu de trouées: {len(self.low_gap_indices)}")
        logger.info(f"Exemples équilibrés: {len(self.balanced_indices)}")
        
        # Préparation des indices selon le mode d'équilibrage
        self._prepare_sampling()
    
    def _compute_gap_ratios(self):
        """Calcule les ratios de trouées pour chaque exemple."""
        logger.info("Calcul des ratios de trouées pour l'échantillonnage équilibré")
        
        for i, mask_path in enumerate(tqdm(self.mask_paths, desc="Analyse des masques")):
            try:
                # Chargement et analyse du masque
                if mask_path.endswith(('.npy', '.npz')):
                    mask = np.load(mask_path)
                    if isinstance(mask, np.lib.npyio.NpzFile):
                        mask = mask['arr_0'] if 'arr_0' in mask.files else next(iter(mask.values()))
                else:
                    with rasterio.open(mask_path) as src:
                        mask = src.read(1)
                
                # Calcul du ratio de trouées
                total_pixels = mask.size
                gap_pixels = np.sum(mask > 0)
                
                if total_pixels > 0:
                    self.gap_ratios[i] = gap_pixels / total_pixels
                else:
                    self.gap_ratios[i] = 0.0
                    
            except Exception as e:
                logger.warning(f"Erreur lors du calcul du ratio pour {mask_path}: {str(e)}")
                self.gap_ratios[i] = 0.0
    
    def _prepare_sampling(self):
        """Prépare la stratégie d'échantillonnage selon le mode choisi."""
        if self.balance_mode == 'batch':
            # Prépare des batchs équilibrés
            self._prepare_batch_balanced_sampling()
        elif self.balance_mode == 'global':
            # Prépare un échantillonnage globalement équilibré
            self._prepare_globally_balanced_sampling()
        elif self.balance_mode == 'alternate':
            # Alternance entre exemples à fort et faible ratio
            self._prepare_alternating_sampling()
        else:
            logger.warning(f"Mode d'équilibrage '{self.balance_mode}' non reconnu. Utilisation du mode 'batch'.")
            self._prepare_batch_balanced_sampling()
    
    def _prepare_batch_balanced_sampling(self):
        """Prépare des batchs équilibrés en termes de ratio de trouées."""
        # Mélange des indices
        np.random.shuffle(self.high_gap_indices)
        np.random.shuffle(self.low_gap_indices)
        np.random.shuffle(self.balanced_indices)
        
        # Calcul du nombre de batchs et d'exemples par catégorie
        self.num_samples = len(self.dataset)
        self.num_batches = self.num_samples // self.batch_size
        
        # Répartition des exemples dans les batchs
        self.sampling_order = []
        high_idx = 0
        low_idx = 0
        balanced_idx = 0
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Dans chaque batch, on mélange les trois catégories
            # avec une proportion adaptée au ratio cible
            high_in_batch = int(self.batch_size * (1 - self.target_ratio))
            low_in_batch = int(self.batch_size * self.target_ratio)
            balanced_in_batch = self.batch_size - high_in_batch - low_in_batch
            
            # Ajout des exemples à haut ratio
            for _ in range(high_in_batch):
                if high_idx < len(self.high_gap_indices):
                    batch_indices.append(self.high_gap_indices[high_idx])
                    high_idx += 1
                elif balanced_idx < len(self.balanced_indices):
                    batch_indices.append(self.balanced_indices[balanced_idx])
                    balanced_idx += 1
                elif low_idx < len(self.low_gap_indices):
                    batch_indices.append(self.low_gap_indices[low_idx])
                    low_idx += 1
                else:
                    # Si on a épuisé tous les exemples, on recommence
                    np.random.shuffle(self.high_gap_indices)
                    np.random.shuffle(self.low_gap_indices)
                    np.random.shuffle(self.balanced_indices)
                    high_idx = 0
                    low_idx = 0
                    balanced_idx = 0
                    batch_indices.append(self.high_gap_indices[high_idx])
                    high_idx += 1
            
            # Ajout des exemples à bas ratio
            for _ in range(low_in_batch):
                if low_idx < len(self.low_gap_indices):
                    batch_indices.append(self.low_gap_indices[low_idx])
                    low_idx += 1
                elif balanced_idx < len(self.balanced_indices):
                    batch_indices.append(self.balanced_indices[balanced_idx])
                    balanced_idx += 1
                elif high_idx < len(self.high_gap_indices):
                    batch_indices.append(self.high_gap_indices[high_idx])
                    high_idx += 1
                else:
                    np.random.shuffle(self.high_gap_indices)
                    np.random.shuffle(self.low_gap_indices)
                    np.random.shuffle(self.balanced_indices)
                    high_idx = 0
                    low_idx = 0
                    balanced_idx = 0
                    batch_indices.append(self.low_gap_indices[low_idx])
                    low_idx += 1
            
            # Ajout des exemples équilibrés
            for _ in range(balanced_in_batch):
                if balanced_idx < len(self.balanced_indices):
                    batch_indices.append(self.balanced_indices[balanced_idx])
                    balanced_idx += 1
                elif high_idx < len(self.high_gap_indices):
                    batch_indices.append(self.high_gap_indices[high_idx])
                    high_idx += 1
                elif low_idx < len(self.low_gap_indices):
                    batch_indices.append(self.low_gap_indices[low_idx])
                    low_idx += 1
                else:
                    np.random.shuffle(self.high_gap_indices)
                    np.random.shuffle(self.low_gap_indices)
                    np.random.shuffle(self.balanced_indices)
                    high_idx = 0
                    low_idx = 0
                    balanced_idx = 0
                    batch_indices.append(self.balanced_indices[balanced_idx])
                    balanced_idx += 1
            
            # Mélange des indices au sein du batch
            np.random.shuffle(batch_indices)
            self.sampling_order.extend(batch_indices)
        
        # Ajout des exemples restants
        remaining = self.num_samples - len(self.sampling_order)
        if remaining > 0:
            remaining_indices = np.random.choice(self.indices, remaining, replace=False)
            self.sampling_order.extend(remaining_indices)
    
    def _prepare_globally_balanced_sampling(self):
        """Prépare un échantillonnage globalement équilibré."""
        # Calcul du nombre d'exemples par catégorie pour atteindre le ratio cible
        self.num_samples = len(self.dataset)
        target_high = int(self.num_samples * (1 - self.target_ratio))
        target_low = int(self.num_samples * self.target_ratio)
        target_balanced = self.num_samples - target_high - target_low
        
        # Sélection des exemples
        high_indices = self.high_gap_indices.copy()
        low_indices = self.low_gap_indices.copy()
        balanced_indices = self.balanced_indices.copy()
        
        np.random.shuffle(high_indices)
        np.random.shuffle(low_indices)
        np.random.shuffle(balanced_indices)
        
        # Ajustement des quantités
        if len(high_indices) > target_high:
            high_indices = high_indices[:target_high]
        else:
            # Si pas assez d'exemples à haut ratio, compléter avec des équilibrés
            deficit = target_high - len(high_indices)
            if len(balanced_indices) > target_balanced + deficit:
                high_indices.extend(balanced_indices[target_balanced:target_balanced+deficit])
                balanced_indices = balanced_indices[:target_balanced]
            else:
                # Si toujours pas assez, compléter avec des bas ratio
                high_indices.extend(balanced_indices)
                deficit = target_high - len(high_indices)
                if deficit > 0 and len(low_indices) > target_low + deficit:
                    high_indices.extend(low_indices[target_low:target_low+deficit])
                    low_indices = low_indices[:target_low]
        
        if len(low_indices) > target_low:
            low_indices = low_indices[:target_low]
        else:
            # Si pas assez d'exemples à bas ratio, compléter avec des équilibrés
            deficit = target_low - len(low_indices)
            if len(balanced_indices) > target_balanced + deficit:
                low_indices.extend(balanced_indices[target_balanced:target_balanced+deficit])
                balanced_indices = balanced_indices[:target_balanced]
            else:
                # Si toujours pas assez, compléter avec des haut ratio
                low_indices.extend(balanced_indices)
                deficit = target_low - len(low_indices)
                if deficit > 0 and len(high_indices) > target_high + deficit:
                    low_indices.extend(high_indices[target_high:target_high+deficit])
                    high_indices = high_indices[:target_high]
        
        # Mélange et combinaison de tous les indices
        all_indices = high_indices + low_indices + balanced_indices
        np.random.shuffle(all_indices)
        
        # Vérification du nombre d'exemples
        if len(all_indices) < self.num_samples:
            # Compléter si nécessaire
            missing = self.num_samples - len(all_indices)
            other_indices = [i for i in self.indices if i not in all_indices]
            if other_indices:
                additional = np.random.choice(other_indices, min(missing, len(other_indices)), replace=False)
                all_indices.extend(additional)
        
        self.sampling_order = all_indices
    
    def _prepare_alternating_sampling(self):
        """Prépare un échantillonnage alternant entre exemples à fort et faible ratio."""
        # Mélange des indices
        high_indices = self.high_gap_indices.copy()
        low_indices = self.low_gap_indices.copy()
        balanced_indices = self.balanced_indices.copy()
        
        np.random.shuffle(high_indices)
        np.random.shuffle(low_indices)
        np.random.shuffle(balanced_indices)
        
        # Alternance entre les catégories
        self.sampling_order = []
        high_idx = 0
        low_idx = 0
        balanced_idx = 0
        
        while len(self.sampling_order) < len(self.dataset):
            # Ajout d'un exemple à haut ratio
            if high_idx < len(high_indices):
                self.sampling_order.append(high_indices[high_idx])
                high_idx += 1
            elif balanced_idx < len(balanced_indices):
                self.sampling_order.append(balanced_indices[balanced_idx])
                balanced_idx += 1
            elif low_idx < len(low_indices):
                self.sampling_order.append(low_indices[low_idx])
                low_idx += 1
            else:
                break
            
            # Ajout d'un exemple à bas ratio
            if low_idx < len(low_indices):
                self.sampling_order.append(low_indices[low_idx])
                low_idx += 1
            elif balanced_idx < len(balanced_indices):
                self.sampling_order.append(balanced_indices[balanced_idx])
                balanced_idx += 1
            elif high_idx < len(high_indices):
                self.sampling_order.append(high_indices[high_idx])
                high_idx += 1
            else:
                break
            
            # Ajout d'un exemple équilibré
            if balanced_idx < len(balanced_indices):
                self.sampling_order.append(balanced_indices[balanced_idx])
                balanced_idx += 1
            elif high_idx < len(high_indices):
                self.sampling_order.append(high_indices[high_idx])
                high_idx += 1
            elif low_idx < len(low_indices):
                self.sampling_order.append(low_indices[low_idx])
                low_idx += 1
            else:
                break
        
        # Si on n'a pas assez d'exemples, compléter avec des indices aléatoires
        if len(self.sampling_order) < len(self.dataset):
            missing = len(self.dataset) - len(self.sampling_order)
            other_indices = [i for i in self.indices if i not in self.sampling_order]
            if other_indices:
                additional = np.random.choice(other_indices, min(missing, len(other_indices)), replace=False)
                self.sampling_order.extend(additional)
    
    def __iter__(self):
        """Retourne un itérateur sur les indices d'échantillonnage."""
        return iter(self.sampling_order)
    
    def __len__(self):
        """Retourne le nombre d'exemples à échantillonner."""
        return len(self.dataset)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Chemins fictifs pour l'exemple
    mask_paths = [f"data/mask_{i}.tif" for i in range(10)]
    
    # Création d'un dataset fictif
    class DummyDataset(Dataset):
        def __init__(self, size=10):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return idx
    
    dataset = DummyDataset()
    
    # Calcul des ratios de trouées (simulé)
    gap_ratios = {
        0: 0.05, 1: 0.15, 2: 0.25, 3: 0.35, 4: 0.45,
        5: 0.55, 6: 0.65, 7: 0.75, 8: 0.85, 9: 0.95
    }
    
    # Création d'un échantillonneur pondéré
    sampler = create_weighted_sampler(
        dataset=dataset,
        gap_ratios=gap_ratios,
        balance_strategy='inverse'
    )
    
    print("Échantillonneur pondéré créé")
    
    # Création d'un échantillonneur équilibré
    balanced_sampler = BalancedGapSampler(
        dataset=dataset,
        mask_paths=mask_paths,
        target_ratio=0.5,
        batch_size=4,
        balance_mode='batch',
        precomputed_ratios=gap_ratios
    )
    
    print("Échantillonneur équilibré créé")
    print(f"Ordre d'échantillonnage: {list(balanced_sampler)}")
