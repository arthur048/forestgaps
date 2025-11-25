"""
Module de gestion des datasets pour la détection des trouées forestières.

Ce module fournit la classe principale ForestGapDataset pour charger et gérer
les paires d'images (DSM/CHM) et masques pour l'entraînement de modèles de segmentation.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import rasterio

from forestgaps.data.preprocessing.conversion import convert_to_numpy

# Configuration du logger
logger = logging.getLogger(__name__)


class ForestGapDataset(Dataset):
    """
    Dataset pour la détection de trouées forestières.
    
    Cette classe charge des paires d'images (DSM/CHM) et de masques pour l'entraînement
    de modèles de segmentation. Elle prend en charge la normalisation, la mise en cache,
    et les transformations d'augmentation de données.
    
    Attributes:
        input_paths (List[str]): Liste des chemins vers les images d'entrée (DSM/CHM).
        mask_paths (List[str]): Liste des chemins vers les masques de trouées.
        transform (Callable): Fonction de transformation pour l'augmentation de données.
        threshold_value (float): Valeur de seuil pour la création de masques binaires.
        cache_data (bool): Si True, met en cache les données en mémoire après le chargement.
        normalize (bool): Si True, normalise les données.
        calculate_norm_stats (bool): Si True, calcule les statistiques de normalisation.
        norm_stats (Dict): Statistiques de normalisation.
        metadata (Dict): Métadonnées du dataset.
        cache (Dict): Cache des données.
    """
    
    def __init__(
        self,
        input_paths: List[str],
        mask_paths: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        threshold_value: float = 2.0,
        below_threshold: bool = True,
        cache_data: bool = False,
        normalize: bool = True,
        calculate_norm_stats: bool = True,
        norm_stats: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise le ForestGapDataset.
        
        Args:
            input_paths: Liste des chemins vers les images d'entrée (DSM/CHM).
            mask_paths: Liste des chemins vers les masques de trouées. Si None, mode inference.
            transform: Fonction de transformation pour l'augmentation de données.
            threshold_value: Valeur de seuil pour la création de masques binaires.
            below_threshold: Si True, les pixels en dessous du seuil sont considérés comme des trouées.
            cache_data: Si True, met en cache les données en mémoire après le chargement.
            normalize: Si True, normalise les données.
            calculate_norm_stats: Si True, calcule les statistiques de normalisation.
            norm_stats: Statistiques de normalisation préexistantes.
            metadata: Métadonnées du dataset.
        """
        self.input_paths = input_paths
        self.mask_paths = mask_paths
        
        # Vérification de la longueur des listes
        if mask_paths is not None and len(input_paths) != len(mask_paths):
            raise ValueError(f"Le nombre d'images d'entrée ({len(input_paths)}) doit correspondre "
                            f"au nombre de masques ({len(mask_paths)})")
        
        self.transform = transform
        self.threshold_value = threshold_value
        self.below_threshold = below_threshold
        self.cache_data = cache_data
        self.normalize = normalize
        self.calculate_norm_stats = calculate_norm_stats
        
        # Initialiser les statistiques de normalisation
        if norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            self.norm_stats = {"mean": None, "std": None, "min": None, "max": None}
            
        # Initialiser les métadonnées
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = {
                "num_samples": len(input_paths),
                "has_masks": mask_paths is not None,
                "normalization": normalize,
                "threshold_value": threshold_value,
                "below_threshold": below_threshold,
                "input_paths": input_paths,
                "mask_paths": mask_paths if mask_paths is not None else []
            }
        
        # Cache pour les données
        self.cache = {}
        
        # Calculer les statistiques de normalisation si nécessaire
        if normalize and calculate_norm_stats and norm_stats is None:
            logger.info("Calcul des statistiques de normalisation...")
            self._calculate_normalization_stats()
    
    def __len__(self) -> int:
        """
        Retourne le nombre d'échantillons dans le dataset.
        
        Returns:
            Nombre d'échantillons.
        """
        return len(self.input_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Récupère un échantillon du dataset.
        
        Args:
            idx: Index de l'échantillon à récupérer.
            
        Returns:
            Tuple contenant l'image d'entrée et le masque (ou tensor vide si pas de masque).
        """
        # Vérifier si les données sont en cache
        if self.cache_data and idx in self.cache:
            image, mask = self.cache[idx]
            
        else:
            # Charger l'image d'entrée
            input_path = self.input_paths[idx]
            try:
                image, _ = self._load_input(input_path)
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'image {input_path}: {str(e)}")
                # Retourner une image et un masque vides en cas d'erreur
                return torch.zeros((1, 64, 64), dtype=torch.float32), torch.zeros((1, 64, 64), dtype=torch.float32)
            
            # Charger le masque si disponible
            if self.mask_paths is not None:
                mask_path = self.mask_paths[idx]
                try:
                    mask, _ = self._load_mask(mask_path)
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du masque {mask_path}: {str(e)}")
                    # Retourner un masque vide en cas d'erreur
                    mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
            else:
                # Masque vide pour l'inférence
                mask = torch.zeros((1, image.shape[1], image.shape[2]), dtype=torch.float32)
            
            # Mettre en cache si nécessaire
            if self.cache_data:
                self.cache[idx] = (image, mask)
        
        # Appliquer les transformations si spécifiées
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        return image, mask
    
    def _load_input(self, path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Charge une image d'entrée et la convertit en tensor.
        
        Args:
            path: Chemin vers l'image à charger.
            
        Returns:
            Tuple contenant le tensor de l'image et les métadonnées.
        """
        # Charger l'image avec rasterio
        with rasterio.open(path) as src:
            # Lire toutes les bandes
            image_np = src.read()
            metadata = src.meta.copy()
        
        # Normaliser si nécessaire
        if self.normalize:
            image_np = self._normalize_data(image_np)
        
        # Convertir en tensor
        image_tensor = torch.from_numpy(image_np.astype(np.float32))
        
        return image_tensor, metadata
    
    def _load_mask(self, path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Charge un masque et le convertit en tensor binaire.
        
        Args:
            path: Chemin vers le masque à charger.
            
        Returns:
            Tuple contenant le tensor du masque et les métadonnées.
        """
        # Charger le masque avec rasterio
        with rasterio.open(path) as src:
            # Lire toutes les bandes
            mask_np = src.read()
            metadata = src.meta.copy()
        
        # Créer un masque binaire basé sur le seuil
        if self.below_threshold:
            binary_mask = (mask_np < self.threshold_value).astype(np.float32)
        else:
            binary_mask = (mask_np >= self.threshold_value).astype(np.float32)
        
        # Convertir en tensor
        mask_tensor = torch.from_numpy(binary_mask)
        
        return mask_tensor, metadata
    
    def _calculate_normalization_stats(self) -> None:
        """
        Calcule les statistiques de normalisation sur un sous-ensemble d'images.
        """
        # Limiter le calcul à un maximum de 50 images pour des raisons de performance
        sample_size = min(len(self.input_paths), 50)
        indices = np.random.choice(len(self.input_paths), sample_size, replace=False)
        
        # Collecter les valeurs pour le calcul des statistiques
        all_values = []
        
        logger.info(f"Calcul des statistiques sur {sample_size} images...")
        
        for i, idx in enumerate(indices):
            input_path = self.input_paths[idx]
            try:
                # Charger l'image sans normalisation
                with rasterio.open(input_path) as src:
                    image_np = src.read()
                
                # Ignorer les valeurs NoData
                mask = image_np != src.nodata if src.nodata is not None else np.ones_like(image_np, dtype=bool)
                values = image_np[mask]
                
                all_values.append(values)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Traitement {i + 1}/{sample_size} images...")
                
            except Exception as e:
                logger.error(f"Erreur lors du calcul des statistiques pour {input_path}: {str(e)}")
        
        # Concaténer toutes les valeurs
        if all_values:
            all_values = np.concatenate(all_values)
            
            # Calculer les statistiques
            self.norm_stats["mean"] = float(np.mean(all_values))
            self.norm_stats["std"] = float(np.std(all_values))
            self.norm_stats["min"] = float(np.min(all_values))
            self.norm_stats["max"] = float(np.max(all_values))
            
            logger.info(f"Statistiques calculées: mean={self.norm_stats['mean']:.2f}, "
                       f"std={self.norm_stats['std']:.2f}, "
                       f"min={self.norm_stats['min']:.2f}, "
                       f"max={self.norm_stats['max']:.2f}")
        else:
            logger.warning("Impossible de calculer les statistiques de normalisation. "
                          "Utilisation des valeurs par défaut.")
            self.norm_stats["mean"] = 0.0
            self.norm_stats["std"] = 1.0
            self.norm_stats["min"] = 0.0
            self.norm_stats["max"] = 1.0
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalise les données en fonction des statistiques calculées.
        
        Args:
            data: Tableau NumPy à normaliser.
            
        Returns:
            Tableau normalisé.
        """
        if self.norm_stats["mean"] is None or self.norm_stats["std"] is None:
            logger.warning("Statistiques de normalisation non disponibles. "
                          "Utilisation de la normalisation min-max.")
            if self.norm_stats["min"] is not None and self.norm_stats["max"] is not None:
                # Normalisation min-max
                min_val = self.norm_stats["min"]
                max_val = self.norm_stats["max"]
                if max_val > min_val:
                    return (data - min_val) / (max_val - min_val)
                else:
                    return data
            else:
                return data
        else:
            # Normalisation Z-score
            return (data - self.norm_stats["mean"]) / (self.norm_stats["std"] + 1e-8)
    
    def save_metadata(self, output_path: str) -> None:
        """
        Sauvegarde les métadonnées du dataset dans un fichier JSON.
        
        Args:
            output_path: Chemin où sauvegarder les métadonnées.
        """
        # Mettre à jour les métadonnées
        metadata = {
            **self.metadata,
            "normalization_stats": self.norm_stats
        }
        
        # Convertir les chemins en chaînes de caractères
        metadata["input_paths"] = [str(p) for p in metadata["input_paths"]]
        metadata["mask_paths"] = [str(p) for p in metadata["mask_paths"]]
        
        try:
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Métadonnées sauvegardées dans {output_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métadonnées: {str(e)}")
    
    @property
    def has_masks(self) -> bool:
        """
        Vérifie si le dataset contient des masques.
        
        Returns:
            True si des masques sont disponibles.
        """
        return self.mask_paths is not None and len(self.mask_paths) > 0


def create_gap_dataset(
    input_paths: List[str],
    mask_paths: Optional[List[str]] = None,
    transform: Optional[Callable] = None,
    threshold_value: float = 2.0,
    below_threshold: bool = True,
    cache_data: bool = False,
    normalize: bool = True,
    metadata_output: Optional[str] = None
) -> ForestGapDataset:
    """
    Crée un ForestGapDataset à partir de listes de chemins.
    
    Args:
        input_paths: Liste des chemins vers les images d'entrée.
        mask_paths: Liste des chemins vers les masques. Si None, mode inference.
        transform: Fonction de transformation pour l'augmentation de données.
        threshold_value: Valeur de seuil pour la création de masques binaires.
        below_threshold: Si True, les pixels en dessous du seuil sont considérés comme des trouées.
        cache_data: Si True, met en cache les données en mémoire.
        normalize: Si True, normalise les données.
        metadata_output: Chemin où sauvegarder les métadonnées. Si None, pas de sauvegarde.
        
    Returns:
        Instance de ForestGapDataset.
    """
    dataset = ForestGapDataset(
        input_paths=input_paths,
        mask_paths=mask_paths,
        transform=transform,
        threshold_value=threshold_value,
        below_threshold=below_threshold,
        cache_data=cache_data,
        normalize=normalize
    )
    
    if metadata_output is not None:
        dataset.save_metadata(metadata_output)
    
    return dataset


def load_dataset_from_metadata(
    metadata_path: str,
    transform: Optional[Callable] = None,
    cache_data: bool = False,
    recalculate_norm_stats: bool = False
) -> ForestGapDataset:
    """
    Charge un dataset à partir d'un fichier de métadonnées.
    
    Args:
        metadata_path: Chemin vers le fichier de métadonnées JSON.
        transform: Fonction de transformation pour l'augmentation de données.
        cache_data: Si True, met en cache les données en mémoire.
        recalculate_norm_stats: Si True, recalcule les statistiques de normalisation.
        
    Returns:
        Instance de ForestGapDataset.
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extraire les informations
        input_paths = metadata.get("input_paths", [])
        mask_paths = metadata.get("mask_paths", [])
        threshold_value = metadata.get("threshold_value", 2.0)
        below_threshold = metadata.get("below_threshold", True)
        normalize = metadata.get("normalization", True)
        norm_stats = metadata.get("normalization_stats", None)
        
        if not mask_paths:
            mask_paths = None
        
        logger.info(f"Chargement du dataset à partir de {metadata_path} avec {len(input_paths)} échantillons")
        
        return ForestGapDataset(
            input_paths=input_paths,
            mask_paths=mask_paths,
            transform=transform,
            threshold_value=threshold_value,
            below_threshold=below_threshold,
            cache_data=cache_data,
            normalize=normalize,
            calculate_norm_stats=recalculate_norm_stats,
            norm_stats=norm_stats,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset à partir de {metadata_path}: {str(e)}")
        raise


def split_dataset(
    dataset: ForestGapDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[ForestGapDataset, ForestGapDataset, ForestGapDataset]:
    """
    Divise un dataset en ensembles d'entraînement, validation et test.
    
    Args:
        dataset: Dataset à diviser.
        train_ratio: Proportion pour l'entraînement.
        val_ratio: Proportion pour la validation.
        test_ratio: Proportion pour le test.
        seed: Graine aléatoire pour la reproductibilité.
        
    Returns:
        Tuple contenant les datasets d'entraînement, validation et test.
    """
    # Vérifier que les ratios somment à 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        logger.warning("Les ratios ne somment pas à 1. Normalisation appliquée.")
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    # Calculer les tailles des sous-ensembles
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Créer le générateur aléatoire
    generator = torch.Generator().manual_seed(seed)
    
    # Diviser le dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    logger.info(f"Dataset divisé en: train={train_size}, validation={val_size}, test={test_size}")
    
    return train_dataset, val_dataset, test_dataset


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Exemple avec des chemins fictifs
    input_paths = [f"path/to/chm_{i}.tif" for i in range(10)]
    mask_paths = [f"path/to/mask_{i}.tif" for i in range(10)]
    
    # Création du dataset
    dataset = create_gap_dataset(
        input_paths=input_paths,
        mask_paths=mask_paths,
        threshold_value=2.0,
        normalize=True,
        cache_data=True,
        metadata_output="path/to/metadata.json"
    )
    
    # Division en ensembles d'entraînement, validation et test
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    
    print(f"Dataset total: {len(dataset)} échantillons")
    print(f"Dataset d'entraînement: {len(train_dataset)} échantillons")
    print(f"Dataset de validation: {len(val_dataset)} échantillons")
    print(f"Dataset de test: {len(test_dataset)} échantillons")
