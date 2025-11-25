"""
Module d'archivage et de stockage optimisé des données pour la détection des trouées forestières.

Ce module fournit des fonctionnalités pour stocker et charger efficacement des données
formatées pour l'entraînement des modèles de détection des trouées forestières.
"""

import os
import tarfile
import json
import logging
import shutil
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Iterator
from pathlib import Path
import io
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import rasterio
from tqdm import tqdm

# Configuration du logger
logger = logging.getLogger(__name__)


class TarArchiveDataset(Dataset):
    """
    Dataset basé sur une archive tar pour un accès séquentiel optimisé.
    
    Cette classe permet de stocker et charger efficacement des paires d'images et de masques
    dans une archive tar, ce qui réduit les opérations d'I/O et améliore les performances
    des DataLoaders, particulièrement dans des environnements comme Google Colab.
    
    Attributes:
        archive_path (str): Chemin vers l'archive tar.
        index (Dict): Index des fichiers dans l'archive avec leurs offsets.
        transform (Callable): Fonction de transformation pour l'augmentation de données.
        metadata (Dict): Métadonnées du dataset.
    """
    
    def __init__(
        self,
        archive_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        metadata_file: str = 'metadata.json'
    ):
        """
        Initialise le dataset basé sur une archive tar.
        
        Args:
            archive_path: Chemin vers l'archive tar.
            transform: Fonction de transformation pour les images d'entrée.
            target_transform: Fonction de transformation pour les masques.
            metadata_file: Nom du fichier de métadonnées dans l'archive.
        """
        self.archive_path = archive_path
        self.transform = transform
        self.target_transform = target_transform
        
        # Vérifier que l'archive existe
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive tar non trouvée: {archive_path}")
        
        # Créer l'index des fichiers dans l'archive
        self.index = self._create_index()
        
        # Charger les métadonnées
        self.metadata = self._load_metadata(metadata_file)
        
        # Liste des paires input/mask
        self.pairs = self._extract_pairs()
        
        logger.info(f"TarArchiveDataset initialisé avec {len(self.pairs)} paires d'échantillons "
                   f"depuis {archive_path}")
    
    def _create_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Crée un index des fichiers dans l'archive avec leurs offsets.
        
        Returns:
            Dictionnaire {nom_fichier: {offset, taille, ...}}
        """
        index = {}
        
        logger.info(f"Création de l'index pour l'archive {self.archive_path}...")
        
        with tarfile.open(self.archive_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    index[member.name] = {
                        'offset': member.offset_data,
                        'size': member.size,
                        'type': os.path.splitext(member.name)[1]
                    }
        
        logger.info(f"Index créé avec {len(index)} fichiers")
        return index
    
    def _load_metadata(self, metadata_file: str) -> Dict[str, Any]:
        """
        Charge les métadonnées depuis l'archive.
        
        Args:
            metadata_file: Nom du fichier de métadonnées dans l'archive.
            
        Returns:
            Dictionnaire des métadonnées.
        """
        if metadata_file in self.index:
            with tarfile.open(self.archive_path, 'r') as tar:
                f = tar.extractfile(metadata_file)
                if f:
                    return json.load(f)
        
        logger.warning(f"Fichier de métadonnées {metadata_file} non trouvé dans l'archive")
        return {}
    
    def _extract_pairs(self) -> List[Dict[str, str]]:
        """
        Extrait la liste des paires input/mask depuis l'index.
        
        Returns:
            Liste de dictionnaires {input: chemin_input, mask: chemin_mask}
        """
        # Regrouper les fichiers par préfixe (nom sans extension)
        files_by_prefix = {}
        for filename in self.index:
            # Ignorer les fichiers de métadonnées
            if filename.endswith('.json'):
                continue
            
            prefix = os.path.splitext(filename)[0]
            
            # Séparer le préfixe de l'indicateur input/mask si présent
            parts = prefix.rsplit('_', 1)
            if len(parts) > 1 and parts[1] in ['input', 'mask']:
                real_prefix = parts[0]
                file_type = parts[1]
            else:
                real_prefix = prefix
                # Détecter le type par d'autres moyens (extension, contenu du chemin)
                if 'input' in filename.lower() or 'dsm' in filename.lower() or 'chm' in filename.lower():
                    file_type = 'input'
                elif 'mask' in filename.lower():
                    file_type = 'mask'
                else:
                    logger.warning(f"Impossible de déterminer le type pour {filename}")
                    continue
            
            if real_prefix not in files_by_prefix:
                files_by_prefix[real_prefix] = {}
            
            files_by_prefix[real_prefix][file_type] = filename
        
        # Créer la liste des paires
        pairs = []
        for prefix, files in files_by_prefix.items():
            if 'input' in files and 'mask' in files:
                pairs.append({
                    'input': files['input'],
                    'mask': files['mask'],
                    'prefix': prefix
                })
            else:
                logger.warning(f"Paire incomplète pour le préfixe {prefix}")
        
        return pairs
    
    def __len__(self) -> int:
        """
        Retourne le nombre de paires d'échantillons dans le dataset.
        
        Returns:
            Nombre d'échantillons.
        """
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Charge une paire d'échantillons depuis l'archive.
        
        Args:
            idx: Indice de la paire à charger.
            
        Returns:
            Tuple (input, mask).
        """
        pair = self.pairs[idx]
        
        # Charger l'input
        input_data = self._load_file(pair['input'])
        
        # Charger le mask
        mask_data = self._load_file(pair['mask'])
        
        # Convertir en tensors
        input_tensor = torch.from_numpy(input_data).float()
        mask_tensor = torch.from_numpy(mask_data).float()
        
        # Appliquer les transformations
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
        
        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask_tensor)
        
        return input_tensor, mask_tensor
    
    def _load_file(self, filename: str) -> np.ndarray:
        """
        Charge un fichier depuis l'archive en mémoire.
        
        Args:
            filename: Nom du fichier dans l'archive.
            
        Returns:
            Données chargées sous forme de ndarray.
        """
        if filename not in self.index:
            raise ValueError(f"Fichier {filename} non trouvé dans l'archive")
        
        with tarfile.open(self.archive_path, 'r') as tar:
            f = tar.extractfile(filename)
            if f is None:
                raise ValueError(f"Impossible d'extraire {filename} de l'archive")
            
            # Détecter le type de fichier et charger en conséquence
            file_type = self.index[filename]['type']
            
            if file_type.lower() in ['.npy']:
                # Fichier NumPy
                buffer = io.BytesIO(f.read())
                return np.load(buffer)
                
            elif file_type.lower() in ['.tif', '.tiff']:
                # Fichier GeoTIFF
                buffer = io.BytesIO(f.read())
                with rasterio.open(buffer) as src:
                    return src.read()
                
            else:
                # Format inconnu, tenter de charger comme NumPy
                logger.warning(f"Type de fichier inconnu {file_type}, tentative de chargement comme NumPy")
                buffer = io.BytesIO(f.read())
                try:
                    return np.load(buffer)
                except:
                    raise ValueError(f"Impossible de charger {filename} avec le type {file_type}")


class IterableTarArchiveDataset(IterableDataset):
    """
    Dataset itérable basé sur une archive tar pour un streaming efficace des données.
    
    Cette classe permet de streamer des paires d'images et de masques depuis une archive tar,
    ce qui est particulièrement utile pour les grands datasets qui ne tiennent pas en mémoire.
    
    Attributes:
        archive_path (str): Chemin vers l'archive tar.
        transform (Callable): Fonction de transformation pour l'augmentation de données.
        metadata (Dict): Métadonnées du dataset.
        shuffle (bool): Si True, mélange les données.
    """
    
    def __init__(
        self,
        archive_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = True,
        metadata_file: str = 'metadata.json',
        buffer_size: int = 100
    ):
        """
        Initialise le dataset itérable basé sur une archive tar.
        
        Args:
            archive_path: Chemin vers l'archive tar.
            transform: Fonction de transformation pour les images d'entrée.
            target_transform: Fonction de transformation pour les masques.
            shuffle: Si True, mélange les données.
            metadata_file: Nom du fichier de métadonnées dans l'archive.
            buffer_size: Taille du buffer pour le shuffling.
        """
        self.archive_path = archive_path
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        
        # Vérifier que l'archive existe
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive tar non trouvée: {archive_path}")
        
        # Créer l'index des fichiers dans l'archive
        self.index = self._create_index()
        
        # Charger les métadonnées
        self.metadata = self._load_metadata(metadata_file)
        
        # Liste des paires input/mask
        self.pairs = self._extract_pairs()
        
        logger.info(f"IterableTarArchiveDataset initialisé avec {len(self.pairs)} paires d'échantillons "
                   f"depuis {archive_path}")
    
    def _create_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Crée un index des fichiers dans l'archive avec leurs offsets.
        
        Returns:
            Dictionnaire {nom_fichier: {offset, taille, ...}}
        """
        index = {}
        
        logger.info(f"Création de l'index pour l'archive {self.archive_path}...")
        
        with tarfile.open(self.archive_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    index[member.name] = {
                        'offset': member.offset_data,
                        'size': member.size,
                        'type': os.path.splitext(member.name)[1]
                    }
        
        logger.info(f"Index créé avec {len(index)} fichiers")
        return index
    
    def _load_metadata(self, metadata_file: str) -> Dict[str, Any]:
        """
        Charge les métadonnées depuis l'archive.
        
        Args:
            metadata_file: Nom du fichier de métadonnées dans l'archive.
            
        Returns:
            Dictionnaire des métadonnées.
        """
        if metadata_file in self.index:
            with tarfile.open(self.archive_path, 'r') as tar:
                f = tar.extractfile(metadata_file)
                if f:
                    return json.load(f)
        
        logger.warning(f"Fichier de métadonnées {metadata_file} non trouvé dans l'archive")
        return {}
    
    def _extract_pairs(self) -> List[Dict[str, str]]:
        """
        Extrait la liste des paires input/mask depuis l'index.
        
        Returns:
            Liste de dictionnaires {input: chemin_input, mask: chemin_mask}
        """
        # Même logique que pour TarArchiveDataset
        files_by_prefix = {}
        for filename in self.index:
            if filename.endswith('.json'):
                continue
            
            prefix = os.path.splitext(filename)[0]
            
            parts = prefix.rsplit('_', 1)
            if len(parts) > 1 and parts[1] in ['input', 'mask']:
                real_prefix = parts[0]
                file_type = parts[1]
            else:
                real_prefix = prefix
                if 'input' in filename.lower() or 'dsm' in filename.lower() or 'chm' in filename.lower():
                    file_type = 'input'
                elif 'mask' in filename.lower():
                    file_type = 'mask'
                else:
                    logger.warning(f"Impossible de déterminer le type pour {filename}")
                    continue
            
            if real_prefix not in files_by_prefix:
                files_by_prefix[real_prefix] = {}
            
            files_by_prefix[real_prefix][file_type] = filename
        
        pairs = []
        for prefix, files in files_by_prefix.items():
            if 'input' in files and 'mask' in files:
                pairs.append({
                    'input': files['input'],
                    'mask': files['mask'],
                    'prefix': prefix
                })
            else:
                logger.warning(f"Paire incomplète pour le préfixe {prefix}")
        
        return pairs
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Crée un itérateur sur les paires d'échantillons.
        
        Returns:
            Itérateur sur les paires (input, mask).
        """
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            indices = list(range(len(self.pairs)))
            if self.shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                yield self._get_item(idx)
        else:
            # Multi-process data loading: Split workload
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            
            # Split indices among workers
            indices = list(range(len(self.pairs)))
            if self.shuffle:
                np.random.shuffle(indices)
            
            per_worker = int(np.ceil(len(indices) / num_workers))
            worker_indices = indices[worker_id * per_worker : (worker_id + 1) * per_worker]
            
            for idx in worker_indices:
                yield self._get_item(idx)
    
    def _get_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Charge une paire d'échantillons depuis l'archive.
        
        Args:
            idx: Indice de la paire à charger.
            
        Returns:
            Tuple (input, mask).
        """
        pair = self.pairs[idx]
        
        # Charger l'input
        input_data = self._load_file(pair['input'])
        
        # Charger le mask
        mask_data = self._load_file(pair['mask'])
        
        # Convertir en tensors
        input_tensor = torch.from_numpy(input_data).float()
        mask_tensor = torch.from_numpy(mask_data).float()
        
        # Appliquer les transformations
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
        
        if self.target_transform is not None:
            mask_tensor = self.target_transform(mask_tensor)
        
        return input_tensor, mask_tensor
    
    def _load_file(self, filename: str) -> np.ndarray:
        """
        Charge un fichier depuis l'archive en mémoire.
        
        Args:
            filename: Nom du fichier dans l'archive.
            
        Returns:
            Données chargées sous forme de ndarray.
        """
        if filename not in self.index:
            raise ValueError(f"Fichier {filename} non trouvé dans l'archive")
        
        with tarfile.open(self.archive_path, 'r') as tar:
            f = tar.extractfile(filename)
            if f is None:
                raise ValueError(f"Impossible d'extraire {filename} de l'archive")
            
            # Détecter le type de fichier et charger en conséquence
            file_type = self.index[filename]['type']
            
            if file_type.lower() in ['.npy']:
                # Fichier NumPy
                buffer = io.BytesIO(f.read())
                return np.load(buffer)
                
            elif file_type.lower() in ['.tif', '.tiff']:
                # Fichier GeoTIFF
                buffer = io.BytesIO(f.read())
                with rasterio.open(buffer) as src:
                    return src.read()
                
            else:
                # Format inconnu, tenter de charger comme NumPy
                logger.warning(f"Type de fichier inconnu {file_type}, tentative de chargement comme NumPy")
                buffer = io.BytesIO(f.read())
                try:
                    return np.load(buffer)
                except:
                    raise ValueError(f"Impossible de charger {filename} avec le type {file_type}")


def create_tar_archive(
    input_paths: List[str],
    mask_paths: List[str],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = 'gz',
    transform_input: Optional[Callable] = None,
    transform_mask: Optional[Callable] = None,
    show_progress: bool = True
) -> str:
    """
    Crée une archive tar contenant des paires d'images et de masques.
    
    Args:
        input_paths: Liste des chemins vers les fichiers d'entrée.
        mask_paths: Liste des chemins vers les fichiers de masque.
        output_path: Chemin de sortie pour l'archive tar.
        metadata: Métadonnées à inclure dans l'archive.
        compression: Type de compression ('gz', 'bz2', 'xz' ou None).
        transform_input: Fonction de transformation pour les images d'entrée.
        transform_mask: Fonction de transformation pour les masques.
        show_progress: Si True, affiche une barre de progression.
        
    Returns:
        Chemin de l'archive créée.
    """
    if len(input_paths) != len(mask_paths):
        raise ValueError("Les listes input_paths et mask_paths doivent avoir la même longueur")
    
    # Déterminer le mode d'ouverture en fonction de la compression
    mode = 'w'
    if compression == 'gz':
        mode += ':gz'
    elif compression == 'bz2':
        mode += ':bz2'
    elif compression == 'xz':
        mode += ':xz'
    elif compression is not None:
        raise ValueError(f"Type de compression inconnu: {compression}")
    
    # Créer le répertoire parent si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Créer l'archive
    with tarfile.open(output_path, mode) as tar:
        # Ajouter les métadonnées
        if metadata is not None:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(metadata, f, indent=2)
                metadata_path = f.name
            
            tar.add(metadata_path, arcname='metadata.json')
            os.unlink(metadata_path)
        
        # Ajouter les fichiers
        iterator = zip(input_paths, mask_paths)
        if show_progress:
            iterator = tqdm(iterator, total=len(input_paths), desc="Création de l'archive")
        
        for i, (input_path, mask_path) in enumerate(iterator):
            # Extraire le préfixe
            input_name = os.path.basename(input_path)
            input_prefix = os.path.splitext(input_name)[0]
            
            # Noms des fichiers dans l'archive
            archive_input_name = f"{input_prefix}_input{os.path.splitext(input_path)[1]}"
            archive_mask_name = f"{input_prefix}_mask{os.path.splitext(mask_path)[1]}"
            
            # Ajouter l'input
            if transform_input is not None:
                # Appliquer la transformation et sauvegarder temporairement
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(input_path)[1], delete=False) as f:
                    temp_input_path = f.name
                transform_input(input_path, temp_input_path)
                tar.add(temp_input_path, arcname=archive_input_name)
                os.unlink(temp_input_path)
            else:
                tar.add(input_path, arcname=archive_input_name)
            
            # Ajouter le mask
            if transform_mask is not None:
                # Appliquer la transformation et sauvegarder temporairement
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(mask_path)[1], delete=False) as f:
                    temp_mask_path = f.name
                transform_mask(mask_path, temp_mask_path)
                tar.add(temp_mask_path, arcname=archive_mask_name)
                os.unlink(temp_mask_path)
            else:
                tar.add(mask_path, arcname=archive_mask_name)
    
    logger.info(f"Archive créée avec succès: {output_path}")
    return output_path


def convert_dataset_to_tar(
    dataset: Dataset,
    output_path: str,
    compression: str = 'gz',
    batch_size: int = 32,
    show_progress: bool = True
) -> str:
    """
    Convertit un dataset PyTorch en archive tar.
    
    Args:
        dataset: Dataset à convertir.
        output_path: Chemin de sortie pour l'archive tar.
        compression: Type de compression ('gz', 'bz2', 'xz' ou None).
        batch_size: Taille des batchs pour le traitement.
        show_progress: Si True, affiche une barre de progression.
        
    Returns:
        Chemin de l'archive créée.
    """
    # Créer un répertoire temporaire pour stocker les fichiers
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, 'input')
    mask_dir = os.path.join(temp_dir, 'mask')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    try:
        # Créer les chemins d'entrée et de sortie
        input_paths = []
        mask_paths = []
        
        # Traiter le dataset par batchs
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Extraire les métadonnées si disponibles
        metadata = getattr(dataset, 'metadata', {})
        
        # Parcourir le dataset
        start_idx = 0
        iterator = dataloader
        if show_progress:
            iterator = tqdm(iterator, desc="Extraction des données")
        
        for batch_inputs, batch_masks in iterator:
            for i in range(len(batch_inputs)):
                # Sauvegarder l'input
                input_path = os.path.join(input_dir, f"sample_{start_idx + i:06d}.npy")
                np.save(input_path, batch_inputs[i].numpy())
                input_paths.append(input_path)
                
                # Sauvegarder le mask
                mask_path = os.path.join(mask_dir, f"sample_{start_idx + i:06d}.npy")
                np.save(mask_path, batch_masks[i].numpy())
                mask_paths.append(mask_path)
            
            start_idx += len(batch_inputs)
        
        # Créer l'archive tar
        create_tar_archive(
            input_paths=input_paths,
            mask_paths=mask_paths,
            output_path=output_path,
            metadata=metadata,
            compression=compression,
            show_progress=show_progress
        )
        
        return output_path
    
    finally:
        # Nettoyer le répertoire temporaire
        shutil.rmtree(temp_dir, ignore_errors=True) 