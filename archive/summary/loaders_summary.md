# Résumé du module `data/loaders/`

## Description du module

Le module `data/loaders/` est responsable de l'optimisation et de la gestion efficace des chargeurs de données (DataLoaders) pour l'entraînement et l'inférence des modèles de détection de trouées forestières. Ce module fournit des fonctionnalités avancées pour améliorer les performances de chargement des données, s'adapter automatiquement aux ressources disponibles, et optimiser l'utilisation de la mémoire et du processeur.

## Fonctionnalités implémentées

### Sous-module `factory.py`

**Classes et fonctions principales :**
- `create_dataloader`: Fonction pour créer facilement un DataLoader avec des paramètres optimisés
- `create_dataset`: Fonction pour créer un dataset à partir de chemins d'entrée et de masque
- `create_train_val_dataloaders`: Fonction pour créer des DataLoaders d'entraînement et de validation

**Caractéristiques :**
- Configuration simplifiée des DataLoaders
- Support des paramètres avancés (prefetch_factor, persistent_workers)
- Intégration avec le module datasets
- Gestion des métadonnées et options de normalisation

### Sous-module `optimization.py`

**Classes et fonctions principales :**
- `benchmark_dataloader`: Fonction pour mesurer les performances d'un DataLoader
- `optimize_batch_size`: Fonction pour déterminer la taille de batch optimale
- `optimize_num_workers`: Fonction pour déterminer le nombre optimal de workers
- `prefetch_data`: Fonction pour précharger des données en mémoire
- `optimize_dataloader`: Fonction pour optimiser tous les paramètres d'un DataLoader

**Caractéristiques :**
- Mesure précise des temps de chargement et de traitement
- Optimisation basée sur le débit (échantillons/seconde)
- Support du plotting pour visualiser les résultats
- Tests de différentes configurations pour trouver l'optimum

### Sous-module `calibration.py`

**Classes et fonctions principales :**
- `DataLoaderCalibrator`: Classe pour calibrer automatiquement les DataLoaders
- `create_calibrated_dataloader`: Fonction pour créer un DataLoader calibré
- `create_calibrated_train_val_dataloaders`: Fonction pour créer des DataLoaders calibrés pour l'entraînement et la validation

**Caractéristiques :**
- Détection automatique de l'environnement (Colab, local)
- Ajustement des paramètres en fonction des ressources disponibles
- Système de cache pour les résultats de calibration
- Adaptation intelligente au matériel et aux caractéristiques des données

### Sous-module `archive.py`

**Classes et fonctions principales :**
- `TarArchiveDataset`: Dataset basé sur une archive tar pour un accès séquentiel optimisé
- `IterableTarArchiveDataset`: Dataset itérable pour le streaming de données depuis une archive tar
- `create_tar_archive`: Fonction pour créer une archive tar à partir de paires d'images et de masques
- `convert_dataset_to_tar`: Fonction pour convertir un dataset PyTorch en archive tar

**Caractéristiques :**
- Stockage efficace et compression des données
- Accès séquentiel optimisé pour réduire les opérations d'I/O
- Support du streaming pour les grands datasets
- Indexation rapide des fichiers et gestion des métadonnées

## Architecture et design

Le module suit une architecture modulaire avec une séparation claire des responsabilités :
- **Factory**: Création et configuration des DataLoaders et datasets
- **Optimization**: Mesure et optimisation des performances
- **Calibration**: Adaptation automatique aux ressources disponibles
- **Archive**: Stockage et accès optimisés aux données

Cette organisation permet une grande flexibilité tout en maintenant un code propre et maintenable.

## Dépendances externes

Le module dépend des bibliothèques suivantes :
- `torch`: Fonctionnalités de base pour les DataLoaders
- `numpy`: Manipulation efficace des données
- `matplotlib`: Visualisation des résultats d'optimisation
- `tqdm`: Affichage des barres de progression
- `rasterio`: Chargement des données géospatiales
- Modules internes: `data.datasets`, `data.normalization`

## Optimisations clés

1. **Calibration dynamique**: Adaptation automatique des paramètres en fonction de l'environnement
2. **Archivage tar**: Réduction drastique des opérations d'I/O en accédant aux fichiers séquentiellement
3. **Prefetching optimisé**: Préchargement intelligent des données pour réduire les temps d'attente
4. **Multi-processing ajusté**: Équilibrage entre parallélisme et utilisation de la mémoire
5. **Streaming pour grands datasets**: Gestion optimisée des datasets qui ne tiennent pas en mémoire

## Exemple d'utilisation

```python
import torch
from data.datasets import ForestGapDataset
from data.loaders import create_calibrated_dataloader, TarArchiveDataset

# Exemple 1: Création d'un DataLoader calibré
dataset = ForestGapDataset(input_paths, mask_paths)
dataloader = create_calibrated_dataloader(
    dataset=dataset,
    shuffle=True,
    drop_last=True
)

# Exemple 2: Utilisation d'un dataset basé sur une archive tar
tar_dataset = TarArchiveDataset('data.tar.gz')
dataloader = torch.utils.data.DataLoader(
    tar_dataset,
    batch_size=16,
    num_workers=4
)

# Exemple 3: Conversion d'un dataset en archive tar
convert_dataset_to_tar(
    dataset=dataset,
    output_path='optimized_data.tar.gz',
    compression='gz'
)
```

## Considérations pour Google Colab

Le module intègre plusieurs optimisations spécifiques pour Google Colab :
- Limitation automatique du nombre de workers pour éviter les problèmes de mémoire
- Détection de l'environnement Colab pour ajuster les paramètres
- Support des archives tar pour un stockage et un chargement efficaces depuis Google Drive
- Calibration adaptée aux contraintes de ressources de Colab

## Améliorations futures

1. Support de formats de compression plus avancés (LZMA, Zstandard)
2. Intégration avec des bibliothèques de préchargement comme DALI
3. Support de la mise en cache sur SSD pour les environnements cloud
4. Mécanisme de sharding pour la distribution sur plusieurs machines 