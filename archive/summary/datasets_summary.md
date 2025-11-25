# Résumé du module `data/datasets/`

## Description du module

Le module `data/datasets/` est responsable de la gestion des jeux de données pour la détection de trouées forestières. Il fournit des classes et fonctions pour charger, transformer et équilibrer les données utilisées pour l'entraînement des modèles.

## Fonctionnalités implémentées

### Sous-module `gap_dataset.py`

**Classes et fonctions principales :**
- `ForestGapDataset`: Classe principale qui charge des paires d'images (DSM/CHM) et de masques pour l'entraînement de modèles de segmentation.
- `create_gap_dataset`: Fonction utilitaire pour créer facilement un dataset à partir de listes de chemins.
- `load_dataset_from_metadata`: Fonction pour charger un dataset à partir d'un fichier de métadonnées.
- `split_dataset`: Fonction pour diviser un dataset en ensembles d'entraînement, validation et test.

**Caractéristiques :**
- Cache en mémoire pour améliorer les performances
- Calcul et application de normalisation
- Gestion des métadonnées et sauvegarde/rechargement
- Support des transformations d'augmentation de données
- Extraction efficace des paires image/masque

### Sous-module `samplers.py`

**Classes et fonctions principales :**
- `calculate_gap_ratios`: Calcule les proportions de trouées forestières dans les masques.
- `create_weighted_sampler`: Crée un échantillonneur pondéré pour équilibrer les classes.
- `BalancedGapSampler`: Classe d'échantillonnage personnalisée pour l'équilibrage des batches.

**Caractéristiques :**
- Différentes stratégies d'équilibrage (inverse, racine carrée inverse, égale)
- Modes d'équilibrage par batch ou global
- Support de l'échantillonnage alternatif
- Calcul et utilisation des ratios de trouées
- Personnalisation de la stratégie d'échantillonnage

### Sous-module `transforms.py`

**Classes et fonctions principales :**
- `ForestGapTransforms`: Transformations CPU pour les paires image/masque.
- `GpuTransforms`: Transformations GPU accélérées utilisant Kornia.
- `elastic_transform`: Fonction d'augmentation par transformation élastique.
- `create_transform_pipeline`: Fonction pour créer le pipeline de transformation adapté.

**Caractéristiques :**
- Support des transformations CPU et GPU
- Transformations classiques: retournement, rotation, recadrage, bruit
- Transformations avancées: élastique, flou de mouvement, perspective
- Application cohérente entre images et masques
- Options flexibles de configuration

## Architecture et design

Le module suit les principes SOLID et une approche modulaire:
- **S** (Responsabilité unique): Chaque fichier a une responsabilité claire et bien définie.
- **O** (Ouvert/fermé): Les classes sont conçues pour être extensibles via l'héritage ou la composition.
- **L** (Substitution de Liskov): Les interfaces sont cohérentes, notamment pour les transformations.
- **I** (Ségrégation des interfaces): Les fonctionnalités sont regroupées logiquement.
- **D** (Inversion des dépendances): Les composants dépendent d'abstractions (ex: transforms).

## Dépendances externes

Le module dépend des bibliothèques suivantes:
- `torch`: Fonctionnalités de base pour les datasets et samplers
- `numpy`: Manipulation efficace des données
- `cv2`: Opérations de transformation d'image
- `scipy`: Transformations élastiques
- `kornia` (optionnel): Transformations GPU accélérées
- `rasterio`: Chargement des données géospatiales

## Exemple d'utilisation

```python
import torch
import logging
from torch.utils.data import DataLoader
from forestgaps.data.datasets import ForestGapDataset, BalancedGapSampler, ForestGapTransforms

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Chemins d'exemple
input_paths = ["path/to/chm1.tif", "path/to/chm2.tif"]
mask_paths = ["path/to/mask1.tif", "path/to/mask2.tif"]

# Création des transformations
transforms = ForestGapTransforms(
    is_train=True,
    prob=0.7,
    enable_elastic=True,
    enable_flip=True
)

# Création du dataset
dataset = ForestGapDataset(
    input_paths=input_paths,
    mask_paths=mask_paths,
    transform=transforms,
    threshold_value=2.0,
    cache_data=True
)

# Division en ensembles d'entraînement et validation
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)

# Création d'un échantillonneur équilibré
sampler = BalancedGapSampler(
    dataset=train_dataset,
    mask_paths=mask_paths,
    threshold_value=2.0,
    balance_mode="batch"
)

# Création du DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    sampler=sampler,
    num_workers=2
)

# Utilisation du DataLoader
for batch_idx, (images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx}: images shape: {images.shape}, masks shape: {masks.shape}")
    # Traitement du batch...
```

## Fonctionnalités avancées

### Gestion de la mémoire
- Mise en cache paramétrable pour optimiser les performances
- Chargement à la demande pour économiser la mémoire
- Préchargement en mémoire pour les datasets qui tiennent en RAM

### Équilibrage des classes
- Calcul précis des ratios de trouées
- Stratégies multiples d'échantillonnage
- Adaptation à différents scénarios d'équilibrage

### Pipeline de transformations
- Transformations cohérentes entre images et masques
- Support de configurations CPU/GPU
- Extensibilité pour ajouter de nouvelles transformations

## Améliorations futures

- Implémentation d'une stratégie de chargement parallèle plus avancée
- Support des formats de dataset distribués (WebDataset, etc.)
- Ajout de nouvelles stratégies d'échantillonnage (curriculum learning)
- Optimisation des performances pour les très grands datasets
- Support de l'entraînement multi-GPU 