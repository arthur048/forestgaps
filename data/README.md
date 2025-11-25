# Module de Données (data)

Ce module fournit toutes les fonctionnalités liées à la préparation, au traitement et au chargement des données pour les modèles de détection de trouées forestières.

## Structure du module

```
data/
├── __init__.py               # Point d'entrée unifié
├── preprocessing/            # Préparation des rasters
│   ├── __init__.py
│   ├── alignment.py          # Alignement des rasters (DSM/CHM)
│   ├── analysis.py           # Analyse des rasters
│   └── conversion.py         # Conversion et traitement des formats
├── datasets/                 # Datasets PyTorch
│   ├── __init__.py
│   ├── gap_dataset.py        # Dataset pour la segmentation de trouées
│   ├── regression_dataset.py # Dataset pour la régression (prédiction CHM)
│   ├── transforms.py         # Transformations pour l'augmentation
│   └── samplers.py           # Stratégies d'échantillonnage équilibré
├── generation/               # Génération des tuiles
│   ├── __init__.py
│   ├── tiling.py             # Découpage en tuiles
│   └── masks.py              # Création des masques de trouées
├── normalization/            # Normalisation des données
│   ├── __init__.py
│   ├── standardization.py    # Techniques de standardisation
│   └── statistics.py         # Calcul et gestion des statistiques
├── loaders/                  # DataLoaders optimisés
│   ├── __init__.py
│   ├── factory.py            # Création des DataLoaders
│   └── optimization.py       # Optimisation dynamique des paramètres
└── storage/                  # Stockage persistant
    ├── __init__.py
    ├── io.py                 # Entrées/sorties
    └── compression.py        # Compression et optimisation du stockage
```

## Fonctionnalités principales

### Prétraitement des données raster

Le sous-module `preprocessing` contient les fonctionnalités pour analyser, aligner et convertir les données raster (DSM/CHM) :

- **Analyse des rasters** : Vérification de l'intégrité, statistiques descriptives, détection des problèmes
- **Alignement des rasters** : Alignement spatial entre DSM et CHM pour garantir la correspondance pixel à pixel
- **Conversion** : Conversion entre formats (GeoTIFF, NumPy, etc.)

```python
from forestgaps.data.preprocessing import analyze_raster_pair, align_rasters

# Analyser une paire de rasters DSM/CHM
analysis = analyze_raster_pair("path/to/dsm.tif", "path/to/chm.tif", "site1")

# Aligner les rasters si nécessaire
if not analysis["is_aligned"]:
    aligned_dsm, aligned_chm = align_rasters(
        "path/to/dsm.tif", 
        "path/to/chm.tif", 
        output_dir="path/to/aligned"
    )
```

### Génération de données

Le sous-module `generation` permet de créer des tuiles et des masques à partir des données raster :

- **Tuilage** : Découpage des rasters en tuiles de taille fixe avec gestion du chevauchement
- **Création de masques** : Génération de masques binaires pour les trouées à différents seuils de hauteur

```python
from forestgaps.data.generation import generate_tiles, generate_gap_masks

# Générer des tuiles à partir d'un raster
tiles = generate_tiles(
    "path/to/aligned_dsm.tif", 
    tile_size=256, 
    overlap=0.2,
    output_dir="path/to/tiles"
)

# Créer des masques de trouées à différents seuils
thresholds = [2.0, 5.0, 10.0, 15.0]
masks = generate_gap_masks(
    "path/to/aligned_chm.tif", 
    thresholds, 
    output_dir="path/to/masks"
)
```

### Datasets PyTorch

Le sous-module `datasets` fournit des implémentations de datasets PyTorch pour différentes tâches :

- **ForestGapDataset** : Dataset pour la segmentation de trouées forestières
- **ForestRegressionDataset** : Dataset pour la régression (prédiction de CHM à partir de DSM)
- **Transformations** : Augmentation de données (rotation, flip, élastique, etc.)
- **Échantillonnage** : Stratégies d'échantillonnage pour gérer le déséquilibre des classes

```python
from forestgaps.data.datasets import create_gap_dataset, create_regression_dataset

# Créer un dataset de segmentation
gap_dataset = create_gap_dataset(
    dsm_files=dsm_files,
    mask_files=mask_files,
    transform_config={"is_train": True, "prob": 0.5}
)

# Créer un dataset de régression
regression_dataset = create_regression_dataset(
    dsm_files=dsm_files,
    chm_files=chm_files,
    normalize=True
)
```

### Normalisation des données

Le sous-module `normalization` gère la normalisation et la standardisation des données :

- **Standardisation** : Normalisation min-max, z-score, etc.
- **Statistiques** : Calcul et stockage des statistiques pour une normalisation cohérente

```python
from forestgaps.data.normalization import normalize_data, compute_statistics

# Calculer les statistiques sur un ensemble de données
stats = compute_statistics(dsm_files)

# Normaliser une image avec les statistiques calculées
normalized_dsm = normalize_data(dsm, stats, method="min_max")
```

### DataLoaders optimisés

Le sous-module `loaders` permet de créer et d'optimiser des DataLoaders PyTorch :

- **Factory** : Création simplifiée de DataLoaders configurables
- **Optimisation** : Calibration dynamique des paramètres pour maximiser les performances

```python
from forestgaps.data.loaders import create_data_loaders, optimize_dataloader_params

# Créer des DataLoaders à partir de la configuration
data_loaders = create_data_loaders(config)

# Optimiser les paramètres du DataLoader
optimal_params = optimize_dataloader_params(
    dataset=gap_dataset,
    batch_size=16,
    max_workers=4
)
```

### Stockage persistant

Le sous-module `storage` gère le stockage et la persistance des données :

- **I/O** : Opérations d'entrée/sortie optimisées
- **Compression** : Techniques de compression pour réduire l'espace de stockage

```python
from forestgaps.data.storage import save_compressed_dataset, load_compressed_dataset

# Sauvegarder un dataset compressé
save_compressed_dataset(dataset, "path/to/compressed_dataset.npz")

# Charger un dataset compressé
loaded_dataset = load_compressed_dataset("path/to/compressed_dataset.npz")
```

## Dépendances

Le module `data` dépend des modules suivants :
- `config` : Pour la gestion des paramètres de configuration
- `environment` : Pour l'adaptation aux différents environnements d'exécution
- `utils` : Pour les fonctions utilitaires et la gestion des erreurs

## Utilisation avec Google Colab

Le module est conçu pour fonctionner de manière transparente dans Google Colab :

```python
from forestgaps.environment import setup_environment
from forestgaps.config import load_default_config
from forestgaps.data.loaders import create_data_loaders

# Configurer l'environnement Colab
env = setup_environment()

# Charger la configuration
config = load_default_config()

# Créer les DataLoaders optimisés pour Colab
data_loaders = create_data_loaders(config)
``` 