# Documentation complète du package ForestGaps

## Sommaire

1. [Vue d'ensemble du package](#1-vue-densemble-du-package)
   - [Introduction](#introduction)
   - [Architecture globale](#architecture-globale)
   - [Diagramme de dépendances](#diagramme-de-dépendances)
   - [Workflow typique](#workflow-typique)

2. [Guide d'installation et prérequis](#2-guide-dinstallation-et-prérequis)
   - [Installation standard](#installation-standard)
   - [Installation dans Google Colab](#installation-dans-google-colab)
   - [Dépendances](#dépendances)

3. [Référence API complète](#3-référence-api-complète)
   - [Module environment](#module-environment)
   - [Module config](#module-config)
   - [Module data](#module-data)
   - [Module models](#module-models)
   - [Module training](#module-training)
   - [Module evaluation](#module-evaluation)
   - [Module inference](#module-inference)
   - [Module utils](#module-utils)
   - [Module cli](#module-cli)
   - [Module benchmarking](#module-benchmarking)

4. [Guides d'utilisation thématiques](#4-guides-dutilisation-thématiques)
   - [Prétraitement des données géospatiales](#prétraitement-des-données-géospatiales)
   - [Entraînement d'un modèle personnalisé](#entraînement-dun-modèle-personnalisé)
   - [Inférence sur de nouvelles zones](#inférence-sur-de-nouvelles-zones)
   - [Évaluation et comparaison des modèles](#évaluation-et-comparaison-des-modèles)

5. [FAQ et dépannage](#5-faq-et-dépannage)
   - [Questions fréquentes](#questions-fréquentes)
   - [Problèmes courants](#problèmes-courants)
   - [Limites connues](#limites-connues)

## 1. Vue d'ensemble du package

### Introduction

ForestGaps est une bibliothèque Python modulaire conçue pour la détection et l'analyse des trouées forestières à partir d'images de télédétection, en utilisant des techniques de deep learning. Les trouées forestières sont des ouvertures dans la canopée forestière résultant de la mort d'arbres, qui jouent un rôle crucial dans la dynamique forestière et la régénération des écosystèmes.

La bibliothèque utilise des modèles numériques de surface (DSM) et des modèles de hauteur de canopée (CHM) pour la segmentation des trouées et la régression de hauteur. Ces données sont typiquement obtenues par LiDAR aéroporté ou par photogrammétrie à partir d'images aériennes ou satellites.

**Objectifs et cas d'utilisation :**

1. **Segmentation automatique des trouées forestières** : Identification précise des trouées dans la canopée forestière à partir de données DSM/CHM.
2. **Régression de hauteur** : Estimation de la hauteur de la canopée à partir de modèles numériques de surface.
3. **Analyse comparative** : Comparaison systématique des performances de différentes architectures de deep learning.
4. **Traitement par lots** : Application efficace des modèles à de grandes zones forestières.
5. **Compatibilité multi-environnements** : Fonctionnement transparent dans Google Colab ou en environnement local.

### Architecture globale

Le package ForestGaps est organisé de manière modulaire avec une structure hiérarchique claire :

```
forestgaps/
├── __init__.py           # Point d'entrée principal
├── __version__.py        # Définition de la version
├── environment/          # Gestion de l'environnement d'exécution
├── config/               # Gestion de la configuration
├── data/                 # Traitement et gestion des données
├── models/               # Architectures de réseaux de neurones
├── training/             # Logique d'entraînement
├── evaluation/           # Évaluation des modèles
├── inference/            # Inférence avec modèles entraînés
├── utils/                # Fonctions utilitaires
├── cli/                  # Interface en ligne de commande
├── benchmarking/         # Comparaison systématique des modèles
├── examples/             # Exemples d'utilisation
└── tests/                # Tests unitaires et d'intégration
```

Le package suit une architecture modulaire avec une séparation claire des responsabilités entre les différents modules. Les principes SOLID sont appliqués pour garantir une architecture extensible et maintenable.

### Diagramme de dépendances

| Module         | Dépend de                                      | Est utilisé par                             | Responsabilité principale                                |
|----------------|------------------------------------------------|--------------------------------------------|---------------------------------------------------------|
| `config`       | -                                              | Tous les autres modules                    | Gestion centralisée de la configuration du projet        |
| `environment`  | `config`, `utils`                              | `data`, `models`, `training`, `cli`        | Détection et configuration de l'environnement d'exécution|
| `data`         | `config`, `environment`, `utils`               | `models`, `training`, `cli`, `benchmarking`| Préparation, transformation et chargement des données   |
| `models`       | `config`, `utils`                              | `training`, `cli`, `benchmarking`          | Implémentation des différentes architectures de réseaux  |
| `training`     | `config`, `data`, `models`, `utils`            | `cli`, `benchmarking`                      | Entraînement, évaluation et monitoring des modèles      |
| `evaluation`   | `config`, `models`, `utils`                    | `cli`, `benchmarking`                      | Évaluation externe des modèles entraînés                |
| `inference`    | `config`, `models`, `utils`                    | `cli`, `evaluation`                        | Application des modèles à de nouvelles données          |
| `utils`        | -                                              | Tous les autres modules                    | Fonctionnalités communes et transversales               |
| `cli`          | `config`, `environment`, `data`, `models`, `training`, `benchmarking` | -                  | Interface utilisateur en ligne de commande              |
| `benchmarking` | `config`, `models`, `training`, `utils`        | `cli`                                      | Comparaison systématique des performances des modèles    |

### Workflow typique

Le flux de traitement typique dans ForestGaps suit ces étapes :

1. **Configuration** : Chargement et validation des paramètres de configuration
2. **Détection d'environnement** : Identification et configuration de l'environnement (Colab/local)
3. **Prétraitement des données** : 
   - Lecture des fichiers DSM/CHM
   - Prétraitement (filtrage, alignement, normalisation)
   - Création de tuiles et masques
   - Construction de datasets PyTorch
4. **Modélisation** : 
   - Sélection d'une architecture de réseau
   - Configuration du modèle
5. **Entraînement** : 
   - Initialisation du trainer
   - Entraînement du modèle avec métriques
   - Suivi avec TensorBoard
   - Sauvegarde des checkpoints
6. **Évaluation** : 
   - Évaluation sur données de test
   - Calcul des métriques
   - Génération de rapports
7. **Inférence** : 
   - Application des modèles entraînés à de nouvelles données
   - Post-traitement des prédictions
   - Sauvegarde des résultats
8. **Benchmarking** : 
   - Comparaison systématique des performances entre modèles
   - Analyse statistique des résultats

## 2. Guide d'installation et prérequis

### Installation standard

```bash
# Installation depuis GitHub
pip install git+https://github.com/arthur048/forestgaps.git

# Installation en mode développement (après clone)
git clone https://github.com/arthur048/forestgaps.git
cd forestgaps
pip install -e .
```

### Installation dans Google Colab

```python
# Méthode recommandée : script d'installation optimisé
!wget -O colab_install.py https://raw.githubusercontent.com/arthur048/forestgaps/main/colab_install.py
%run colab_install.py

# Redémarrer le runtime puis :
from forestgaps.environment import setup_environment
env = setup_environment()
```

### Dépendances

#### Dépendances principales

- Python 3.8+
- PyTorch >= 1.8.0
- TorchVision >= 0.9.0
- NumPy >= 1.19.0
- Rasterio >= 1.2.0

#### Dépendances par catégorie

1. **Deep Learning** :
   - PyTorch (>= 1.8.0)
   - TorchVision (>= 0.9.0)
   - TensorBoard (>= 2.8.0)

2. **Traitement de données géospatiales** :
   - Rasterio (>= 1.2.0)
   - GeoPandas (>= 0.10.0)
   - NumPy (>= 1.19.0)
   - Pandas (>= 1.3.0)

3. **Visualisation** :
   - Matplotlib (>= 3.3.0)
   - Scikit-image (>= 0.18.0)

4. **Utilitaires** :
   - PyYAML (>= 6.0)
   - Pydantic (>= 1.8.0)
   - tqdm (>= 4.60.0)
   - Tabulate (>= 0.8.0)
   - Markdown (>= 3.3.0)

## 3. Référence API complète

### Module environment

*Module pour la gestion de l'environnement d'exécution (détection Colab/local, configuration GPU)*

**Chemin d'importation :** `from forestgaps.environment import setup_environment, get_device`

**Description :** Ce module gère la détection et la configuration de l'environnement d'exécution, qu'il s'agisse de Google Colab ou d'un environnement local. Il configure automatiquement les ressources disponibles (GPU, RAM) et adapte le comportement du package en conséquence.

**Dépendances spécifiques :** `torch`, `yaml`, `pydantic`

**Exemples d'importation :**
```python
# Import simple
from forestgaps.environment import setup_environment

# Import complet
from forestgaps.environment import setup_environment, get_device, Environment, ColabEnvironment, LocalEnvironment
```

#### Classes

##### `Environment` (Classe abstraite)
**Signature :** `class Environment(ABC)`

**Description :** Classe de base abstraite pour tous les environnements d'exécution.

**Attributs :**
- `name (str)` : Nom de l'environnement
- `platform (str)` : Plateforme d'exécution (OS)
- `device (torch.device)` : Dispositif de calcul (CPU/GPU)
- `is_cuda_available (bool)` : Disponibilité de CUDA
- `cuda_device_count (int)` : Nombre de GPU disponibles
- `cuda_device_name (str, optional)` : Nom du GPU principal si disponible

**Méthodes :**
- `setup() -> None` : Configure l'environnement
- `get_device() -> torch.device` : Renvoie le dispositif optimal
- `is_colab() -> bool` : Vérifie si l'environnement est Google Colab
- `__str__() -> str` : Représentation sous forme de chaîne de caractères

##### `ColabEnvironment`
**Signature :** `class ColabEnvironment(Environment)`

**Description :** Environnement spécifique pour Google Colab avec gestion des ressources cloud.

**Attributs spécifiques :**
- `tpu_address (str, optional)` : Adresse TPU si disponible
- `is_tpu_available (bool)` : Disponibilité de TPU
- `ram_size_gb (float)` : Taille de la RAM disponible en Go

**Méthodes spécifiques :**
- `mount_drive() -> str` : Monte Google Drive et renvoie le chemin
- `setup_tpu() -> bool` : Configure TPU si disponible
- `get_free_memory() -> float` : Renvoie la mémoire libre en Go

##### `LocalEnvironment`
**Signature :** `class LocalEnvironment(Environment)`

**Description :** Environnement pour exécution locale avec détection des ressources système.

**Attributs spécifiques :**
- `cpu_count (int)` : Nombre de cœurs CPU disponibles
- `ram_size_gb (float)` : Taille de la RAM système en Go
- `gpu_memory_gb (float, optional)` : Mémoire GPU disponible en Go si applicable

**Méthodes spécifiques :**
- `setup_gpu() -> bool` : Configure le GPU si disponible
- `get_cpu_usage() -> float` : Renvoie l'utilisation CPU en pourcentage
- `get_memory_usage() -> float` : Renvoie l'utilisation mémoire en pourcentage

#### Fonctions

##### `detect_environment`
**Signature :** `detect_environment() -> Environment`

**Description :** Détecte automatiquement l'environnement d'exécution (Colab ou local).

**Paramètres :** Aucun

**Retourne :**
- `Environment` : Instance de ColabEnvironment ou LocalEnvironment selon l'environnement détecté

**Exceptions :**
- `EnvironmentError` : Si l'environnement ne peut pas être déterminé

**Exemple d'utilisation :**
```python
env = detect_environment()
print(f"Environnement détecté : {env.name}")
print(f"GPU disponible : {env.is_cuda_available}")
```

##### `setup_environment`
**Signature :** `setup_environment(verbose: bool = True) -> Environment`

**Description :** Configure l'environnement d'exécution et retourne l'instance correspondante.

**Paramètres :**
- `verbose (bool, optional)` : Affiche les informations détaillées pendant la configuration. Par défaut: True

**Retourne :**
- `Environment` : Instance de l'environnement configuré

**Exemple d'utilisation :**
```python
env = setup_environment()
device = env.get_device()
print(f"Utilisation du dispositif : {device}")
```

##### `get_device`
**Signature :** `get_device() -> torch.device`

**Description :** Fonction utilitaire qui retourne le meilleur dispositif disponible (CUDA ou CPU).

**Paramètres :** Aucun

**Retourne :**
- `torch.device` : Dispositif à utiliser pour les calculs

**Exemple d'utilisation :**
```python
device = get_device()
model = MyModel().to(device)
```

### Module config

*Module pour la gestion centralisée de la configuration du projet*

**Chemin d'importation :** `from forestgaps.config import load_default_config, Config`

**Description :** Ce module fournit un système de configuration flexible pour le projet ForestGaps. Il permet de gérer les paramètres pour le traitement des données, les modèles et l'entraînement avec validation des valeurs et formats standards.

**Dépendances spécifiques :** `pydantic`, `pyyaml`

**Exemples d'importation :**
```python
# Import simple pour charger la configuration par défaut
from forestgaps.config import load_default_config

# Import plus avancé pour manipuler des configurations
from forestgaps.config import forestgaps.config, load_config_from_file, create_config_from_dict
```

#### Classes

##### `Config`
**Signature :** `class Config`

**Description :** Classe de configuration de base pour le projet forestgaps. Cette classe fournit les fonctionnalités pour charger et sauvegarder des configurations à partir de fichiers YAML ou JSON.

**Attributs :**
- `BASE_DIR (str)` : Répertoire de base du projet
- `DATA_DIR (str)` : Répertoire des données
- `PROCESSED_DIR (str)` : Répertoire des données prétraitées
- `MODELS_DIR (str)` : Répertoire des modèles
- `CONFIG_DIR (str)` : Répertoire des configurations
- `TILES_DIR (str)` : Répertoire des tuiles d'images
- `TRAIN_TILES_DIR (str)` : Répertoire des tuiles d'entraînement
- `VAL_TILES_DIR (str)` : Répertoire des tuiles de validation
- `TEST_TILES_DIR (str)` : Répertoire des tuiles de test
- *Divers autres paramètres selon les fichiers de configuration chargés*

**Méthodes :**
- `__init__(config_path: Optional[str] = None)` : Initialise une configuration, optionnellement à partir d'un fichier
- `save_config(filepath: Optional[str] = None, format: str = 'yaml') -> str` : Sauvegarde la configuration
- `load_config(filepath: str) -> None` : Charge une configuration depuis un fichier
- `update_from_dict(config_dict: Dict[str, Any]) -> None` : Met à jour la configuration à partir d'un dictionnaire
- `merge_configs(*config_paths: str) -> None` : Fusionne plusieurs fichiers de configuration
- `create_directories(directories: List[str]) -> None` : Crée les répertoires nécessaires

##### `DataSchema`
**Signature :** `class DataSchema(BaseModel)`

**Description :** Schéma de validation pour les configurations de données.

**Attributs :**
- `TILE_SIZE (int)` : Taille des tuiles en pixels
- `OVERLAP (int)` : Chevauchement entre les tuiles en pixels
- `MIN_VALID_PIXELS (int)` : Nombre minimal de pixels valides pour qu'une tuile soit considérée comme valide
- *Autres paramètres de configuration de données*

##### `ModelSchema`
**Signature :** `class ModelSchema(BaseModel)`

**Description :** Schéma de validation pour les configurations des modèles.

**Attributs :**
- `MODEL_TYPE (str)` : Type de modèle (unet, unet_film, deeplabv3plus, etc.)
- `INPUT_CHANNELS (int)` : Nombre de canaux d'entrée
- `OUTPUT_CHANNELS (int)` : Nombre de canaux de sortie
- *Autres paramètres de configuration des modèles*

##### `TrainingSchema`
**Signature :** `class TrainingSchema(BaseModel)`

**Description :** Schéma de validation pour les configurations d'entraînement.

**Attributs :**
- `BATCH_SIZE (int)` : Taille du lot d'entraînement
- `EPOCHS (int)` : Nombre d'époques d'entraînement
- `LEARNING_RATE (float)` : Taux d'apprentissage initial
- `OPTIMIZER (str)` : Optimiseur à utiliser (adam, sgd, etc.)
- *Autres paramètres de configuration d'entraînement*

#### Fonctions

##### `load_default_config`
**Signature :** `load_default_config() -> Config`

**Description :** Charge la configuration par défaut à partir des fichiers YAML dans le dossier 'defaults'.

**Paramètres :** Aucun

**Retourne :**
- `Config` : Instance de Config avec les paramètres par défaut

**Exemple d'utilisation :**
```python
# Charger la configuration par défaut
config = load_default_config()

# Accéder aux paramètres
print(f"Taille des tuiles : {config.TILE_SIZE}")
print(f"Type de modèle : {config.MODEL_TYPE}")
```

##### `load_config_from_file`
**Signature :** `load_config_from_file(config_path: str) -> Config`

**Description :** Charge une configuration à partir d'un fichier YAML ou JSON.

**Paramètres :**
- `config_path (str)` : Chemin vers le fichier de configuration

**Retourne :**
- `Config` : Instance de Config avec les paramètres chargés

**Exemple d'utilisation :**
```python
# Charger une configuration personnalisée
config = load_config_from_file("path/to/my_config.yaml")
```

##### `create_config_from_dict`
**Signature :** `create_config_from_dict(config_dict: Dict[str, Any]) -> Config`

**Description :** Crée une configuration à partir d'un dictionnaire. Le dictionnaire est validé avant création.

**Paramètres :**
- `config_dict (Dict[str, Any])` : Dictionnaire contenant les paramètres de configuration

**Retourne :**
- `Config` : Instance de Config avec les paramètres spécifiés

**Exemple d'utilisation :**
```python
# Créer une configuration à partir d'un dictionnaire
config_dict = {
    "TILE_SIZE": 512,
    "BATCH_SIZE": 32,
    "MODEL_TYPE": "unet_film"
}
config = create_config_from_dict(config_dict)
```

##### `validate_config`
**Signature :** `validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]`

**Description :** Valide un dictionnaire de configuration en utilisant les schémas définis.

**Paramètres :**
- `config_dict (Dict[str, Any])` : Dictionnaire de configuration à valider

**Retourne :**
- `Dict[str, Any]` : Dictionnaire de configuration validé

**Exceptions :**
- `ValidationError` : Si la validation échoue

**Exemple d'utilisation :**
```python
from forestgaps.config import validate_config

# Valider une configuration personnalisée
try:
    validated_config = validate_config(my_config_dict)
    print("Configuration valide !")
except ValidationError as e:
    print(f"Erreur de validation : {e}")
```

### Module data

*Module pour la préparation, le traitement et le chargement des données*

**Chemin d'importation :** `from forestgaps import data`

**Description :** Ce module fournit toutes les fonctionnalités liées à la préparation, au traitement et au chargement des données pour les modèles de détection de trouées forestières. Il contient des sous-modules spécialisés pour chaque étape du pipeline de données.

**Dépendances spécifiques :** `numpy`, `rasterio`, `torch`, `geopandas`, `scikit-image`

**Exemples d'importation :**
```python
# Import des sous-modules spécifiques
from forestgaps.data import preprocessing
from forestgaps.data import datasets
from forestgaps.data import loaders

# Import direct des fonctionnalités fréquemment utilisées
from forestgaps.data.preprocessing import process_raster_pair_robustly
from forestgaps.data.generation import create_gap_masks
from forestgaps.data.loaders import create_data_loaders
```

#### Sous-modules

##### Sous-module `preprocessing`
**Description :** Fonctionnalités pour analyser, aligner et convertir les données raster (DSM/CHM).

**Fonctions principales :**
- `analyze_raster_pair(dsm_path, chm_path, site_name)` : Analyse une paire de rasters DSM/CHM
- `align_rasters(reference_path, target_path, output_dir)` : Aligne deux rasters spatialement
- `process_raster_pair_robustly(dsm_path, chm_path, site_name, config)` : Traitement robuste d'une paire DSM/CHM
- `verify_raster(raster_path, expected_crs=None)` : Vérifie l'intégrité d'un fichier raster
- `compute_raster_statistics(raster_path)` : Calcule les statistiques descriptives d'un raster

**Classes principales :**
- `RasterProcessor` : Classe pour le traitement avancé des rasters

##### Sous-module `generation`
**Description :** Génération de tuiles et de masques à partir des données raster.

**Fonctions principales :**
- `generate_tiles(raster_path, tile_size, overlap, output_dir)` : Découpe un raster en tuiles
- `create_gap_masks(chm_path, thresholds, output_dir, site_name)` : Crée des masques de trouées à différents seuils
- `tile_raster_pair(dsm_path, chm_path, config)` : Découpe une paire DSM/CHM en tuiles correspondantes
- `filter_tiles(tiles_dir, min_valid_pixels)` : Filtre les tuiles selon un critère de validité

**Classes principales :**
- `TileGenerator` : Classe pour la génération avancée de tuiles
- `MaskGenerator` : Classe pour la génération de masques de trouées

##### Sous-module `datasets`
**Description :** Implémentations de datasets PyTorch pour différentes tâches.

**Classes principales :**
- `ForestGapDataset` : Dataset pour la segmentation de trouées forestières
- `ForestRegressionDataset` : Dataset pour la régression (prédiction de CHM)
- `GapTransform` : Transformations pour l'augmentation des données
- `BalancedGapSampler` : Échantillonneur équilibré pour les datasets de trouées

**Fonctions principales :**
- `create_gap_dataset(dsm_files, mask_files, transform_config)` : Crée un dataset de segmentation
- `create_regression_dataset(dsm_files, chm_files, normalize)` : Crée un dataset de régression

##### Sous-module `normalization`
**Description :** Gestion de la normalisation et standardisation des données.

**Fonctions principales :**
- `normalize_data(data, stats, method)` : Normalise des données selon différentes méthodes
- `compute_statistics(data_files)` : Calcule les statistiques pour normalisation
- `apply_normalization(dataset, method, stats=None)` : Applique une normalisation à un dataset

**Classes principales :**
- `Normalizer` : Classe pour appliquer différentes techniques de normalisation
- `StatisticsManager` : Gestion des statistiques de normalisation

##### Sous-module `loaders`
**Description :** Création et optimisation des DataLoaders PyTorch.

**Fonctions principales :**
- `create_data_loaders(config)` : Crée des DataLoaders à partir de la configuration
- `optimize_dataloader_params(dataset, batch_size, max_workers)` : Optimise les paramètres
- `create_balanced_loader(dataset, batch_size, balance_strategy)` : Crée un DataLoader équilibré
- `configure_colab_loader(dataset, memory_limit_gb)` : Configure un loader optimisé pour Colab

**Classes principales :**
- `DataLoaderFactory` : Fabrique pour création de DataLoaders
- `PerformanceMonitor` : Surveillance des performances des DataLoaders

##### Sous-module `storage`
**Description :** Gestion du stockage et de la persistance des données.

**Fonctions principales :**
- `save_compressed_dataset(dataset, output_path)` : Sauvegarde un dataset compressé
- `load_compressed_dataset(input_path)` : Charge un dataset compressé
- `cache_raster_data(raster_paths, cache_dir)` : Met en cache des données raster
- `export_dataset_metadata(dataset, output_path)` : Exporte les métadonnées d'un dataset

**Classes principales :**
- `DatasetArchive` : Classe pour l'archivage et la compression de datasets
- `RasterCache` : Gestion du cache pour les données raster

#### Fonctions principales du module

##### `preprocessing.process_raster_pair_robustly`
**Signature :** `process_raster_pair_robustly(dsm_path: str, chm_path: str, site_name: str, config: Config) -> Dict[str, str]`

**Description :** Traite une paire DSM/CHM de manière robuste avec vérification d'intégrité, alignement et normalisation.

**Paramètres :**
- `dsm_path (str)` : Chemin vers le fichier DSM
- `chm_path (str)` : Chemin vers le fichier CHM
- `site_name (str)` : Nom du site (pour la nomenclature des fichiers)
- `config (Config)` : Configuration avec paramètres de traitement

**Retourne :**
- `Dict[str, str]` : Dictionnaire contenant les chemins vers les fichiers traités

**Exemple d'utilisation :**
```python
from forestgaps.config import load_default_config
from forestgaps.data.preprocessing import process_raster_pair_robustly

config = load_default_config()
result = process_raster_pair_robustly(
    dsm_path="path/to/dsm.tif", 
    chm_path="path/to/chm.tif", 
    site_name="site1", 
    config=config
)

print(f"DSM traité : {result['aligned_dsm']}")
print(f"CHM traité : {result['aligned_chm']}")
```

##### `generation.create_gap_masks`
**Signature :** `create_gap_masks(chm_path: str, thresholds: List[float], output_dir: str, site_name: str) -> Dict[float, str]`

**Description :** Crée des masques binaires pour les trouées forestières à différents seuils de hauteur.

**Paramètres :**
- `chm_path (str)` : Chemin vers le fichier CHM
- `thresholds (List[float])` : Liste des seuils de hauteur pour définir les trouées
- `output_dir (str)` : Répertoire de sortie
- `site_name (str)` : Nom du site (pour la nomenclature des fichiers)

**Retourne :**
- `Dict[float, str]` : Dictionnaire associant les seuils aux chemins des masques générés

**Exemple d'utilisation :**
```python
from forestgaps.data.generation import create_gap_masks

thresholds = [2.0, 5.0, 10.0]
mask_paths = create_gap_masks(
    chm_path="path/to/chm.tif",
    thresholds=thresholds,
    output_dir="path/to/masks",
    site_name="site1"
)

for threshold, path in mask_paths.items():
    print(f"Masque pour seuil {threshold}m : {path}")
```

##### `loaders.create_data_loaders`
**Signature :** `create_data_loaders(config: Config, split_ratios: Optional[Dict[str, float]] = None) -> Dict[str, DataLoader]`

**Description :** Crée des DataLoaders pour l'entraînement, la validation et le test à partir de la configuration.

**Paramètres :**
- `config (Config)` : Configuration contenant les paramètres des DataLoaders
- `split_ratios (Dict[str, float], optional)` : Ratios pour la division des données (train/val/test)

**Retourne :**
- `Dict[str, DataLoader]` : Dictionnaire contenant les DataLoaders pour 'train', 'val' et 'test'

**Exemple d'utilisation :**
```python
from forestgaps.config import load_default_config
from forestgaps.data.loaders import create_data_loaders

config = load_default_config()
data_loaders = create_data_loaders(
    config, 
    split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}
)

train_loader = data_loaders['train']
val_loader = data_loaders['val']
test_loader = data_loaders['test']
```

### Module models

*Module pour les architectures de réseaux de neurones*

**Chemin d'importation :** `from forestgaps import models`

**Description :** Ce module fournit les implémentations des différentes architectures de réseaux de neurones utilisées pour la détection des trouées forestières par segmentation et régression. Il implémente un système de registre qui permet d'enregistrer et d'instancier des modèles dynamiquement.

**Dépendances spécifiques :** `torch`, `torchvision`, `einops`

**Exemples d'importation :**
```python
# Import simple pour créer un modèle
from forestgaps.models import create_model

# Import plus avancé pour étendre le registre
from forestgaps.models import model_registry, ForestGapModel
```

#### Classes

##### `ForestGapModel` (Classe abstraite)
**Signature :** `class ForestGapModel(nn.Module)`

**Description :** Classe de base abstraite pour tous les modèles de détection de trouées forestières.

**Attributs :**
- `name (str)` : Nom du modèle
- `in_channels (int)` : Nombre de canaux d'entrée
- `out_channels (int)` : Nombre de canaux de sortie

**Méthodes :**
- `forward(x: torch.Tensor) -> torch.Tensor` : Méthode forward pour l'inférence
- `get_parameters_count() -> int` : Renvoie le nombre de paramètres du modèle
- `save(path: str) -> None` : Sauvegarde le modèle
- `load(path: str) -> None` : Charge les poids du modèle

##### `ThresholdConditionedModel`
**Signature :** `class ThresholdConditionedModel(ForestGapModel)`

**Description :** Modèle conditionné par un seuil de hauteur pour la détection de trouées.

**Attributs spécifiques :**
- `threshold (float)` : Seuil de hauteur pour la détection des trouées

**Méthodes spécifiques :**
- `forward(x: torch.Tensor, threshold: float = None) -> torch.Tensor` : Forward avec conditionnement par seuil
- `set_threshold(threshold: float) -> None` : Définit le seuil de hauteur

##### `UNetBaseModel`
**Signature :** `class UNetBaseModel(ForestGapModel)`

**Description :** Classe de base pour les architectures de type U-Net.

**Attributs spécifiques :**
- `encoder_channels (List[int])` : Liste des canaux de l'encodeur
- `decoder_channels (List[int])` : Liste des canaux du décodeur
- `use_batchnorm (bool)` : Utilisation de la normalisation par lots

**Méthodes spécifiques :**
- `encode(x: torch.Tensor) -> List[torch.Tensor]` : Encodage de l'entrée
- `decode(features: List[torch.Tensor]) -> torch.Tensor` : Décodage des caractéristiques

##### `ModelRegistry`
**Signature :** `class ModelRegistry`

**Description :** Registre global des architectures de modèles disponibles. Permet d'enregistrer des architectures et de les créer dynamiquement à partir de leur nom.

**Méthodes :**
- `register(name: str) -> Callable` : Décorateur pour enregistrer une classe de modèle
- `create(model_type: str, **kwargs) -> nn.Module` : Crée une instance d'un modèle
- `list_models() -> List[str]` : Liste tous les modèles disponibles
- `get_model_class(model_type: str) -> Type[nn.Module]` : Récupère la classe d'un modèle

#### Sous-modules

##### Sous-module `unet`
**Description :** Implémentations de différentes variantes de l'architecture U-Net.

**Classes principales :**
- `BasicUNet` : Implémentation standard de U-Net
- `AttentionUNet` : U-Net avec mécanismes d'attention
- `FiLMUNet` : U-Net avec Feature-wise Linear Modulation
- `ResidualUNet` : U-Net avec connexions résiduelles

##### Sous-module `deeplabv3`
**Description :** Implémentations de l'architecture DeepLabV3+.

**Classes principales :**
- `DeepLabV3Plus` : Implémentation standard de DeepLabV3+
- `AdvancedDeepLab` : DeepLabV3+ avec améliorations

##### Sous-module `unet_regression`
**Description :** Adaptations de U-Net pour la régression de hauteur.

**Classes principales :**
- `UNetRegressor` : U-Net adapté pour la régression
- `EnsembleRegressor` : Ensemble de modèles de régression

##### Sous-module `blocks`
**Description :** Blocs réutilisables pour la construction de modèles.

**Classes principales :**
- `AttentionBlock` : Bloc d'attention
- `FiLMBlock` : Bloc de modulation FiLM
- `ResidualBlock` : Bloc résiduel
- `UpsamplingBlock` : Bloc d'upsampling

#### Fonctions

##### `create_model`
**Signature :** `create_model(model_name: str, **kwargs) -> ForestGapModel`

**Description :** Crée une instance d'un modèle à partir de son nom.

**Paramètres :**
- `model_name (str)` : Nom du modèle à créer
- `**kwargs` : Arguments spécifiques au modèle

**Retourne :**
- `ForestGapModel` : Instance du modèle créé

**Exceptions :**
- `ValueError` : Si le modèle n'est pas trouvé dans le registre

**Exemple d'utilisation :**
```python
from forestgaps.models import create_model

# Créer un modèle U-Net standard
model = create_model("unet", in_channels=3, out_channels=1)

# Créer un modèle U-Net avec FiLM
model_film = create_model("unet_film", in_channels=3, out_channels=1)
```

##### `list_available_models`
**Signature :** `list_available_models() -> Dict[str, Type[ForestGapModel]]`

**Description :** Liste tous les modèles disponibles dans le registre.

**Paramètres :** Aucun

**Retourne :**
- `Dict[str, Type[ForestGapModel]]` : Dictionnaire des noms de modèles et leurs classes

**Exemple d'utilisation :**
```python
from forestgaps.models import list_available_models

# Obtenir la liste des modèles disponibles
models = list_available_models()
print(f"Modèles disponibles : {list(models.keys())}")
```

##### `get_model_from_config`
**Signature :** `get_model_from_config(config: dict) -> ForestGapModel`

**Description :** Crée un modèle à partir d'une configuration.

**Paramètres :**
- `config (dict)` : Dictionnaire de configuration du modèle

**Retourne :**
- `ForestGapModel` : Instance du modèle créé

**Exemple d'utilisation :**
```python
from forestgaps.config import load_default_config
from forestgaps.models import get_model_from_config

# Charger la configuration
config = load_default_config()

# Créer un modèle à partir de la configuration
model = get_model_from_config(config)
```

### Module training

*Module pour l'entraînement des modèles de segmentation et de régression*

**Chemin d'importation :** `from forestgaps import training`

**Description :** Ce module fournit des classes et des fonctions pour l'entraînement des modèles de segmentation et de régression dans le cadre du projet ForestGaps. Il implémente un système modulaire et extensible pour entraîner, évaluer et tester des modèles de détection de trouées forestières.

**Dépendances spécifiques :** `torch`, `tensorboard`, `tqdm`

**Exemples d'importation :**
```python
# Import simple de la classe Trainer
from forestgaps.training import Trainer

# Import plus avancé pour les métriques et les pertes
from forestgaps.training import Trainer, SegmentationMetrics, CombinedFocalDiceLoss
```

#### Classes

##### `Trainer`
**Signature :** `class Trainer`

**Description :** Classe principale qui encapsule toute la logique d'entraînement, de validation et de test des modèles. Elle gère également les points de contrôle, les métriques et l'optimisation.

**Attributs :**
- `model (nn.Module)` : Modèle à entraîner
- `config (Config)` : Configuration d'entraînement
- `train_loader (DataLoader)` : DataLoader pour les données d'entraînement
- `val_loader (DataLoader)` : DataLoader pour les données de validation
- `test_loader (DataLoader, optional)` : DataLoader pour les données de test
- `optimizer (Optimizer)` : Optimiseur pour l'entraînement
- `scheduler (LRScheduler, optional)` : Scheduler de taux d'apprentissage
- `loss_fn (callable)` : Fonction de perte
- `metrics (dict)` : Dictionnaire des métriques à suivre
- `device (torch.device)` : Dispositif d'entraînement (CPU/GPU)
- `callbacks (list)` : Liste des callbacks pour personnaliser l'entraînement

**Méthodes :**
- `train(epochs=None) -> dict` : Entraîne le modèle pour un nombre donné d'époques
- `validate() -> dict` : Évalue le modèle sur les données de validation
- `test() -> dict` : Évalue le modèle sur les données de test
- `predict(inputs, thresholds=None) -> torch.Tensor` : Effectue des prédictions avec le modèle
- `save_checkpoint(path) -> None` : Sauvegarde un point de contrôle du modèle
- `load_checkpoint(path) -> None` : Charge un point de contrôle du modèle
- `resume_training(checkpoint_path) -> dict` : Reprend l'entraînement à partir d'un point de contrôle

##### `SegmentationMetrics`
**Signature :** `class SegmentationMetrics`

**Description :** Classe pour calculer et suivre les métriques de segmentation.

**Attributs :**
- `device (torch.device)` : Dispositif pour les calculs
- `metrics (dict)` : Dictionnaire des métriques calculées

**Méthodes :**
- `update(predictions, targets) -> None` : Met à jour les métriques avec de nouvelles prédictions
- `compute() -> dict` : Calcule et renvoie toutes les métriques
- `reset() -> None` : Réinitialise toutes les métriques
- `get_metric(name) -> float` : Récupère la valeur d'une métrique spécifique

##### `CombinedFocalDiceLoss`
**Signature :** `class CombinedFocalDiceLoss(nn.Module)`

**Description :** Fonction de perte combinant la perte Focal et la perte Dice, adaptée à la segmentation des trouées forestières.

**Attributs :**
- `alpha (float)` : Poids de la perte Focal
- `gamma (float)` : Paramètre gamma de la perte Focal
- `beta (float)` : Poids de la perte Dice
- `smooth (float)` : Facteur de lissage pour la perte Dice

**Méthodes :**
- `forward(predictions, targets) -> torch.Tensor` : Calcule la perte combinée

##### `Callback`
**Signature :** `class Callback`

**Description :** Classe de base pour tous les callbacks d'entraînement.

**Méthodes :**
- `on_train_begin(logs=None) -> None` : Appelé au début de l'entraînement
- `on_train_end(logs=None) -> None` : Appelé à la fin de l'entraînement
- `on_epoch_begin(epoch, logs=None) -> None` : Appelé au début de chaque époque
- `on_epoch_end(epoch, logs=None) -> None` : Appelé à la fin de chaque époque
- `on_batch_begin(batch, logs=None) -> None` : Appelé au début de chaque batch
- `on_batch_end(batch, logs=None) -> None` : Appelé à la fin de chaque batch

#### Sous-modules

##### Sous-module `metrics`
**Description :** Métriques pour évaluer les performances des modèles.

**Classes principales :**
- `SegmentationMetrics` : Métriques pour la segmentation (IoU, Dice, précision, rappel)
- `ThresholdMetrics` : Métriques spécifiques par seuil de hauteur
- `RegressionMetrics` : Métriques pour les modèles de régression (MAE, MSE, RMSE)

##### Sous-module `loss`
**Description :** Fonctions de perte pour l'entraînement des modèles.

**Classes principales :**
- `CombinedFocalDiceLoss` : Perte combinant Focal et Dice
- `WeightedBCEWithLogitsLoss` : BCE pondérée pour les déséquilibres de classes
- `HeightThresholdLoss` : Perte adaptée aux seuils de hauteur

**Fonctions principales :**
- `create_loss_function(config)` : Crée une fonction de perte à partir de la configuration

##### Sous-module `callbacks`
**Description :** Système de callbacks pour personnaliser l'entraînement.

**Classes principales :**
- `LoggingCallback` : Journalisation des métriques
- `TensorBoardCallback` : Visualisation avec TensorBoard
- `CheckpointingCallback` : Sauvegarde des points de contrôle
- `EarlyStoppingCallback` : Arrêt anticipé de l'entraînement
- `VisualizationCallback` : Visualisation des prédictions pendant l'entraînement

##### Sous-module `optimization`
**Description :** Techniques d'optimisation pour l'entraînement.

**Classes principales :**
- `CompositeRegularization` : Combinaison de techniques de régularisation
- `AdaptiveNormalization` : Normalisation adaptative selon la taille du batch

**Fonctions principales :**
- `create_optimizer(config, model_parameters)` : Crée un optimiseur à partir de la configuration
- `create_scheduler(optimizer, config, steps_per_epoch)` : Crée un scheduler de learning rate

#### Fonctions principales du module

##### `train_model`
**Signature :** `train_model(model: nn.Module, config: Config, train_loader: DataLoader, val_loader: DataLoader, test_loader: Optional[DataLoader] = None) -> Tuple[nn.Module, dict]`

**Description :** Fonction utilitaire pour entraîner un modèle avec une configuration donnée.

**Paramètres :**
- `model (nn.Module)` : Modèle à entraîner
- `config (Config)` : Configuration d'entraînement
- `train_loader (DataLoader)` : DataLoader pour les données d'entraînement
- `val_loader (DataLoader)` : DataLoader pour les données de validation
- `test_loader (DataLoader, optional)` : DataLoader pour les données de test

**Retourne :**
- `Tuple[nn.Module, dict]` : Modèle entraîné et historique d'entraînement

**Exemple d'utilisation :**
```python
from forestgaps.training import train_model
from forestgaps.models import create_model
from forestgaps.config import load_default_config
from forestgaps.data.loaders import create_data_loaders

# Charger la configuration
config = load_default_config()

# Créer les dataloaders
data_loaders = create_data_loaders(config)

# Créer un modèle
model = create_model("unet", in_channels=3, out_channels=1)

# Entraîner le modèle
trained_model, history = train_model(
    model=model,
    config=config,
    train_loader=data_loaders['train'],
    val_loader=data_loaders['val'],
    test_loader=data_loaders['test']
)
```

### Module evaluation

*Module pour l'évaluation externe des modèles entraînés*

**Chemin d'importation :** `from forestgaps import evaluation`

**Description :** Ce module fournit les fonctionnalités nécessaires pour évaluer les modèles entraînés sur des paires DSM/CHM indépendantes. Il permet de calculer des métriques détaillées de performance, de générer des rapports d'évaluation complets et de comparer différents modèles.

**Dépendances spécifiques :** `torch`, `numpy`, `matplotlib`, `pandas`, `tabulate`

**Exemples d'importation :**
```python
# Import simple des fonctions d'évaluation
from forestgaps.evaluation import evaluate_model, compare_models

# Import plus avancé pour les classes d'évaluation
from forestgaps.evaluation import ExternalEvaluator, EvaluationResult, EvaluationConfig
```

#### Classes

##### `EvaluationConfig`
**Signature :** `class EvaluationConfig`

**Description :** Configuration pour le processus d'évaluation.

**Attributs :**
- `thresholds (List[float])` : Seuils de hauteur pour l'évaluation
- `metrics (List[str])` : Liste des métriques à calculer
- `visualization_options (Dict[str, Any])` : Options pour la visualisation
- `report_options (Dict[str, Any])` : Options pour la génération de rapports
- `comparison_options (Dict[str, Any])` : Options pour la comparaison de modèles

**Méthodes :**
- `from_dict(config_dict: Dict[str, Any]) -> EvaluationConfig` : Crée une configuration à partir d'un dictionnaire
- `to_dict() -> Dict[str, Any]` : Convertit la configuration en dictionnaire
- `validate() -> bool` : Valide la configuration

##### `EvaluationResult`
**Signature :** `class EvaluationResult`

**Description :** Encapsule les résultats d'une évaluation.

**Attributs :**
- `metrics (Dict[str, Any])` : Métriques calculées (précision, rappel, F1, IoU, etc.)
- `threshold_metrics (Dict[float, Dict[str, Any]])` : Métriques par seuil de hauteur
- `confusion_matrices (Dict[float, np.ndarray])` : Matrices de confusion par seuil
- `predictions (np.ndarray)` : Prédictions du modèle
- `ground_truth (np.ndarray)` : Vérités terrain
- `model_name (str)` : Nom du modèle évalué
- `dataset_info (Dict[str, Any])` : Informations sur le jeu de données

**Méthodes :**
- `save(output_dir: str) -> str` : Sauvegarde les résultats
- `load(input_dir: str) -> EvaluationResult` : Charge des résultats sauvegardés
- `generate_report(output_path: str) -> str` : Génère un rapport détaillé
- `visualize(output_dir: Optional[str] = None) -> Dict[str, str]` : Visualise les résultats
- `get_metric(name: str, threshold: Optional[float] = None) -> float` : Récupère la valeur d'une métrique

##### `ExternalEvaluator`
**Signature :** `class ExternalEvaluator`

**Description :** Gère le processus d'évaluation complet.

**Attributs :**
- `model (nn.Module)` : Modèle à évaluer
- `config (EvaluationConfig)` : Configuration d'évaluation
- `device (torch.device)` : Dispositif pour l'évaluation
- `thresholds (List[float])` : Seuils de hauteur pour l'évaluation

**Méthodes :**
- `evaluate_pair(dsm_path: str, chm_path: str) -> EvaluationResult` : Évalue sur une paire DSM/CHM
- `evaluate_site(site_dsm_dir: str, site_chm_dir: str) -> EvaluationResult` : Évalue sur un site
- `evaluate_sites(sites_config: Dict[str, Dict[str, str]]) -> Dict[str, EvaluationResult]` : Évalue sur plusieurs sites
- `compare_with(other_model_path: str, dsm_path: str, chm_path: str) -> Dict[str, EvaluationResult]` : Compare avec un autre modèle

#### Fonctions

##### `evaluate_model`
**Signature :** `evaluate_model(model_path: str, dsm_path: str, chm_path: str, output_dir: Optional[str] = None, thresholds: Optional[List[float]] = None, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None, visualize: bool = False) -> EvaluationResult`

**Description :** Évalue un modèle sur une paire DSM/CHM.

**Paramètres :**
- `model_path (str)` : Chemin vers le modèle entraîné
- `dsm_path (str)` : Chemin vers le fichier DSM
- `chm_path (str)` : Chemin vers le fichier CHM
- `output_dir (str, optional)` : Répertoire de sortie pour les résultats
- `thresholds (List[float], optional)` : Seuils de hauteur pour l'évaluation
- `config (Dict[str, Any], optional)` : Configuration d'évaluation
- `device (str, optional)` : Dispositif pour l'évaluation ('cuda' ou 'cpu')
- `visualize (bool)` : Génère des visualisations des résultats

**Retourne :**
- `EvaluationResult` : Résultats de l'évaluation

**Exemple d'utilisation :**
```python
from forestgaps.evaluation import evaluate_model

# Évaluer un modèle sur une paire DSM/CHM
result = evaluate_model(
    model_path="path/to/model.pt",
    dsm_path="path/to/dsm.tif",
    chm_path="path/to/chm.tif",
    output_dir="path/to/output",
    thresholds=[2.0, 5.0, 10.0, 15.0],
    visualize=True
)

# Accéder aux métriques
print(f"IoU global : {result.metrics['iou']}")
print(f"F1-Score à 5m : {result.get_metric('f1', threshold=5.0)}")
```

##### `evaluate_site`
**Signature :** `evaluate_site(model_path: str, site_dsm_dir: str, site_chm_dir: str, output_dir: str, thresholds: Optional[List[float]] = None, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None, batch_size: int = 1, num_workers: int = 4, visualize: bool = False) -> EvaluationResult`

**Description :** Évalue un modèle sur toutes les paires DSM/CHM d'un site.

**Paramètres :**
- `model_path (str)` : Chemin vers le modèle entraîné
- `site_dsm_dir (str)` : Répertoire contenant les fichiers DSM du site
- `site_chm_dir (str)` : Répertoire contenant les fichiers CHM du site
- `output_dir (str)` : Répertoire de sortie pour les résultats
- `thresholds (List[float], optional)` : Seuils de hauteur pour l'évaluation
- `config (Dict[str, Any], optional)` : Configuration d'évaluation
- `device (str, optional)` : Dispositif pour l'évaluation ('cuda' ou 'cpu')
- `batch_size (int)` : Taille du lot pour l'inférence
- `num_workers (int)` : Nombre de workers pour le chargement des données
- `visualize (bool)` : Génère des visualisations des résultats

**Retourne :**
- `EvaluationResult` : Résultats agrégés de l'évaluation sur le site

**Exemple d'utilisation :**
```python
from forestgaps.evaluation import evaluate_site

# Évaluer un modèle sur un site complet
result = evaluate_site(
    model_path="path/to/model.pt",
    site_dsm_dir="path/to/site1/dsm",
    site_chm_dir="path/to/site1/chm",
    output_dir="path/to/output/site1",
    thresholds=[2.0, 5.0, 10.0],
    batch_size=4,
    visualize=True
)

# Générer un rapport
result.generate_report("path/to/output/site1/report.html")
```

##### `evaluate_model_on_sites`
**Signature :** `evaluate_model_on_sites(model_path: str, sites_config: Dict[str, Dict[str, str]], output_dir: str, thresholds: Optional[List[float]] = None, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None, batch_size: int = 1, num_workers: int = 4, visualize: bool = False, aggregate_results: bool = True) -> Dict[str, EvaluationResult]`

**Description :** Évalue un modèle sur plusieurs sites et agrège les résultats.

**Paramètres :**
- `model_path (str)` : Chemin vers le modèle entraîné
- `sites_config (Dict[str, Dict[str, str]])` : Configuration des sites à évaluer
- `output_dir (str)` : Répertoire de sortie pour les résultats
- `thresholds (List[float], optional)` : Seuils de hauteur pour l'évaluation
- `config (Dict[str, Any], optional)` : Configuration d'évaluation
- `device (str, optional)` : Dispositif pour l'évaluation ('cuda' ou 'cpu')
- `batch_size (int)` : Taille du lot pour l'inférence
- `num_workers (int)` : Nombre de workers pour le chargement des données
- `visualize (bool)` : Génère des visualisations des résultats
- `aggregate_results (bool)` : Agrège les résultats de tous les sites

**Retourne :**
- `Dict[str, EvaluationResult]` : Dictionnaire des résultats par site

**Exemple d'utilisation :**
```python
from forestgaps.evaluation import evaluate_model_on_sites

# Configuration des sites
sites_config = {
    "site1": {
        "dsm_dir": "path/to/site1/dsm",
        "chm_dir": "path/to/site1/chm"
    },
    "site2": {
        "dsm_dir": "path/to/site2/dsm",
        "chm_dir": "path/to/site2/chm"
    }
}

# Évaluer un modèle sur plusieurs sites
results = evaluate_model_on_sites(
    model_path="path/to/model.pt",
    sites_config=sites_config,
    output_dir="path/to/output",
    thresholds=[2.0, 5.0, 10.0],
    aggregate_results=True
)

# Accéder aux résultats par site
site1_result = results["site1"]
site2_result = results["site2"]
aggregated_result = results["aggregated"]
```

##### `compare_models`
**Signature :** `compare_models(model_paths: Dict[str, str], dsm_path: str, chm_path: str, output_dir: str, thresholds: Optional[List[float]] = None, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None, visualize: bool = True) -> Dict[str, EvaluationResult]`

**Description :** Compare les performances de différents modèles sur une même paire DSM/CHM.

**Paramètres :**
- `model_paths (Dict[str, str])` : Dictionnaire des noms de modèles et leurs chemins
- `dsm_path (str)` : Chemin vers le fichier DSM
- `chm_path (str)` : Chemin vers le fichier CHM
- `output_dir (str)` : Répertoire de sortie pour les résultats
- `thresholds (List[float], optional)` : Seuils de hauteur pour l'évaluation
- `config (Dict[str, Any], optional)` : Configuration d'évaluation
- `device (str, optional)` : Dispositif pour l'évaluation ('cuda' ou 'cpu')
- `visualize (bool)` : Génère des visualisations comparatives

**Retourne :**
- `Dict[str, EvaluationResult]` : Dictionnaire des résultats par modèle

**Exemple d'utilisation :**
```python
from forestgaps.evaluation import compare_models

# Dictionnaire des modèles à comparer
models = {
    "unet": "path/to/unet.pt",
    "unet_film": "path/to/unet_film.pt",
    "deeplabv3plus": "path/to/deeplabv3plus.pt"
}

# Comparer les modèles
results = compare_models(
    model_paths=models,
    dsm_path="path/to/dsm.tif",
    chm_path="path/to/chm.tif",
    output_dir="path/to/comparison",
    thresholds=[2.0, 5.0, 10.0]
)

# Accéder aux résultats par modèle
unet_result = results["unet"]
unet_film_result = results["unet_film"]
deeplab_result = results["deeplabv3plus"]
``` 

### Module inference

*Module pour l'application des modèles entraînés à de nouvelles données*

**Chemin d'importation :** `from forestgaps import inference`

**Description :** Ce module fournit les fonctionnalités nécessaires pour appliquer les modèles entraînés à de nouvelles données DSM. Il permet d'effectuer des prédictions sur des données non vues pendant l'entraînement, de visualiser les résultats et de sauvegarder les prédictions.

**Dépendances spécifiques :** `torch`, `rasterio`, `numpy`, `matplotlib`

**Exemples d'importation :**
```python
# Import simple des fonctions d'inférence
from forestgaps.inference import run_inference, run_batch_inference

# Import plus avancé pour les classes d'inférence
from forestgaps.inference import InferenceManager, InferenceResult, InferenceConfig
```

#### Classes

##### `InferenceConfig`
**Signature :** `class InferenceConfig`

**Description :** Configuration pour le processus d'inférence.

**Attributs :**
- `tile_size (int)` : Taille des tuiles pour le traitement
- `overlap (int)` : Chevauchement entre les tuiles
- `batch_size (int)` : Taille des lots pour l'inférence
- `threshold (float)` : Seuil de probabilité pour la binarisation
- `post_processing (Dict[str, Any])` : Options de post-traitement
- `visualization_options (Dict[str, Any])` : Options pour la visualisation

**Méthodes :**
- `from_dict(config_dict: Dict[str, Any]) -> InferenceConfig` : Crée une configuration à partir d'un dictionnaire
- `to_dict() -> Dict[str, Any]` : Convertit la configuration en dictionnaire
- `validate() -> bool` : Valide la configuration

##### `InferenceResult`
**Signature :** `class InferenceResult`

**Description :** Encapsule les résultats d'une opération d'inférence.

**Attributs :**
- `predictions (np.ndarray)` : Prédictions brutes du modèle
- `binary_predictions (np.ndarray)` : Prédictions binarisées
- `post_processed (np.ndarray)` : Prédictions après post-traitement
- `metadata (Dict[str, Any])` : Métadonnées géospatiales
- `model_name (str)` : Nom du modèle utilisé
- `threshold (float)` : Seuil utilisé pour la binarisation
- `input_path (str)` : Chemin du fichier d'entrée
- `output_path (str, optional)` : Chemin du fichier de sortie

**Méthodes :**
- `save(output_path: str) -> str` : Sauvegarde les prédictions
- `visualize(output_path: Optional[str] = None) -> str` : Visualise les prédictions
- `get_statistics() -> Dict[str, float]` : Calcule des statistiques sur les prédictions
- `apply_threshold(threshold: float) -> np.ndarray` : Applique un nouveau seuil

##### `InferenceManager`
**Signature :** `class InferenceManager`

**Description :** Gère le processus d'inférence complet.

**Attributs :**
- `model (nn.Module)` : Modèle chargé
- `config (InferenceConfig)` : Configuration d'inférence
- `device (torch.device)` : Dispositif pour l'inférence
- `threshold (float)` : Seuil de probabilité

**Méthodes :**
- `predict(dsm_path: str) -> InferenceResult` : Effectue des prédictions sur un fichier DSM
- `batch_predict(dsm_paths: List[str]) -> Dict[str, InferenceResult]` : Prédit sur plusieurs fichiers
- `process_tile(tile: np.ndarray) -> np.ndarray` : Traite une tuile individuelle
- `post_process(predictions: np.ndarray) -> np.ndarray` : Applique le post-traitement

#### Fonctions

##### `run_inference`
**Signature :** `run_inference(model_path: str, dsm_path: str, output_path: Optional[str] = None, threshold: float = 5.0, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None, visualize: bool = False) -> InferenceResult`

**Description :** Exécute l'inférence sur un fichier DSM.

**Paramètres :**
- `model_path (str)` : Chemin vers le modèle entraîné
- `dsm_path (str)` : Chemin vers le fichier DSM
- `output_path (str, optional)` : Chemin pour sauvegarder les prédictions
- `threshold (float)` : Seuil de hauteur pour la binarisation
- `config (Dict[str, Any], optional)` : Configuration d'inférence
- `device (str, optional)` : Dispositif pour l'inférence ('cuda' ou 'cpu')
- `visualize (bool)` : Génère des visualisations des prédictions

**Retourne :**
- `InferenceResult` : Résultats de l'inférence

**Exemple d'utilisation :**
```python
from forestgaps.inference import run_inference

# Exécuter l'inférence sur un fichier DSM
result = run_inference(
    model_path="path/to/model.pt",
    dsm_path="path/to/dsm.tif",
    output_path="path/to/output.tif",
    threshold=5.0,
    visualize=True
)

# Accéder aux prédictions
predictions = result.predictions

# Visualiser les résultats
result.visualize()

# Sauvegarder les résultats
result.save("path/to/save")
```

##### `run_batch_inference`
**Signature :** `run_batch_inference(model_path: str, dsm_paths: List[str], output_dir: str, threshold: float = 5.0, config: Optional[Dict[str, Any]] = None, device: Optional[str] = None, batch_size: int = 1, num_workers: int = 4, visualize: bool = False) -> Dict[str, InferenceResult]`

**Description :** Exécute l'inférence sur plusieurs fichiers DSM.

**Paramètres :**
- `model_path (str)` : Chemin vers le modèle entraîné
- `dsm_paths (List[str])` : Liste des chemins vers les fichiers DSM
- `output_dir (str)` : Répertoire pour sauvegarder les prédictions
- `threshold (float)` : Seuil de hauteur pour la binarisation
- `config (Dict[str, Any], optional)` : Configuration d'inférence
- `device (str, optional)` : Dispositif pour l'inférence ('cuda' ou 'cpu')
- `batch_size (int)` : Nombre de fichiers à traiter en parallèle
- `num_workers (int)` : Nombre de workers pour le chargement des données
- `visualize (bool)` : Génère des visualisations des prédictions

**Retourne :**
- `Dict[str, InferenceResult]` : Dictionnaire des résultats par fichier

**Exemple d'utilisation :**
```python
from forestgaps.inference import run_batch_inference

# Liste des fichiers DSM à traiter
dsm_paths = [
    "path/to/dsm1.tif",
    "path/to/dsm2.tif",
    "path/to/dsm3.tif"
]

# Exécuter l'inférence par lots
results = run_batch_inference(
    model_path="path/to/model.pt",
    dsm_paths=dsm_paths,
    output_dir="path/to/outputs",
    threshold=5.0,
    batch_size=4,
    num_workers=2,
    visualize=True
)

# Accéder aux résultats individuels
result1 = results["dsm1.tif"]
result2 = results["dsm2.tif"]
result3 = results["dsm3.tif"]
```

#### Fonctions utilitaires

##### `load_raster`
**Signature :** `load_raster(raster_path: str) -> Tuple[np.ndarray, Dict[str, Any]]`

**Description :** Charge un fichier raster et ses métadonnées.

**Paramètres :**
- `raster_path (str)` : Chemin vers le fichier raster

**Retourne :**
- `Tuple[np.ndarray, Dict[str, Any]]` : Données raster et métadonnées

##### `save_raster`
**Signature :** `save_raster(data: np.ndarray, output_path: str, metadata: Dict[str, Any]) -> str`

**Description :** Sauvegarde des données sous forme de fichier raster.

**Paramètres :**
- `data (np.ndarray)` : Données à sauvegarder
- `output_path (str)` : Chemin de sortie
- `metadata (Dict[str, Any])` : Métadonnées géospatiales

**Retourne :**
- `str` : Chemin du fichier sauvegardé

##### `preprocess_dsm`
**Signature :** `preprocess_dsm(dsm: np.ndarray) -> np.ndarray`

**Description :** Prétraite un DSM pour l'inférence.

**Paramètres :**
- `dsm (np.ndarray)` : Données DSM

**Retourne :**
- `np.ndarray` : DSM prétraité

##### `postprocess_prediction`
**Signature :** `postprocess_prediction(prediction: np.ndarray, options: Dict[str, Any] = None) -> np.ndarray`

**Description :** Applique des opérations de post-traitement aux prédictions.

**Paramètres :**
- `prediction (np.ndarray)` : Prédictions brutes
- `options (Dict[str, Any], optional)` : Options de post-traitement

**Retourne :**
- `np.ndarray` : Prédictions post-traitées 

### Module utils

*Contenu à compléter avec l'exploration systématique du code*

### Module cli

*Contenu à compléter avec l'exploration systématique du code*

### Module benchmarking

*Contenu à compléter avec l'exploration systématique du code*

## 4. Guides d'utilisation thématiques

### Prétraitement des données géospatiales

*Contenu à compléter avec des exemples concrets de prétraitement des données géospatiales*

### Entraînement d'un modèle personnalisé

*Contenu à compléter avec des exemples d'entraînement de modèles personnalisés*

### Inférence sur de nouvelles zones

*Contenu à compléter avec des exemples d'inférence sur de nouvelles données*

### Évaluation et comparaison des modèles

*Contenu à compléter avec des exemples d'évaluation et de comparaison de modèles*

## 5. FAQ et dépannage

### Questions fréquentes

*Contenu à compléter avec les questions fréquentes et leurs réponses*

### Problèmes courants

*Contenu à compléter avec les problèmes courants et leurs solutions*

### Limites connues

*Contenu à compléter avec les limites connues du package* 