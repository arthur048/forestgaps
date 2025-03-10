# ForestGaps-DL

![Version](https://img.shields.io/badge/version-0.1.1-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**ForestGaps-DL** est une bibliothèque Python pour la détection et l'analyse automatique des trouées forestières en utilisant le deep learning.

[English version](#english-version)

## Présentation

ForestGaps-DL permet d'analyser des modèles numériques de surface (DSM) et de hauteur de canopée (CHM) pour :
- **Identifier** précisément les trouées dans la canopée forestière
- **Évaluer** leurs caractéristiques géométriques
- **Comparer** les performances de différentes approches d'apprentissage profond

La bibliothèque est compatible à la fois avec un environnement local et Google Colab, offrant une flexibilité maximale selon vos besoins.

## Fonctionnalités principales

- 🔍 **Segmentation** des trouées forestières avec différents modèles (U-Net, DeepLabV3+, etc.)
- 📏 **Estimation** des hauteurs de canopée par régression 
- 🔄 **Prétraitement** des données géospatiales optimisé
- 📈 **Évaluation** exhaustive des modèles avec métriques adaptées
- 🔮 **Inférence** sur de nouvelles zones forestières
- 📊 **Benchmarking** des différentes architectures

## Prérequis

- Python 3.8+
- PyTorch 1.8.0+
- Système d'exploitation : Windows, macOS ou Linux
- GPU compatible CUDA (recommandé mais facultatif)

## Installation

### Installation locale

```bash
# Installation depuis GitHub
pip install git+https://github.com/arthur048/forestgaps-dl.git

# Installation en mode développement (après clone)
git clone https://github.com/arthur048/forestgaps-dl.git
cd forestgaps-dl
pip install -e .
```

### Installation sur Google Colab

```python
# Méthode recommandée : script d'installation optimisé
!wget -O colab_install.py https://raw.githubusercontent.com/arthur048/forestgaps-dl/main/colab_install.py
%run colab_install.py

# Redémarrer le runtime puis :
from forestgaps.environment import setup_environment
env = setup_environment()
```

## Guide de démarrage rapide

### 1. Configuration de l'environnement

```python
# Détection et configuration automatiques de l'environnement
from forestgaps.environment import setup_environment

# Configure l'environnement (Colab ou local)
env = setup_environment()
```

### 2. Prétraitement des données

```python
from forestgaps.config import load_default_config
from forestgaps.data.preprocessing import process_raster_pair_robustly
from forestgaps.data.generation import create_gap_masks

# Charger la configuration
config = load_default_config()

# Prétraiter une paire DSM/CHM
result = process_raster_pair_robustly(
    dsm_path="path/to/dsm.tif", 
    chm_path="path/to/chm.tif", 
    site_name="site1", 
    config=config
)

# Créer des masques de trouées à différents seuils
thresholds = [2.0, 5.0, 10.0]
mask_paths = create_gap_masks(
    chm_path=result["aligned_chm"], 
    thresholds=thresholds,
    output_dir=config.PROCESSED_DIR, 
    site_name="site1"
)
```

### 3. Entraînement d'un modèle

```python
from forestgaps.config import load_default_config
from forestgaps.models import create_model
from forestgaps.data.loaders import create_data_loaders
from forestgaps.training import Trainer

# Charger la configuration
config = load_default_config()

# Créer les dataloaders
data_loaders = create_data_loaders(config)

# Créer un modèle
model = create_model("unet_film")

# Configurer et lancer l'entraînement
trainer = Trainer(
    model=model,
    config=config,
    train_loader=data_loaders['train'],
    val_loader=data_loaders['val'],
    test_loader=data_loaders['test']
)

# Entraîner le modèle
results = trainer.train(epochs=50)
```

### 4. Inférence avec un modèle entraîné

```python
from forestgaps.inference import run_inference

# Exécuter l'inférence sur un nouveau DSM
result = run_inference(
    model_path="path/to/model.pt",
    dsm_path="path/to/new_dsm.tif",
    output_path="path/to/prediction.tif",
    threshold=5.0
)

# Visualiser et sauvegarder les résultats
result.visualize()
result.save("path/to/outputs")
```

### 5. Évaluation de modèles

```python
from forestgaps.evaluation import compare_models

# Comparer différents modèles
models = {
    "unet": "path/to/unet.pt",
    "unet_film": "path/to/unet_film.pt",
    "deeplabv3plus": "path/to/deeplabv3plus.pt"
}

# Évaluer sur une paire DSM/CHM
results = compare_models(
    model_paths=models,
    dsm_path="path/to/dsm.tif",
    chm_path="path/to/chm.tif",
    output_dir="path/to/comparison",
    thresholds=[2.0, 5.0, 10.0]
)
```

## Utilisation Docker

ForestGaps-DL peut être utilisé via Docker pour garantir un environnement cohérent et portable :

```bash
# Construire les images Docker
bash scripts/docker-build.sh

# Exécuter un modèle en inférence
bash scripts/docker-run.sh predict --model /app/models/model.pt --input /app/data/input.tif
```

Plus d'informations dans la [documentation Docker](docker/README.md).

## Documentation détaillée

Pour une documentation complète de chaque module :

- [Module Environment](environment/README.md) - Configuration de l'environnement
- [Module Evaluation](evaluation/README.md) - Évaluation des modèles
- [Module Inference](inference/README.md) - Inférence avec modèles entraînés

Une documentation technique complète pour les LLM est disponible dans [context_llm.md](context_llm.md).

## Projet d'extension

ForestGaps-DL est en développement actif. Voici nos principaux axes de développement :

- Support pour d'autres types de données géospatiales (Sentinel-2, LiDAR, etc.)
- Ajout de nouvelles architectures de modèles plus performantes
- Outils d'analyse spatiale pour les trouées détectées
- Interface graphique pour faciliter l'utilisation

## Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le dépôt
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/ma-fonctionnalite`)
3. Faites vos modifications en respectant les conventions de code
4. Soumettez une pull request

## Auteur

- Arthur - [GitHub](https://github.com/arthur048)

## Licence

Ce projet est sous licence [MIT](LICENSE).

---

# English version

**ForestGaps-DL** is a Python library for automatic detection and analysis of forest gaps using deep learning.

## Overview

ForestGaps-DL analyzes Digital Surface Models (DSM) and Canopy Height Models (CHM) to:
- **Identify** forest canopy gaps with precision
- **Evaluate** their geometric characteristics
- **Compare** different deep learning approaches

The library is compatible with both local environments and Google Colab, offering maximum flexibility based on your needs.

## Key Features

- 🔍 **Segmentation** of forest gaps with various models (U-Net, DeepLabV3+, etc.)
- 📏 **Estimation** of canopy heights through regression
- 🔄 **Preprocessing** of optimized geospatial data
- 📈 **Evaluation** with comprehensive metrics
- 🔮 **Inference** on new forest areas
- 📊 **Benchmarking** of different architectures

## Requirements

- Python 3.8+
- PyTorch 1.8.0+
- Operating System: Windows, macOS, or Linux
- CUDA-compatible GPU (recommended but optional)

## Installation

### Local Installation

```bash
# Installation from GitHub
pip install git+https://github.com/arthur048/forestgaps-dl.git

# Development installation (after cloning)
git clone https://github.com/arthur048/forestgaps-dl.git
cd forestgaps-dl
pip install -e .
```

### Google Colab Installation

```python
# Recommended method: optimized installation script
!wget -O colab_install.py https://raw.githubusercontent.com/arthur048/forestgaps-dl/main/colab_install.py
%run colab_install.py

# Restart the runtime then:
from forestgaps.environment import setup_environment
env = setup_environment()
```

For detailed instructions and examples in English, see the documentation linked above. 