# Module `cli`

## Vue d'ensemble

Le module `cli` fournit des interfaces en ligne de commande pour les différentes fonctionnalités du workflow ForestGaps, permettant d'exécuter les tâches de prétraitement des données et d'entraînement des modèles de manière scriptée.

## Structure

```
cli/
├── __init__.py               # Point d'entrée unifié
├── preprocessing_cli.py      # Interface CLI pour le prétraitement
├── training_cli.py           # Interface CLI pour l'entraînement
├── data.py                   # Commandes pour la manipulation des données
├── train.py                  # Commandes pour l'entraînement
└── evaluate.py               # Commandes pour l'évaluation
```

## Fonctionnalités principales

### Prétraitement (`preprocessing_cli.py`)

- **align**: Aligne les rasters DSM et CHM
- **analyze**: Analyse les rasters et calcule des statistiques
- **generate-tiles**: Génère des tuiles à partir des rasters
- **generate-masks**: Génère des masques de trouées à partir des CHM

### Entraînement (`training_cli.py`)

- **train**: Entraîne un modèle avec différentes architectures
- **evaluate**: Évalue un modèle sur un ensemble de données
- **export**: Exporte un modèle au format ONNX ou TorchScript
- **benchmark**: Évalue les performances d'un modèle

## Utilisation

### Prétraitement

```bash
# Aligner les rasters DSM et CHM
python -m cli.preprocessing_cli align --dsm path/to/dsm.tif --chm path/to/chm.tif

# Analyser un raster
python -m cli.preprocessing_cli analyze --input path/to/raster.tif --save-stats

# Générer des tuiles
python -m cli.preprocessing_cli generate-tiles --dsm path/to/dsm.tif --chm path/to/chm.tif --tile-size 256 --overlap 0.1

# Générer des masques de trouées
python -m cli.preprocessing_cli generate-masks --chm path/to/chm.tif --thresholds 10,15,20,25,30
```

### Entraînement

```bash
# Entraîner un modèle
python -m cli.training_cli train --model-type unet_film_cbam --data-dir path/to/data --epochs 100 --batch-size 32

# Évaluer un modèle
python -m cli.training_cli evaluate --model-path path/to/model.pt --data-dir path/to/test_data --visualize

# Exporter un modèle
python -m cli.training_cli export --model-path path/to/model.pt --format onnx

# Benchmark d'un modèle
python -m cli.training_cli benchmark --model-path path/to/model.pt --batch-sizes 1,4,16,32,64
```

## Intégration avec les autres modules

- **config**: Utilise les configurations validées pour paramétrer les fonctionnalités
- **environment**: S'adapte à l'environnement d'exécution (local ou Colab)
- **data**: Utilise les fonctions de prétraitement et de chargement des données
- **models**: Crée et manipule les modèles via le registre de modèles
- **training**: Utilise les fonctions d'entraînement et d'évaluation
- **utils**: Utilise les fonctions de visualisation, d'entrées/sorties et de gestion des erreurs

## Caractéristiques clés

- **Interface unifiée**: Interface cohérente pour toutes les fonctionnalités du workflow
- **Gestion des erreurs**: Gestion robuste des erreurs avec journalisation détaillée
- **Configuration externalisée**: Utilisation de fichiers de configuration YAML avec validation
- **Extensibilité**: Facilement extensible pour ajouter de nouvelles commandes

## Améliorations futures

- Ajout de commandes pour la visualisation des résultats
- Intégration avec des outils de suivi d'expériences (MLflow, Weights & Biases)
- Ajout de commandes pour le déploiement des modèles
- Création d'une interface utilisateur web pour les utilisateurs non techniques 