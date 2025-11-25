# Résumé du module d'inférence

## Vue d'ensemble

Le module d'inférence de ForestGaps fournit une infrastructure complète pour appliquer des modèles entraînés à de nouvelles données DSM. Il permet d'effectuer des prédictions sur des données non vues pendant l'entraînement, de visualiser les résultats et de sauvegarder les prédictions dans différents formats.

## Architecture

Le module est organisé selon une architecture modulaire avec trois composants principaux :

1. **Classes principales** (`core.py`) :
   - `InferenceConfig` : Configuration pour le processus d'inférence
   - `InferenceResult` : Encapsulation des résultats d'inférence
   - `InferenceManager` : Gestion du processus d'inférence

2. **Utilitaires** (`utils/`) :
   - `geospatial.py` : Manipulation des données géospatiales
   - `image_processing.py` : Traitement d'images pour l'inférence
   - `visualization.py` : Visualisation des prédictions

3. **Interface publique** (`__init__.py`) :
   - Fonctions de haut niveau pour l'inférence
   - Exposition des classes principales

## Fonctionnalités clés

### Gestion des modèles
- Chargement de modèles PyTorch pré-entraînés
- Support pour différentes architectures (U-Net, DeepLabV3+, etc.)
- Gestion des modèles sur CPU ou GPU

### Prétraitement des données
- Normalisation des données DSM
- Découpage en tuiles pour les grandes images
- Gestion des métadonnées géospatiales

### Inférence
- Prédiction sur des fichiers DSM uniques
- Traitement par lots de plusieurs fichiers
- Optimisation des performances (batch processing, parallélisation)

### Post-traitement
- Application de seuils de probabilité
- Filtrage et lissage des prédictions
- Reconstruction des tuiles en images complètes

### Visualisation et sauvegarde
- Génération de visualisations des prédictions
- Sauvegarde des résultats avec métadonnées géospatiales
- Exportation dans différents formats (GeoTIFF, PNG, etc.)

## Interfaces principales

### Fonctions de haut niveau

```python
# Inférence sur un fichier unique
result = run_inference(model_path, dsm_path, output_path, threshold, config, device, visualize)

# Inférence par lots
results = run_batch_inference(model_path, dsm_paths, output_dir, threshold, config, device, batch_size, num_workers, visualize)
```

### Classes principales

```python
# Configuration
config = InferenceConfig(
    tiled_processing=True,
    tile_size=512,
    tile_overlap=64,
    batch_size=8,
    threshold_probability=0.5,
    post_processing={...}
)

# Gestionnaire d'inférence
manager = InferenceManager(model_path, config, device)
result = manager.predict(dsm_path)
results = manager.predict_batch(dsm_paths, output_dir)

# Résultat d'inférence
predictions = result.predictions
metadata = result.metadata
result.save(output_path)
result.visualize()
```

## Intégration avec d'autres modules

Le module d'inférence s'intègre avec :

- **models** : Utilise les modèles entraînés
- **utils** : Utilise les fonctions utilitaires communes
- **evaluation** : Fournit des prédictions pour l'évaluation
- **cli** : Expose ses fonctionnalités via l'interface CLI

## Optimisations

- **Traitement par tuiles** : Gestion efficace des grandes images
- **Parallélisation** : Utilisation de multiprocessing pour le traitement par lots
- **Gestion de la mémoire** : Optimisation de l'utilisation de la mémoire GPU/CPU
- **Mise en cache** : Réutilisation des résultats intermédiaires

## Exemples d'utilisation

```python
from forestgaps.inference import run_inference, run_batch_inference

# Inférence simple
result = run_inference(
    model_path="models/unet_model.pt",
    dsm_path="data/test_site/dsm.tif",
    output_path="results/prediction.tif",
    threshold=5.0,
    visualize=True
)

# Inférence par lots
dsm_paths = ["data/site1/dsm.tif", "data/site2/dsm.tif"]
results = run_batch_inference(
    model_path="models/unet_model.pt",
    dsm_paths=dsm_paths,
    output_dir="results/batch",
    threshold=5.0,
    batch_size=4,
    num_workers=2
) 