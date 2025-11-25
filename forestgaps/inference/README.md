# Module d'inférence pour ForestGaps

Ce module fournit les fonctionnalités nécessaires pour appliquer les modèles entraînés à de nouvelles données DSM. Il permet d'effectuer des prédictions sur des données non vues pendant l'entraînement, de visualiser les résultats et de sauvegarder les prédictions.

## Architecture du module

Le module d'inférence est organisé comme suit :

```
inference/
├── __init__.py        # Point d'entrée du module avec fonctions principales exposées
├── core.py            # Classes principales pour l'inférence
└── utils/             # Utilitaires pour l'inférence
    ├── geospatial.py  # Fonctions pour manipuler les données géospatiales
    ├── image_processing.py  # Fonctions de traitement d'images
    └── visualization.py     # Fonctions de visualisation des prédictions
```

## Classes principales

### InferenceConfig

Configuration pour le processus d'inférence, incluant :
- Paramètres de traitement par tuiles
- Taille des lots (batch size)
- Seuils de probabilité
- Options de post-traitement

### InferenceResult

Encapsule les résultats d'une opération d'inférence, incluant :
- Prédictions brutes
- Prédictions post-traitées
- Métadonnées géospatiales
- Méthodes pour sauvegarder et visualiser les résultats

### InferenceManager

Gère le processus d'inférence complet :
- Chargement des modèles pré-entraînés
- Prétraitement des données DSM
- Exécution de l'inférence
- Post-traitement des prédictions
- Gestion des prédictions par lots

## Fonctionnalités principales

1. **Inférence sur fichier unique** : Appliquer un modèle à un seul fichier DSM
2. **Inférence par lots** : Traiter plusieurs fichiers DSM en séquence
3. **Traitement par tuiles** : Gérer efficacement les grandes images en les découpant en tuiles
4. **Préservation des métadonnées** : Conserver les informations géospatiales dans les prédictions
5. **Visualisation des résultats** : Générer des visualisations des prédictions
6. **Exportation des résultats** : Sauvegarder les prédictions dans différents formats

## Utilisation

### Inférence sur un fichier unique

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

### Inférence par lots

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
for dsm_path, result in results.items():
    print(f"Résultats pour {dsm_path}: {result.summary()}")
```

### Configuration avancée

```python
from forestgaps.inference import InferenceManager, InferenceConfig

# Configuration personnalisée
config = InferenceConfig(
    tiled_processing=True,
    tile_size=512,
    tile_overlap=64,
    batch_size=8,
    threshold_probability=0.5,
    post_processing={
        "apply_smoothing": True,
        "smoothing_kernel_size": 3,
        "remove_small_objects": True,
        "min_size": 100
    }
)

# Créer un gestionnaire d'inférence
manager = InferenceManager(
    model_path="path/to/model.pt",
    config=config,
    device="cuda"
)

# Exécuter l'inférence
result = manager.predict("path/to/dsm.tif")
```

## Intégration avec d'autres modules

Le module d'inférence s'intègre avec d'autres modules de ForestGaps :

- **models** : Utilise les modèles entraînés pour l'inférence
- **utils** : Utilise les fonctions utilitaires pour le traitement des données
- **evaluation** : Peut être utilisé en conjonction avec le module d'évaluation pour évaluer les performances des modèles

## Performances et optimisation

Le module d'inférence est optimisé pour :

- Utiliser efficacement le GPU lorsqu'il est disponible
- Gérer les grandes images via le traitement par tuiles
- Minimiser l'utilisation de la mémoire
- Paralléliser le traitement lorsque c'est possible 