# Module `utils`

## Vue d'ensemble

Le module `utils` fournit des fonctionnalités communes utilisées dans l'ensemble du package ForestGaps-DL, notamment pour la visualisation, les entrées/sorties, le profilage et la gestion des erreurs.

## Structure

```
utils/
├── __init__.py               # Point d'entrée unifié
├── errors.py                 # Système hiérarchique d'exceptions
├── visualization/            # Visualisations
│   ├── __init__.py
│   ├── plots.py              # Création de graphiques
│   ├── maps.py               # Visualisation des cartes
│   └── tensorboard.py        # Intégration TensorBoard
├── io/                       # Entrées/sorties
│   ├── __init__.py
│   ├── raster.py             # Opérations sur les rasters
│   └── serialization.py      # Sérialisation/désérialisation
└── profiling/                # Profilage des performances
    ├── __init__.py
    └── benchmarks.py         # Outils de benchmarking
```

## Fonctionnalités principales

### Gestion des erreurs (`errors.py`)

- Système hiérarchique d'exceptions personnalisées pour une gestion précise des erreurs
- Classes d'erreurs spécifiques pour chaque module (données, modèles, entraînement, etc.)
- Gestionnaire d'erreurs centralisé pour la journalisation et l'affichage des erreurs

### Visualisation (`visualization/`)

- **plots.py**: Fonctions pour créer des graphiques (évolution des métriques, métriques par seuil, matrices de confusion)
- **maps.py**: Fonctions pour visualiser les données géospatiales (DSM avec trouées, prédictions, comparaisons)
- **tensorboard.py**: Intégration avec TensorBoard pour le suivi des expériences, avec un système de monitoring centralisé

### Entrées/sorties (`io/`)

- **raster.py**: Fonctions pour manipuler les données raster (chargement, sauvegarde, normalisation, statistiques)
- **serialization.py**: Fonctions pour sérialiser et désérialiser des objets (JSON, YAML, pickle, modèles PyTorch, etc.)

### Profilage (`profiling/`)

- **benchmarks.py**: Outils pour mesurer et analyser les performances (temps d'exécution, transferts CPU/GPU, optimisation des DataLoaders)

## Utilisation

### Gestion des erreurs

```python
from utils.errors import DataProcessingError, ErrorHandler

# Initialiser le gestionnaire d'erreurs
error_handler = ErrorHandler(log_file="errors.log", verbose=True)

try:
    # Code susceptible de générer une erreur
    process_data()
except Exception as e:
    # Gérer l'erreur
    error_handler.handle(e, context={'operation': 'process_data'})
```

### Visualisation

```python
from utils.visualization.plots import visualize_metrics_by_threshold
from utils.visualization.maps import visualize_dsm_with_gaps
from utils.visualization.tensorboard import MonitoringSystem

# Créer un système de monitoring
monitoring = MonitoringSystem(log_dir="logs")

# Visualiser des métriques
visualize_metrics_by_threshold(metrics, save_path="metrics.png")

# Visualiser un DSM avec des trouées
visualize_dsm_with_gaps(dsm, gaps_mask, save_path="dsm_with_gaps.png")
```

### Entrées/sorties

```python
from utils.io.raster import load_raster, normalize_raster
from utils.io.serialization import save_json, load_model

# Charger et normaliser un raster
data, metadata = load_raster("dsm.tif")
normalized_data = normalize_raster(data, method="minmax")

# Sauvegarder des données au format JSON
save_json(metrics, "metrics.json")

# Charger un modèle PyTorch
checkpoint = load_model("model.pt", model, optimizer, device)
```

### Profilage

```python
from utils.profiling.benchmarks import timeit, optimize_dataloader_params

# Mesurer le temps d'exécution d'une fonction
@timeit
def process_data():
    # Traitement des données
    pass

# Optimiser les paramètres d'un DataLoader
optimal_params = optimize_dataloader_params(dataset, batch_size=32)
```

## Intégration avec les autres modules

- **config**: Utilise les configurations validées pour paramétrer les fonctionnalités
- **environment**: S'adapte à l'environnement d'exécution (local ou Colab)
- **data**: Fournit des fonctions pour manipuler les données raster et visualiser les résultats
- **models**: Offre des fonctions pour sérialiser et désérialiser les modèles
- **training**: Fournit des outils de visualisation et de monitoring pour l'entraînement

## Améliorations futures

- Ajout de tests unitaires pour chaque fonction
- Optimisation des performances des fonctions de visualisation pour de grands ensembles de données
- Intégration avec d'autres outils de visualisation (Weights & Biases, MLflow, etc.)
- Ajout de fonctionnalités de profilage plus avancées (mémoire, GPU, etc.) 