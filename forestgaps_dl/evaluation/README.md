# Module d'évaluation pour ForestGaps-DL

Ce module fournit les fonctionnalités nécessaires pour évaluer les modèles entraînés sur des paires DSM/CHM indépendantes. Il permet de calculer des métriques détaillées de performance, de générer des rapports d'évaluation complets et de comparer différents modèles.

## Architecture du module

Le module d'évaluation est organisé comme suit :

```
evaluation/
├── __init__.py           # Point d'entrée du module avec fonctions principales exposées
├── core.py               # Classes principales pour l'évaluation
├── metrics.py            # Implémentation des métriques d'évaluation
└── utils/                # Utilitaires pour l'évaluation
    └── evaluation_utils.py  # Fonctions utilitaires pour l'évaluation
```

## Classes principales

### EvaluationConfig

Configuration pour le processus d'évaluation, incluant :
- Seuils de hauteur pour l'évaluation
- Options de comparaison avec des modèles précédents
- Paramètres de génération de rapports
- Métriques à calculer

### EvaluationResult

Encapsule les résultats d'une évaluation, incluant :
- Métriques calculées (précision, rappel, F1, IoU, etc.)
- Prédictions et vérités terrain
- Visualisations des résultats
- Méthodes pour sauvegarder et générer des rapports

### ExternalEvaluator

Gère le processus d'évaluation complet :
- Chargement des modèles pré-entraînés
- Création des vérités terrain à partir des CHM
- Évaluation sur des paires DSM/CHM
- Agrégation des résultats sur plusieurs sites
- Génération de rapports détaillés

## Fonctionnalités principales

1. **Évaluation sur paire unique** : Évaluer un modèle sur une paire DSM/CHM unique
2. **Évaluation sur site** : Évaluer un modèle sur toutes les paires DSM/CHM d'un site
3. **Évaluation multi-sites** : Évaluer un modèle sur plusieurs sites et agréger les résultats
4. **Comparaison de modèles** : Comparer les performances de différents modèles
5. **Métriques par seuil** : Calculer des métriques pour différents seuils de hauteur
6. **Génération de rapports** : Créer des rapports détaillés avec visualisations

## Métriques implémentées

- **Métriques de segmentation** : Précision, Rappel, F1-Score, IoU
- **Métriques de régression** : RMSE, MAE, R²
- **Métriques par seuil** : Métriques calculées à différents seuils de hauteur
- **Matrices de confusion** : Analyse détaillée des vrais/faux positifs/négatifs

## Utilisation

### Évaluation sur une paire DSM/CHM

```python
from forestgaps_dl.evaluation import evaluate_model

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
metrics = result.metrics

# Générer un rapport
result.generate_report("path/to/report")

# Visualiser les résultats
result.visualize()
```

### Évaluation sur un site complet

```python
from forestgaps_dl.evaluation import evaluate_site

# Évaluer un modèle sur un site complet
result = evaluate_site(
    model_path="path/to/model.pt",
    site_dsm_dir="path/to/dsm_dir",
    site_chm_dir="path/to/chm_dir",
    output_dir="path/to/output",
    thresholds=[2.0, 5.0, 10.0, 15.0],
    batch_size=4,
    num_workers=2,
    visualize=True
)

# Accéder aux métriques agrégées
site_metrics = result.metrics

# Générer un rapport pour le site
result.generate_report("path/to/site_report")
```

### Évaluation sur plusieurs sites

```python
from forestgaps_dl.evaluation import evaluate_model_on_sites

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
    thresholds=[2.0, 5.0, 10.0, 15.0],
    batch_size=4,
    num_workers=2,
    visualize=True,
    aggregate_results=True
)

# Accéder aux résultats par site
for site_name, result in results.items():
    if site_name != "aggregated":
        print(f"Métriques pour {site_name}: {result.metrics}")

# Accéder aux résultats agrégés
if "aggregated" in results:
    print(f"Métriques agrégées: {results['aggregated'].metrics}")
```

### Comparaison de modèles

```python
from forestgaps_dl.evaluation import compare_models

# Modèles à comparer
model_paths = {
    "unet": "path/to/unet.pt",
    "deeplabv3": "path/to/deeplabv3.pt",
    "unet_film": "path/to/unet_film.pt"
}

# Comparer les modèles
results = compare_models(
    model_paths=model_paths,
    dsm_path="path/to/dsm.tif",
    chm_path="path/to/chm.tif",
    output_dir="path/to/comparison",
    thresholds=[2.0, 5.0, 10.0, 15.0],
    visualize=True
)

# Accéder aux résultats par modèle
for model_name, result in results.items():
    print(f"Métriques pour {model_name}: {result.metrics}")
```

## Configuration avancée

```python
from forestgaps_dl.evaluation import ExternalEvaluator, EvaluationConfig

# Configuration personnalisée
config = EvaluationConfig(
    thresholds=[2.0, 5.0, 10.0, 15.0],
    compare_with_previous=True,
    previous_model_path="path/to/previous_model.pt",
    save_predictions=True,
    save_visualizations=True,
    detailed_reporting=True,
    metrics=["precision", "recall", "f1", "iou"],
    batch_size=8,
    num_workers=4,
    tiled_processing=True
)

# Créer un évaluateur
evaluator = ExternalEvaluator(
    model_path="path/to/model.pt",
    config=config,
    device="cuda"
)

# Évaluer sur une paire DSM/CHM
result = evaluator.evaluate(
    dsm_path="path/to/dsm.tif",
    chm_path="path/to/chm.tif",
    output_dir="path/to/output"
)
```

## Intégration avec d'autres modules

Le module d'évaluation s'intègre avec d'autres modules de ForestGaps-DL :

- **models** : Utilise les modèles entraînés pour l'évaluation
- **inference** : Utilise les fonctionnalités d'inférence pour générer des prédictions
- **utils** : Utilise les fonctions utilitaires pour le traitement des données et la visualisation

## Performances et optimisation

Le module d'évaluation est optimisé pour :

- Traiter efficacement de grandes quantités de données
- Paralléliser l'évaluation sur plusieurs paires DSM/CHM
- Générer des rapports détaillés et informatifs
- Visualiser les résultats de manière claire et concise 