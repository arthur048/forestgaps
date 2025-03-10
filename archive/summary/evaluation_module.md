# Résumé du module d'évaluation

## Vue d'ensemble

Le module d'évaluation de ForestGaps fournit une infrastructure complète pour évaluer les performances des modèles entraînés sur des paires DSM/CHM indépendantes. Il permet de calculer des métriques détaillées, de générer des rapports d'évaluation complets et de comparer différents modèles.

## Architecture

Le module est organisé selon une architecture modulaire avec trois composants principaux :

1. **Classes principales** (`core.py`) :
   - `EvaluationConfig` : Configuration pour le processus d'évaluation
   - `EvaluationResult` : Encapsulation des résultats d'évaluation
   - `ExternalEvaluator` : Gestion du processus d'évaluation

2. **Métriques** (`metrics.py`) :
   - Implémentation des métriques de segmentation
   - Métriques par seuil de hauteur
   - Matrices de confusion

3. **Utilitaires** (`utils/evaluation_utils.py`) :
   - Fonctions pour la création de vérités terrain
   - Génération de rapports
   - Visualisation des résultats

4. **Interface publique** (`__init__.py`) :
   - Fonctions de haut niveau pour l'évaluation
   - Exposition des classes principales

## Fonctionnalités clés

### Évaluation de modèles
- Évaluation sur des paires DSM/CHM uniques
- Évaluation sur des sites complets
- Évaluation sur plusieurs sites avec agrégation des résultats
- Comparaison de différents modèles

### Métriques
- Métriques de segmentation (précision, rappel, F1, IoU)
- Métriques par seuil de hauteur
- Matrices de confusion détaillées
- Métriques agrégées sur plusieurs échantillons

### Génération de rapports
- Rapports détaillés au format HTML/PDF
- Tableaux de métriques
- Visualisations des prédictions vs vérités terrain
- Exportation des métriques au format CSV

### Visualisation
- Visualisation des prédictions
- Visualisation des erreurs
- Comparaison visuelle entre modèles
- Graphiques de performance

## Interfaces principales

### Fonctions de haut niveau

```python
# Évaluation sur une paire DSM/CHM
result = evaluate_model(model_path, dsm_path, chm_path, output_dir, thresholds, config, device, visualize)

# Évaluation sur un site
result = evaluate_site(model_path, site_dsm_dir, site_chm_dir, output_dir, thresholds, config, device, batch_size, num_workers, visualize)

# Évaluation sur plusieurs sites
results = evaluate_model_on_sites(model_path, sites_config, output_dir, thresholds, config, device, batch_size, num_workers, visualize, aggregate_results)

# Comparaison de modèles
results = compare_models(model_paths, dsm_path, chm_path, output_dir, thresholds, config, device, visualize)
```

### Classes principales

```python
# Configuration
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

# Évaluateur
evaluator = ExternalEvaluator(model_path, config, device)
result = evaluator.evaluate(dsm_path, chm_path, output_dir)
site_result = evaluator.evaluate_site(site_dsm_dir, site_chm_dir, output_dir)
multi_site_results = evaluator.evaluate_multi_sites(sites_config, output_dir)

# Résultat d'évaluation
metrics = result.metrics
result.generate_report(output_dir)
result.visualize()
result.save_metrics(output_path)
```

## Métriques implémentées

- **Précision** : Proportion de vrais positifs parmi les prédictions positives
- **Rappel** : Proportion de vrais positifs détectés parmi tous les positifs réels
- **F1-Score** : Moyenne harmonique de la précision et du rappel
- **IoU (Intersection over Union)** : Mesure du chevauchement entre prédiction et vérité terrain
- **Métriques par seuil** : Métriques calculées à différents seuils de hauteur
- **Matrices de confusion** : Analyse détaillée des vrais/faux positifs/négatifs

## Intégration avec d'autres modules

Le module d'évaluation s'intègre avec :

- **models** : Utilise les modèles entraînés pour l'évaluation
- **inference** : Utilise les fonctionnalités d'inférence pour générer des prédictions
- **utils** : Utilise les fonctions utilitaires communes
- **cli** : Expose ses fonctionnalités via l'interface CLI

## Optimisations

- **Traitement parallèle** : Évaluation parallèle sur plusieurs paires DSM/CHM
- **Mise en cache** : Réutilisation des résultats intermédiaires
- **Génération efficace de rapports** : Optimisation de la génération de rapports volumineux
- **Visualisation optimisée** : Génération efficace de visualisations pour de grandes images

## Exemples d'utilisation

```python
from forestgaps.evaluation import evaluate_model, evaluate_site, compare_models

# Évaluation simple
result = evaluate_model(
    model_path="models/unet_model.pt",
    dsm_path="data/test_site/dsm.tif",
    chm_path="data/test_site/chm.tif",
    output_dir="results/evaluation",
    thresholds=[2.0, 5.0, 10.0, 15.0],
    visualize=True
)

# Évaluation d'un site
site_result = evaluate_site(
    model_path="models/unet_model.pt",
    site_dsm_dir="data/site1/dsm",
    site_chm_dir="data/site1/chm",
    output_dir="results/site_evaluation",
    thresholds=[2.0, 5.0, 10.0, 15.0],
    batch_size=4,
    num_workers=2
)

# Comparaison de modèles
model_paths = {
    "unet": "models/unet.pt",
    "deeplabv3": "models/deeplabv3.pt",
    "unet_film": "models/unet_film.pt"
}
comparison_results = compare_models(
    model_paths=model_paths,
    dsm_path="data/test_site/dsm.tif",
    chm_path="data/test_site/chm.tif",
    output_dir="results/model_comparison",
    thresholds=[2.0, 5.0, 10.0, 15.0]
)
``` 