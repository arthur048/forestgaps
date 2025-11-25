# Module de Benchmarking

Ce module fournit des outils pour la comparaison systématique des performances des différents modèles de détection de trouées forestières. Il permet d'évaluer et de comparer les modèles sur différents ensembles de données et avec différentes configurations.

## Structure du module

```
benchmarking/
├── __init__.py               # Point d'entrée unifié
├── comparison.py             # Classe principale de comparaison de modèles
├── metrics.py                # Agrégation et analyse des métriques
├── visualization.py          # Visualisation des résultats comparatifs
└── reporting.py              # Génération de rapports détaillés
```

## Fonctionnalités principales

### Comparaison de modèles

La classe `ModelComparison` dans `comparison.py` est le cœur du module. Elle permet de :

- Comparer plusieurs architectures de modèles sur les mêmes données
- Évaluer l'impact des hyperparamètres sur les performances
- Tester différents seuils de détection de trouées
- Exécuter des expériences avec différentes configurations d'entraînement

### Métriques d'évaluation

Le module `metrics.py` fournit des fonctionnalités pour :

- Calculer des métriques agrégées sur plusieurs exécutions
- Analyser la stabilité des modèles
- Comparer statistiquement les performances des modèles
- Identifier les forces et faiblesses de chaque approche

### Visualisation des résultats

Le module `visualization.py` offre des outils pour :

- Générer des graphiques comparatifs (courbes ROC, précision-rappel, etc.)
- Visualiser les distributions de métriques
- Créer des heatmaps de performance par configuration
- Produire des visualisations interactives pour l'analyse approfondie

### Génération de rapports

Le module `reporting.py` permet de :

- Générer des rapports détaillés au format HTML ou PDF
- Créer des tableaux récapitulatifs des performances
- Produire des fiches techniques pour chaque modèle
- Exporter les résultats dans différents formats (CSV, JSON, etc.)

## Dépendances internes

- **config** : Pour charger et gérer les configurations d'expériences
- **models** : Pour instancier les différents modèles à comparer
- **training** : Pour l'entraînement et l'évaluation des modèles
- **utils** : Pour les fonctionnalités communes et la visualisation

## Utilisation

### Comparaison simple de modèles

```python
from forestgaps.config import load_default_config
from forestgaps.benchmarking import ModelComparison

# Définir les modèles à comparer
model_configs = [
    {"name": "unet", "display_name": "U-Net Base"},
    {"name": "unet_film", "display_name": "U-Net FiLM"},
    {"name": "deeplabv3_plus", "display_name": "DeepLabV3+"}
]

# Créer et exécuter la comparaison
benchmark = ModelComparison(
    model_configs=model_configs,
    base_config=load_default_config(),
    threshold_values=[2.0, 5.0, 10.0, 15.0]
)

# Exécuter la comparaison
results = benchmark.run()

# Visualiser les résultats
benchmark.visualize_results()

# Générer un rapport
benchmark.generate_report("rapport_comparaison.html")
```

### Analyse approfondie des performances

```python
from forestgaps.benchmarking import MetricsAnalyzer

# Analyser les résultats d'une comparaison
analyzer = MetricsAnalyzer(results)

# Effectuer des tests statistiques
significance = analyzer.statistical_tests()

# Identifier les points forts et faibles
strengths_weaknesses = analyzer.analyze_strengths_weaknesses()

# Analyser la sensibilité aux hyperparamètres
sensitivity = analyzer.hyperparameter_sensitivity()
```

### Visualisations personnalisées

```python
from forestgaps.benchmarking import BenchmarkVisualizer

# Créer un visualiseur avec les résultats
visualizer = BenchmarkVisualizer(results)

# Générer une matrice de confusion comparative
visualizer.plot_confusion_matrix_comparison()

# Créer des courbes ROC comparatives
visualizer.plot_roc_curves()

# Visualiser les distributions de métriques
visualizer.plot_metric_distributions("f1_score")
```

## Intégration avec l'interface CLI

Le module de benchmarking s'intègre avec l'interface en ligne de commande via le module `cli`, permettant d'exécuter des comparaisons de modèles directement depuis la ligne de commande :

```bash
python -m forestgaps.cli.benchmark --models unet,unet_film,deeplabv3_plus --thresholds 2,5,10,15 --output rapport.html
```

## Principes de conception

Le module `benchmarking` suit les principes SOLID :

1. **Principe de responsabilité unique** : Chaque classe a une responsabilité spécifique (comparaison, métriques, visualisation, rapports).
2. **Principe ouvert/fermé** : L'architecture permet d'ajouter de nouvelles métriques ou visualisations sans modifier le code existant.
3. **Principe de substitution de Liskov** : Les classes dérivées peuvent remplacer leurs classes de base sans altérer le comportement.
4. **Principe de ségrégation d'interface** : Les interfaces sont spécifiques et minimales.
5. **Principe d'inversion de dépendance** : Le code dépend des abstractions, pas des implémentations concrètes.

## Extensibilité

Le module est conçu pour être facilement extensible :

- Ajout de nouvelles métriques d'évaluation
- Intégration de nouvelles visualisations
- Support de formats de rapport supplémentaires
- Implémentation de nouvelles stratégies de comparaison 