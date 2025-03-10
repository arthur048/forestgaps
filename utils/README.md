# Module d'utilitaires pour ForestGaps

Ce module fournit des fonctions et classes utilitaires utilisées à travers le package, organisées par domaine fonctionnel.

## Structure du module

- **visualization/** : Visualisation des données et résultats
- **io/** : Opérations d'entrée/sortie
- **profiling/** : Outils de profilage
- **errors.py** : Gestion hiérarchique des erreurs

## Sous-modules

### Module de visualisation (`visualization`)

Fonctions pour visualiser les données géospatiales, les prédictions et les résultats d'évaluation.

```python
from forestgaps.utils.visualization import plots

# Visualiser un fichier raster
plots.plot_raster("path/to/dsm.tif", title="Modèle numérique de surface")

# Comparer prédiction et vérité terrain
plots.plot_comparison(
    prediction="path/to/prediction.tif",
    ground_truth="path/to/ground_truth.tif",
    title="Comparaison"
)
```

### Module d'entrée/sortie (`io`)

Fonctions pour la lecture et l'écriture de fichiers, la gestion des chemins et les opérations de système de fichiers.

```python
from forestgaps.utils.io import serialization, raster

# Charger un modèle
model = serialization.load_model("path/to/model.pt")

# Sauvegarder des résultats
serialization.save_json(results, "path/to/results.json")

# Charger un raster
data, metadata = raster.load_raster("path/to/dsm.tif")
```

### Module de profilage (`profiling`)

Outils pour mesurer les performances et la consommation de ressources des différentes parties du code.

```python
from forestgaps.utils.profiling import benchmarks

# Mesurer le temps d'exécution
with benchmarks.Timer("Prétraitement") as timer:
    # Code à chronométrer
    pass

print(f"Temps écoulé: {timer.elapsed:.2f} secondes")

# Comparer les performances de différentes implémentations
results = benchmarks.compare_functions(
    functions=[func1, func2, func3],
    input_data=test_data,
    num_runs=10
)
```

### Module de gestion d'erreurs (`errors`)

Hiérarchie d'exceptions personnalisées pour une gestion d'erreurs plus précise et informative.

```python
from forestgaps.utils.errors import ForestGapsError, DataError, ModelError

# Lever une exception spécifique
if not os.path.exists(config_path):
    raise ConfigError(f"Le fichier de configuration '{config_path}' n'existe pas")

try:
    # Traitement...
except Exception as e:
    raise DataProcessingError(f"Erreur lors du traitement de l'image: {str(e)}")
```

## Utilisation avancée

Pour des cas d'utilisation plus avancés et des exemples détaillés, consultez la documentation de chaque sous-module. 