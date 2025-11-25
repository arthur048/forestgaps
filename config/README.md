# Configuration de ForestGaps

Ce module fournit un système de configuration flexible pour le projet ForestGaps. Il permet de gérer les paramètres pour le traitement des données, les modèles et l'entraînement.

## Structure

- `base.py` : Classe de configuration de base
- `schema.py` : Schémas de validation pour les configurations
- `__init__.py` : Point d'entrée du module avec fonctions utilitaires
- `defaults/` : Configurations par défaut au format YAML
  - `data.yaml` : Configuration par défaut pour les données
  - `models.yaml` : Configuration par défaut pour les modèles
  - `training.yaml` : Configuration par défaut pour l'entraînement

## Utilisation

### Charger la configuration par défaut

```python
from forestgaps.config import load_default_config

# Charger la configuration par défaut
config = load_default_config()

# Accéder aux paramètres
print(f"Taille des tuiles : {config.TILE_SIZE}")
print(f"Type de modèle : {config.MODEL_TYPE}")
```

### Charger une configuration personnalisée

```python
from forestgaps.config import load_config_from_file

# Charger une configuration à partir d'un fichier YAML ou JSON
config = load_config_from_file("path/to/my_config.yaml")
```

### Créer une configuration à partir d'un dictionnaire

```python
from forestgaps.config import create_config_from_dict

# Créer une configuration à partir d'un dictionnaire
config_dict = {
    "TILE_SIZE": 512,
    "BATCH_SIZE": 32,
    "MODEL_TYPE": "basic"
}
config = create_config_from_dict(config_dict)
```

### Sauvegarder une configuration

```python
# Sauvegarder la configuration dans un fichier YAML
config.save_config("path/to/save/config.yaml", format="yaml")

# Sauvegarder la configuration dans un fichier JSON
config.save_config("path/to/save/config.json", format="json")
```

### Fusionner plusieurs configurations

```python
# Charger la configuration par défaut
config = load_default_config()

# Fusionner avec d'autres fichiers de configuration
config.merge_configs("path/to/data_config.yaml", "path/to/model_config.yaml")
```

## Création de configurations personnalisées

Pour créer une configuration personnalisée, vous pouvez :

1. Modifier les fichiers YAML dans le dossier `defaults/`
2. Créer vos propres fichiers YAML et les charger avec `load_config_from_file()`
3. Créer un dictionnaire de configuration et le charger avec `create_config_from_dict()`

## Validation des configurations

Les configurations sont automatiquement validées lors du chargement grâce aux schémas définis dans `schema.py`. Cela garantit que les paramètres sont du bon type et dans les plages acceptables.

## Migration depuis l'ancien système

Ce système de configuration remplace les classes `Config` des fichiers legacy :
- `forestgaps_u_net_training.py`
- `forestgaps_data_preparation.py`

Il offre une plus grande flexibilité et une meilleure organisation des paramètres. 