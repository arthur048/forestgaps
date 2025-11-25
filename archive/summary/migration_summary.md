# Résumé de la migration de la classe Config

## Objectif
Migrer la classe Config des fichiers legacy (`forestgaps_u_net_training.py` et `forestgaps_data_preparation.py`) vers une structure plus moderne avec des schémas YAML.

## Fichiers créés ou modifiés

### Classes de base et schémas
- `config/base.py` : Classe Config de base avec les méthodes pour charger et sauvegarder des configurations
- `config/schema.py` : Schémas de validation pour les différentes configurations (données, modèles, entraînement)
- `config/__init__.py` : Point d'entrée du module avec fonctions utilitaires

### Fichiers de configuration YAML
- `config/defaults/data.yaml` : Configuration par défaut pour les données
- `config/defaults/models.yaml` : Configuration par défaut pour les modèles
- `config/defaults/training.yaml` : Configuration par défaut pour l'entraînement

### Documentation et exemples
- `config/README.md` : Documentation sur l'utilisation du système de configuration
- `examples/config_usage.py` : Exemple d'utilisation du système de configuration

### Autres modifications
- `setup.py` : Ajout des dépendances nécessaires (pydantic, PyYAML, geopandas)

## Améliorations apportées

1. **Séparation des préoccupations** : Les configurations sont maintenant séparées par domaine (données, modèles, entraînement).

2. **Validation des données** : Utilisation de Pydantic pour valider les configurations et garantir leur cohérence.

3. **Flexibilité** : Possibilité de charger des configurations à partir de fichiers YAML ou JSON, ou de les créer à partir de dictionnaires.

4. **Extensibilité** : Facilité d'ajout de nouveaux paramètres ou de nouvelles sections de configuration.

5. **Maintenabilité** : Code plus propre et mieux documenté, avec des types explicites.

6. **Réutilisabilité** : Les configurations peuvent être facilement partagées et réutilisées entre différentes parties du projet.

## Utilisation

```python
# Charger la configuration par défaut
from forestgaps.config import load_default_config
config = load_default_config()

# Accéder aux paramètres
print(f"Taille des tuiles : {config.TILE_SIZE}")
print(f"Type de modèle : {config.MODEL_TYPE}")

# Modifier la configuration
config.BATCH_SIZE = 32
config.MODEL_TYPE = "basic"

# Sauvegarder la configuration
config.save_config("path/to/save/config.yaml", format="yaml")
```

## Prochaines étapes

1. Mettre à jour les autres parties du code pour utiliser le nouveau système de configuration.

2. Créer des configurations spécifiques pour différents cas d'utilisation (par exemple, entraînement rapide, inférence, etc.).

3. Ajouter des tests unitaires pour s'assurer que le système de configuration fonctionne correctement. 