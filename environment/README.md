# Gestion d'Environnement pour ForestGaps

Ce module fournit un système flexible pour détecter et configurer automatiquement l'environnement d'exécution (Google Colab ou local) pour le projet ForestGaps.

## Structure

- `base.py` : Classe abstraite définissant l'interface commune
- `colab.py` : Implémentation spécifique pour Google Colab
- `local.py` : Implémentation spécifique pour l'environnement local
- `__init__.py` : Point d'entrée du module avec fonctions utilitaires

## Fonctionnalités

Le système de gestion d'environnement offre les fonctionnalités suivantes :

- **Détection automatique** de l'environnement d'exécution (Colab ou local)
- **Configuration automatique** de l'environnement
- **Montage de Google Drive** dans Colab
- **Installation des dépendances** nécessaires
- **Configuration du GPU** si disponible
- **Récupération d'informations** sur l'environnement

## Utilisation

### Détection et configuration automatique

```python
from forestgaps.environment import setup_environment

# Détecte et configure automatiquement l'environnement
env = setup_environment()

# Obtenir le répertoire de base
base_dir = env.get_base_dir()
```

### Détection uniquement

```python
from forestgaps.environment import detect_environment

# Détecte l'environnement sans le configurer
env = detect_environment()
```

### Obtenir le dispositif à utiliser pour les calculs

```python
from forestgaps.environment import get_device

# Détecte si un GPU est disponible et renvoie 'cuda' ou 'cpu'
device = get_device()
```

### Obtenir des informations sur l'environnement

```python
env = setup_environment()
env_info = env.get_environment_info()

# Afficher les informations
for key, value in env_info.items():
    print(f"{key}: {value}")
```

## Classes d'environnement

### Environment (base)

Classe abstraite définissant l'interface commune pour tous les environnements.

### ColabEnvironment

Classe spécifique pour Google Colab qui :
- Monte Google Drive
- Installe les dépendances nécessaires
- Configure le GPU si disponible
- Utilise le répertoire `/content/drive/MyDrive/ForestGaps_DeepLearning` comme base

### LocalEnvironment

Classe spécifique pour l'environnement local qui :
- Vérifie les dépendances
- Configure le GPU si disponible
- Utilise le répertoire racine du projet comme base

## Exemple complet

Voir le fichier `examples/environment_usage.py` pour un exemple complet d'utilisation.

## Intégration avec le module Config

Ce système de gestion d'environnement peut être utilisé en combinaison avec le module de configuration pour charger des configurations spécifiques à l'environnement :

```python
from forestgaps.environment import setup_environment
from forestgaps.config import load_config_from_file

# Configurer l'environnement
env = setup_environment()

# Charger la configuration spécifique à l'environnement
if isinstance(env, ColabEnvironment):
    config = load_config_from_file("config/defaults/colab.yaml")
else:
    config = load_config_from_file("config/defaults/local.yaml")
``` 