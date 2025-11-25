# Installation de ForestGaps dans Google Colab

Ce guide explique comment installer correctement ForestGaps dans l'environnement Google Colab.

## Méthode recommandée

La méthode recommandée utilise notre script d'installation spécial pour Colab qui:
- Évite de réinstaller les dépendances déjà présentes
- Configure correctement l'environnement Colab
- Simplifie le processus d'installation

```python
# Télécharger le script d'installation pour Colab
!wget -O colab_install.py https://raw.githubusercontent.com/arthur048/forestgaps/main/colab_install.py

# Exécuter le script d'installation
%run colab_install.py

# Redémarrer le runtime (nécessaire après l'installation)
# Cliquez sur "Runtime" > "Restart runtime" une fois l'installation terminée

# Après redémarrage, importez et configurez le module:
from forestgaps.environment import setup_environment
env = setup_environment()  # Détecte et configure automatiquement l'environnement Colab
```

## Méthode alternative

Si vous rencontrez des problèmes avec la méthode recommandée, vous pouvez utiliser cette approche alternative:

```python
# Installer sans dépendances (évite les réinstallations inutiles)
!pip install --no-dependencies git+https://github.com/arthur048/forestgaps.git

# Redémarrer le runtime (nécessaire après l'installation)
# Cliquez sur "Runtime" > "Restart runtime" une fois l'installation terminée

# Après redémarrage, importez et configurez le module:
from forestgaps.environment import setup_environment
env = setup_environment()
```

## Troubleshooting

### Module not found

Si vous obtenez l'erreur `ModuleNotFoundError: No module named 'forestgaps'`:

1. Vérifiez que vous avez bien redémarré le runtime après l'installation
2. Essayez d'installer le package avec l'option `--force-reinstall`:
   ```python
   !pip install --force-reinstall git+https://github.com/arthur048/forestgaps.git
   ```
3. Vérifiez l'installation avec:
   ```python
   !pip show forestgaps
   ```

### Dépendances manquantes

Si certaines dépendances sont manquantes:

```python
# Installer uniquement la dépendance manquante
!pip install nom_dependance

# Puis redémarrer le runtime
```

## Vérification de l'installation

Pour vérifier que tout fonctionne correctement:

```python
# Vérifier la version du package
import forestgaps
print(forestgaps.__version__)

# Vérifier la détection de l'environnement
from forestgaps.environment import detect_environment
env = detect_environment()
print(f"Environnement détecté: {type(env).__name__}")

# Vérifier la disponibilité du GPU
from forestgaps.environment import get_device
device = get_device()
print(f"Dispositif disponible: {device}")
``` 