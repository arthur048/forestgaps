# Configuration de ForestGaps sur Google Colab

Ce guide explique comment installer et utiliser ForestGaps sur Google Colab.

## Installation Rapide

### Méthode 1: Script d'installation automatique (Recommandé)

```python
!wget -O colab_install.py https://raw.githubusercontent.com/arthur048/forestgaps/main/colab_install.py
%run colab_install.py
```

**IMPORTANT**: Après l'installation, **redémarrez le runtime Colab** (`Runtime > Restart runtime`)

### Méthode 2: Installation manuelle

```python
# Installer le package depuis GitHub
!pip install --no-dependencies git+https://github.com/arthur048/forestgaps.git

# Vérifier l'installation
import forestgaps
print(f"ForestGaps version: {forestgaps.__version__}")
```

## ⚠️ Nom du Package

**Le package s'appelle `forestgaps`** (pas `forestgaps_dl` ni `forestgaps-dl`)

### Imports Corrects

```python
# ✅ CORRECT
from forestgaps.environment import setup_environment
from forestgaps.config import load_default_config
from forestgaps.models import create_model

# ❌ INCORRECT
from forestgaps_dl.config import ...  # Nom incorrect
from forestgaps-dl.environment import ...  # Syntaxe invalide
```

## Setup de l'Environnement

### Détection Automatique et Configuration

```python
from forestgaps.environment import setup_environment

# Setup automatique (détecte Colab)
env = setup_environment()

# Afficher les infos d'environnement
info = env.get_environment_info()
print(f"Environnement: {info['environment']}")
print(f"GPU disponible: {info['gpu_available']}")
print(f"Device: {info['device']}")
```

### Montage de Google Drive

Le script tente de monter automatiquement Google Drive. Si cela échoue:

```python
from google.colab import drive
drive.mount('/content/drive')

# Vérifier le montage
import os
base_dir = '/content/drive/MyDrive/ForestGaps_DeepLearning'
if os.path.exists(base_dir):
    print(f"✅ Drive monté: {base_dir}")
else:
    print("❌ Répertoire introuvable")
```

### Configuration du Répertoire de Travail

```python
import os

# Définir le répertoire de base sur Drive
BASE_DIR = '/content/drive/MyDrive/ForestGaps_DeepLearning'
os.makedirs(BASE_DIR, exist_ok=True)

# Structure recommandée
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

for dir_path in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
```

## Vérification de l'Installation

### Test d'Import Complet

```python
# Test des imports principaux
try:
    from forestgaps.environment import setup_environment
    from forestgaps.config import load_default_config
    from forestgaps.models import create_model
    from forestgaps.data.loaders import create_data_loaders
    from forestgaps.training import Trainer
    print("✅ Tous les imports fonctionnent")
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
```

### Test de Configuration

```python
from forestgaps.config import load_default_config

# Charger config par défaut
config = load_default_config()
print(f"Config chargée: {type(config)}")

# Accéder aux paramètres
batch_size = config.get('training.batch_size', 8)
print(f"Batch size: {batch_size}")
```

## Dépendances

Les dépendances principales sont installées automatiquement par le script `colab_install.py`:

- `torch` et `torchvision`
- `numpy`, `matplotlib`
- `rasterio`, `geopandas`
- `pydantic`, `PyYAML`
- `tqdm`, `tensorboard`

## Troubleshooting

### ModuleNotFoundError: No module named 'forestgaps'

**Problème**: Le package n'est pas installé ou le runtime n'a pas été redémarré.

**Solution**:
1. Vérifier l'installation: `!pip list | grep forestgaps`
2. Redémarrer le runtime: `Runtime > Restart runtime`
3. Réinstaller si nécessaire: `!pip install --no-dependencies git+https://github.com/arthur048/forestgaps.git`

### ModuleNotFoundError: No module named 'forestgaps_dl'

**Problème**: Utilisation du mauvais nom de package.

**Solution**: Remplacer tous les imports `forestgaps_dl` par `forestgaps`

### Erreur de montage Google Drive

**Problème**: `Error: credential propagation was unsuccessful`

**Solution**:
```python
# Montage manuel avec authentification
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### GPU non disponible

**Problème**: `❌ Aucun GPU détecté`

**Solution**:
1. Vérifier les paramètres runtime: `Runtime > Change runtime type`
2. Sélectionner `GPU` dans Hardware accelerator
3. Redémarrer le runtime

### Dépendances manquantes

**Problème**: `ModuleNotFoundError` pour rasterio, geopandas, etc.

**Solution**:
```python
# Installer manuellement les dépendances manquantes
!pip install rasterio geopandas
```

## Workflow Complet

Voici un exemple de workflow complet sur Colab:

```python
# 1. Installation
!wget -O colab_install.py https://raw.githubusercontent.com/arthur048/forestgaps/main/colab_install.py
%run colab_install.py

# Redémarrer runtime ici

# 2. Setup environnement
from forestgaps.environment import setup_environment
env = setup_environment()

# 3. Montage Drive (si nécessaire)
from google.colab import drive
drive.mount('/content/drive')

# 4. Chargement configuration
from forestgaps.config import load_default_config
config = load_default_config()

# 5. Préparation des données
from forestgaps.data.preprocessing import process_raster_pair_robustly
from forestgaps.data.generation import generate_gap_masks

# 6. Création DataLoaders
from forestgaps.data.loaders import create_data_loaders
loaders = create_data_loaders(config)

# 7. Création modèle
from forestgaps.models import create_model
model = create_model('unet', in_channels=1, num_classes=2)

# 8. Entraînement
from forestgaps.training import Trainer
trainer = Trainer(model, config, loaders['train'], loaders['val'])
results = trainer.train(epochs=10)

# 9. Sauvegarde
trainer.save_checkpoint('/content/drive/MyDrive/ForestGaps/models/model.pt')
```

## Ressources

- **GitHub**: https://github.com/arthur048/forestgaps
- **Documentation principale**: [README.md](../README.md)
- **Guide Google Drive**: [GOOGLE_DRIVE_SETUP.md](./GOOGLE_DRIVE_SETUP.md)
- **Script d'installation**: [colab_install.py](../colab_install.py)

## Notes Importantes

1. **Nom du package**: Toujours utiliser `forestgaps` (pas `forestgaps_dl`)
2. **Redémarrage**: Toujours redémarrer le runtime après installation
3. **Persistance**: Sauvegarder sur Google Drive pour conserver entre sessions
4. **GPU**: Vérifier que GPU est activé pour l'entraînement
5. **Batch size**: Réduire si OutOfMemoryError (4-8 pour Colab gratuit)

## Mise à Jour

Pour mettre à jour vers la dernière version:

```python
!pip uninstall -y forestgaps
!pip install --no-dependencies git+https://github.com/arthur048/forestgaps.git

# Redémarrer runtime
```

## Support

En cas de problème:
1. Vérifier cette documentation
2. Consulter les issues GitHub
3. Vérifier que vous utilisez `forestgaps` (pas `forestgaps_dl`)
