# Google Colab Setup Guide

Guide complet pour utiliser ForestGaps sur Google Colab.

## Installation Rapide

```python
# 1. Installer le package
!pip install git+https://github.com/<username>/forestgaps-dl.git

# 2. Vérifier l'installation
import forestgaps
print(f"✓ ForestGaps {forestgaps.__version__} installé")
```

## Chargement des Configurations

### Configs par Défaut

Les configs par défaut sont **automatiquement incluses** dans le package:

```python
from forestgaps.config import load_training_config, load_data_config

# Charge les configs par défaut
training_config = load_training_config()
data_config = load_data_config()

print(f"✓ Epochs: {training_config.epochs}")
print(f"✓ Batch size: {training_config.batch_size}")
```

### Configs Test/Production

Les configs test et production sont également incluses. Le loader cherche automatiquement dans le package installé:

```python
# Option 1: Chemin relatif (recommandé sur Colab)
# Le loader cherche dans forestgaps/configs/ automatiquement
training_config = load_training_config("configs/test/quick.yaml")
data_config = load_data_config("configs/test/data_quick.yaml")

# Option 2: Chemin absolu (si vous avez cloné le repo)
training_config = load_training_config("/content/forestgaps-dl/configs/test/quick.yaml")
```

### Configs Personnalisées

Si vous voulez utiliser vos propres configs sur Colab:

```python
# 1. Créer un fichier YAML sur Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Charger depuis Google Drive
training_config = load_training_config("/content/drive/MyDrive/my_config.yaml")
```

## Setup Environnement

ForestGaps détecte automatiquement l'environnement Colab:

```python
from forestgaps.environment import setup_environment

# Setup automatique
env = setup_environment()

# Vérifier l'environnement
print(f"Environment: {env.name}")  # "colab"
print(f"GPU available: {env.device}")  # cuda ou cpu
```

## Workflow Complet sur Colab

### 1. Installation et Setup

```python
# Install
!pip install -q git+https://github.com/<username>/forestgaps-dl.git

# Imports
import forestgaps
from forestgaps.config import load_training_config, load_data_config, load_model_config
from forestgaps.environment import setup_environment
from forestgaps.models import create_model

# Setup
env = setup_environment()
print(f"✓ Environment: {env.name}, Device: {env.device}")
```

### 2. Configuration

```python
# Charger configs (quick test pour Colab)
training_config = load_training_config("configs/test/quick.yaml")
data_config = load_data_config("configs/test/data_quick.yaml")
model_config = load_model_config("configs/test/model_quick.yaml")

print(f"✓ Config loaded: {training_config.epochs} epochs, batch {training_config.batch_size}")
```

### 3. Données

```python
# Monter Google Drive pour accéder aux données
from google.colab import drive
drive.mount('/content/drive')

# Pointer vers vos données
data_dir = "/content/drive/MyDrive/forestgaps_data/Plot137"

# Créer dataloaders
from forestgaps.data.loaders import create_data_loaders

dataloaders = create_data_loaders(
    data_dir=data_dir,
    config=data_config,
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)

print(f"✓ Train: {len(dataloaders['train'])} batches")
print(f"✓ Val: {len(dataloaders['val'])} batches")
```

### 4. Modèle

```python
# Créer un modèle
model = create_model(
    model_config.model_type,
    in_channels=model_config.in_channels,
    out_channels=model_config.out_channels,
    init_features=model_config.base_channels
)

# Envoyer sur GPU
model = model.to(env.device)

print(f"✓ Model: {model.__class__.__name__}")
print(f"✓ Params: {sum(p.numel() for p in model.parameters()):,}")
```

### 5. Entraînement

```python
from forestgaps.training import Trainer

# Créer trainer
trainer = Trainer(
    model=model,
    config=training_config,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    test_loader=dataloaders['test'],
    device=env.device
)

# Train!
results = trainer.train(epochs=training_config.epochs)

print(f"✓ Training complete!")
print(f"  Best val loss: {results['best_val_loss']:.4f}")
```

## Troubleshooting

### Erreur: "Configuration file not found"

**Problème**: `FileNotFoundError: Configuration file not found: configs/test/quick.yaml`

**Solution**: Le loader cherche automatiquement dans le package. Vérifiez:
1. Le package est bien installé: `pip list | grep forestgaps`
2. Utilisez le chemin relatif: `"configs/test/quick.yaml"` (pas absolu)

Si l'erreur persiste, réinstallez:
```python
!pip uninstall -y forestgaps
!pip install --no-cache-dir git+https://github.com/<username>/forestgaps-dl.git
```

### Erreur: "No module named forestgaps"

**Solution**:
```python
!pip install git+https://github.com/<username>/forestgaps-dl.git
```

### GPU Non Détecté

**Vérifier**:
1. Runtime > Change runtime type > GPU
2. Redémarrer le runtime

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Erreur de Mémoire GPU

**Solutions**:
1. Réduire `batch_size` dans la config
2. Réduire `init_features` du modèle
3. Utiliser configs "minimal" ou "quick"

```python
# Réduire batch size
training_config.batch_size = 4  # au lieu de 8 ou 16

# Ou utiliser config minimal
training_config = load_training_config("configs/test/minimal.yaml")
```

## Exemples de Notebooks

### Quick Test (5 minutes)

```python
# Install & Setup
!pip install -q git+https://github.com/<username>/forestgaps-dl.git
from forestgaps.config import load_training_config, load_data_config
from forestgaps.models import create_model
import torch

# Load configs
cfg = load_training_config("configs/test/minimal.yaml")

# Create model
model = create_model("unet", in_channels=1, out_channels=1, init_features=16)

# Test forward pass
x = torch.randn(2, 1, 256, 256)
y = model(x)

print(f"✓ Input: {x.shape}")
print(f"✓ Output: {y.shape}")
print("✅ All working!")
```

### Full Training (30-60 minutes)

Voir `notebooks/Test_Package_ForestGaps.ipynb` pour l'exemple complet.

## Notes Importantes

1. **Configs incluses**: Toutes les configs (defaults, test, production) sont incluses dans le package
2. **Chemins relatifs**: Utilisez toujours des chemins relatifs comme `"configs/test/quick.yaml"`
3. **Google Drive**: Montez Drive pour accéder à vos données
4. **GPU**: Activez GPU dans Runtime settings pour un training rapide
5. **Mémoire**: Utilisez configs "minimal" ou "quick" pour éviter les OOM

## Support

- Issues GitHub: https://github.com/<username>/forestgaps-dl/issues
- Documentation: https://github.com/<username>/forestgaps-dl/tree/main/docs
