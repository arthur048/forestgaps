# Module de Modèles (models)

Ce module fournit les implémentations des différentes architectures de réseaux de neurones utilisées pour la détection des trouées forestières par segmentation et régression.

## Structure du module

```
models/
├── __init__.py               # Point d'entrée unifié et registre
├── base.py                   # Classes abstraites pour tous les modèles
├── registry.py               # Système de registre de modèles
├── unet/                     # Implémentations de U-Net
│   ├── __init__.py
│   ├── base_unet.py          # U-Net classique
│   ├── film_unet.py          # U-Net avec mécanisme FiLM
│   └── cbam_unet.py          # U-Net avec attention CBAM
├── deeplabv3/                # Implémentations de DeepLabV3+
│   ├── __init__.py
│   ├── deeplabv3_plus.py     # DeepLabV3+ standard
│   └── advanced_deeplab.py   # DeepLabV3+ avec améliorations
├── unet_regression/          # U-Net pour la régression
│   ├── __init__.py
│   ├── base_regressor.py     # Régresseur U-Net standard
│   └── ensemble.py           # Ensemble de régresseurs
├── blocks/                   # Blocs réutilisables
│   ├── __init__.py
│   ├── attention.py          # Mécanismes d'attention
│   ├── normalization.py      # Couches de normalisation
│   └── upsampling.py         # Méthodes d'upsampling
└── export/                   # Fonctionnalités d'export
    ├── __init__.py
    ├── onnx_export.py        # Export au format ONNX
    └── torchscript.py        # Export en TorchScript
```

## Fonctionnalités principales

### Système de registre

Ce module implémente un système de registre qui permet d'enregistrer et d'instancier des modèles dynamiquement. Cela facilite l'ajout de nouveaux modèles sans modifier le code existant.

```python
from forestgaps.models import model_registry, ForestGapModel

@model_registry.register("mon_nouveau_modele")
class MonNouveauModele(ForestGapModel):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Implémentation du modèle
```

### Classes de base

Les classes abstraites dans `base.py` fournissent les interfaces communes pour tous les modèles, assurant une cohérence dans l'implémentation :

- `ForestGapModel` : Classe de base pour tous les modèles
- `ThresholdConditionedModel` : Pour les modèles conditionnés par un seuil
- `UNetBaseModel` : Structure commune pour les variantes U-Net

### Modèles implémentés

#### U-Net

- **U-Net classique** : Implémentation de base pour la segmentation
- **U-Net FiLM** : U-Net avec Feature-wise Linear Modulation
- **U-Net CBAM** : U-Net avec Convolutional Block Attention Module

#### DeepLabV3+

- **DeepLabV3+ standard** : Implémentation de base
- **DeepLabV3+ avancé** : Variante avec améliorations (backbone modifiable, etc.)

#### U-Net pour régression

- **Régresseur U-Net** : Adaptation de U-Net pour la prédiction de hauteurs
- **Ensemble** : Combinaison de plusieurs modèles de régression

## Dépendances internes

- **config** : Pour charger les configurations spécifiques aux modèles
- **utils** : Pour les fonctionnalités communes, la gestion des erreurs, etc.

## Dépendances externes

- **PyTorch** : Framework de deep learning sous-jacent
- **torchvision** : Pour certains modèles backbone ou blocs
- **einops** : Pour des manipulations de tenseurs optimisées

## Utilisation

### Créer un modèle

```python
from forestgaps.models import create_model

# Créer un modèle U-Net standard
model = create_model("unet", in_channels=3, out_channels=1)

# Créer un modèle U-Net avec FiLM conditionné par un seuil
model = create_model("unet_film_threshold", in_channels=3, out_channels=1)

# Créer un modèle pour la régression
regressor = create_model("unet_regressor", in_channels=3, out_channels=1)
```

### Lister les modèles disponibles

```python
from forestgaps.models import list_available_models

# Obtenir la liste des modèles disponibles
models = list_available_models()
print(models)
```

### Créer un modèle à partir d'une configuration

```python
from forestgaps.config import load_default_config
from forestgaps.models import get_model_from_config

# Charger la configuration
config = load_default_config()

# Créer un modèle à partir de la configuration
model = get_model_from_config(config.MODELS)
```

## Principes de conception

Le module `models` suit les principes SOLID :

1. **Principe de responsabilité unique** : Chaque classe de modèle a une responsabilité spécifique.
2. **Principe ouvert/fermé** : Le système de registre permet d'ajouter de nouveaux modèles sans modifier le code existant.
3. **Principe de substitution de Liskov** : Tous les modèles héritent des classes abstraites et peuvent être utilisés de manière interchangeable.
4. **Principe de ségrégation d'interface** : Les interfaces sont spécifiques et minimales.
5. **Principe d'inversion de dépendance** : Le code dépend des abstractions, pas des implémentations concrètes.

## Contribution et extension

Pour ajouter un nouveau modèle :

1. Créer une nouvelle classe qui hérite de `ForestGapModel` ou d'une autre classe de base appropriée
2. Implémenter les méthodes requises (`forward`, etc.)
3. Décorer la classe avec `@model_registry.register("nom_du_modele")`
4. Importer la classe dans le fichier `__init__.py` du sous-module approprié

## Performance et benchmarking

Les performances des différents modèles peuvent être comparées à l'aide du module `benchmarking`, qui permet d'évaluer systématiquement les modèles sur différents ensembles de données et avec différentes configurations. 