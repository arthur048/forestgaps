# Résumé du module `models/`

## Description du module

Le module `models/` fournit un ensemble complet d'architectures de réseaux de neurones pour la segmentation d'images de télédétection forestière, avec un accent particulier sur la détection des trouées forestières. Ce module est conçu de manière modulaire, permettant de combiner différents blocs architecturaux, mécanismes d'attention et techniques de modulation pour créer des modèles adaptés à différents besoins.

## Architectures implémentées

### U-Net de base (`models/unet/basic.py`)

**Caractéristiques principales :**
- Architecture encodeur-décodeur symétrique
- Connexions de saut (skip connections) entre l'encodeur et le décodeur
- Profondeur configurable
- Support pour différentes couches de normalisation et fonctions d'activation

L'architecture U-Net de base est particulièrement efficace pour la segmentation d'images où les caractéristiques à différentes échelles sont importantes, comme dans la détection des trouées forestières de différentes tailles.

### ResUNet (`models/unet/residual.py`)

**Caractéristiques principales :**
- Intégration de blocs résiduels dans l'architecture U-Net
- Meilleure propagation du gradient
- Capacité à entraîner des réseaux plus profonds
- Amélioration des performances par rapport au U-Net standard

ResUNet combine les avantages de l'architecture U-Net avec ceux des connexions résiduelles, permettant un apprentissage plus stable et des performances améliorées.

### Attention U-Net (`models/unet/attention.py`)

**Caractéristiques principales :**
- Intégration de mécanismes d'attention dans l'architecture U-Net
- Support pour différents types d'attention (portes d'attention, CBAM)
- Capacité à se concentrer sur les régions pertinentes de l'image
- Particulièrement efficace pour les objets de petite taille

Attention U-Net améliore la capacité du modèle à se concentrer sur les caractéristiques importantes, ce qui est crucial pour la détection précise des petites trouées forestières.

### FiLM U-Net (`models/unet/film.py`)

**Caractéristiques principales :**
- Intégration de Feature-wise Linear Modulation (FiLM)
- Conditionnement du modèle en fonction de paramètres externes
- Adaptabilité à différentes conditions (saisons, types de forêts, etc.)
- Flexibilité accrue pour des applications spécifiques

FiLM U-Net permet d'adapter le comportement du modèle en fonction de paramètres externes, ce qui est utile pour prendre en compte des facteurs comme le type de forêt, la saison, ou d'autres métadonnées.

### UNet3+ (`models/unet/advanced.py`)

**Caractéristiques principales :**
- Connexions denses entre tous les niveaux d'encodeur et de décodeur
- Fusion multi-échelle des caractéristiques
- Option de supervision profonde pour améliorer l'apprentissage
- Performances supérieures pour les détails fins

UNet3+ représente l'état de l'art en matière de segmentation d'images, avec une architecture avancée qui capture efficacement les informations à toutes les échelles.

## Blocs architecturaux (`models/blocks/`)

### Blocs de convolution (`convolution.py`)

- `ConvBlock`: Bloc de convolution simple avec normalisation et activation
- `DoubleConvBlock`: Double bloc de convolution utilisé dans U-Net original

### Blocs de pooling (`pooling.py`)

- `DownsampleBlock`: Bloc de réduction de résolution (downsampling)
- `UpsampleBlock`: Bloc d'augmentation de résolution (upsampling)

### Blocs résiduels (`residual.py`)

- `ResidualBlock`: Bloc résiduel standard avec connexion de contournement
- `BottleneckBlock`: Bloc bottleneck avec trois couches de convolution (1x1, 3x3, 1x1)

## Mécanismes d'attention (`models/attention/`)

### CBAM (`cbam.py`)

- `CBAM`: Convolutional Block Attention Module complet
- `ChannelAttention`: Module d'attention des canaux
- `SpatialAttention`: Module d'attention spatiale

### Auto-attention (`self_attention.py`)

- `SelfAttention`: Module d'auto-attention pour capturer des dépendances à longue distance
- `PositionAttention`: Module d'attention de position pour les relations spatiales

## Feature-wise Linear Modulation (`models/film/`)

### Couches FiLM (`layers.py`)

- `FiLMLayer`: Couche de modulation linéaire des caractéristiques
- `FiLMGenerator`: Générateur de paramètres pour la modulation
- `AdaptiveFiLM`: Module FiLM adaptatif qui génère ses propres paramètres

### Blocs FiLM (`blocks.py`)

- `FiLMBlock`: Bloc de convolution avec modulation FiLM
- `FiLMResidualBlock`: Bloc résiduel avec modulation FiLM
- `ConditionedBlock`: Bloc conditionné par un vecteur externe

## Utilisation

### Exemple d'utilisation de U-Net de base

```python
import torch
from forestgaps.models import UNet

# Créer un modèle U-Net
model = UNet(
    in_channels=1,  # 1 canal pour les images DSM/CHM
    out_channels=1,  # 1 canal pour les masques binaires
    init_features=64,
    depth=4,
    dropout_rate=0.2
)

# Préparer les données
x = torch.randn(4, 1, 256, 256)  # Batch de 4 images 256x256

# Faire une prédiction
with torch.no_grad():
    y_pred = model(x)
```

### Exemple d'utilisation de FiLM U-Net

```python
import torch
from forestgaps.models import FiLMUNet

# Créer un modèle FiLM U-Net
model = FiLMUNet(
    in_channels=1,
    out_channels=1,
    init_features=64,
    depth=4,
    conditioning_size=10  # Taille du vecteur de conditionnement
)

# Préparer les données
x = torch.randn(4, 1, 256, 256)  # Batch de 4 images 256x256
conditioning = torch.randn(4, 10)  # Vecteurs de conditionnement

# Faire une prédiction
with torch.no_grad():
    y_pred = model(x, conditioning)
```

## Avantages et limitations

### Avantages

- **Modularité**: Les composants peuvent être combinés de différentes façons
- **Flexibilité**: Support pour différentes configurations et hyperparamètres
- **Performance**: Implémentations optimisées des architectures état de l'art
- **Extensibilité**: Facile à étendre avec de nouvelles architectures ou composants

### Limitations

- **Coût computationnel**: Les modèles avancés comme UNet3+ sont plus lourds
- **Besoins en données**: Les architectures complexes nécessitent plus de données d'entraînement
- **Hyperparamètres**: Le choix des hyperparamètres optimaux peut être difficile

## Prochaines améliorations

1. Intégration de techniques d'attention plus avancées (Transformer, etc.)
2. Support pour l'apprentissage auto-supervisé
3. Optimisation des performances pour les grands modèles
4. Intégration de techniques de quantification et de pruning pour le déploiement 