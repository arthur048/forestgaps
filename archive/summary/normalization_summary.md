# Résumé du module `data/normalization/`

## Description du module

Le module `data/normalization/` est responsable des techniques avancées de normalisation des données pour la détection de trouées forestières. Il fournit des fonctionnalités complètes pour calculer, stocker et appliquer différentes stratégies de normalisation aux données, permettant d'adapter la normalisation selon les caractéristiques des données et d'exporter les paramètres de normalisation avec les modèles.

## Fonctionnalités implémentées

### Sous-module `statistics.py`

**Classes et fonctions principales :**
- `NormalizationStatistics`: Classe pour calculer et gérer les statistiques de normalisation.
- `compute_normalization_statistics`: Fonction utilitaire pour le calcul des statistiques à partir de fichiers.
- `batch_compute_statistics`: Calcule des statistiques pour plusieurs répertoires d'un coup.

**Caractéristiques :**
- Calcul de statistiques détaillées (min, max, moyenne, médiane, percentiles, etc.)
- Échantillonnage configurable pour gérer de grands volumes de données
- Calcul d'histogrammes pour analyser la distribution des données
- Stockage et chargement des statistiques au format JSON
- Calcul par fichier ou global pour une analyse fine

### Sous-module `strategies.py`

**Classes et fonctions principales :**
- `NormalizationStrategy`: Interface abstraite pour toutes les stratégies de normalisation.
- `MinMaxNormalization`: Stratégie de normalisation min-max classique.
- `ZScoreNormalization`: Normalisation Z-score (standardisation) des données.
- `RobustNormalization`: Normalisation robuste basée sur les percentiles.
- `AdaptiveNormalization`: Stratégie qui choisit automatiquement la meilleure méthode.
- `BatchNormStrategy`: Stratégie utilisant la normalisation par batch de PyTorch.
- `create_normalization_strategy`: Fonction factory pour créer la stratégie appropriée.

**Caractéristiques :**
- Support de nombreuses méthodes de normalisation
- Adaptation au type de données (NumPy/PyTorch)
- Gestion des valeurs aberrantes
- Détection automatique de la meilleure stratégie
- Interface cohérente pour toutes les stratégies

### Sous-module `normalization.py`

**Classes et fonctions principales :**
- `NormalizationLayer`: Couche PyTorch encapsulant une stratégie de normalisation.
- `InputNormalization`: Module pour normaliser les entrées d'un modèle.
- `normalize_batch` / `denormalize_batch`: Fonctions utilitaires pour les batches.
- `create_normalization_layer`: Fonction factory pour créer facilement des couches.

**Caractéristiques :**
- Intégration transparente avec PyTorch
- Normalisation exportable avec les modèles
- Support de paramètres apprenables
- Flexibilité pour différentes dimensions de données
- Fonctions pratiques pour l'utilisation rapide

### Sous-module `io.py`

**Fonctions principales :**
- Fonctions de sauvegarde/chargement dans différents formats (JSON, pickle, CSV)
- `plot_stats_histogram`: Génère des histogrammes pour visualiser les distributions.
- `generate_stats_report`: Crée un rapport complet des statistiques.
- `compare_stats`: Aide à comparer visuellement différents ensembles de statistiques.
- `merge_stats`: Combine plusieurs ensembles de statistiques.
- `export_stats_to_onnx`: Exporte les paramètres de normalisation au format ONNX.

**Caractéristiques :**
- Visualisation riche des statistiques
- Conversion entre différents formats
- Génération de rapports détaillés
- Outils de comparaison et d'analyse
- Support pour l'exportation en production

## Architecture et design

Le module suit une architecture en couches avec une séparation claire des responsabilités :

1. **Couche de calcul et stockage des statistiques** (`statistics.py`) : Responsable du calcul et de la gestion des statistiques à partir des données brutes.

2. **Couche de stratégies de normalisation** (`strategies.py`) : Implémente les différentes méthodes de normalisation sous une interface commune.

3. **Couche d'intégration PyTorch** (`normalization.py`) : Fournit des classes qui s'intègrent directement dans les modèles PyTorch.

4. **Couche d'entrées/sorties** (`io.py`) : Gère la persistance, la visualisation et la conversion des statistiques.

Cette architecture modulaire permet une grande flexibilité, notamment :
- Ajouter facilement de nouvelles stratégies de normalisation
- Changer la méthode de normalisation sans modifier le reste du code
- Exporter les paramètres de normalisation avec les modèles
- Adapter dynamiquement la normalisation selon les caractéristiques des données

## Dépendances externes

Le module dépend des bibliothèques suivantes :
- `numpy`: Manipulation efficace des tableaux de données
- `torch`: Support d'intégration avec PyTorch
- `matplotlib`: Visualisation des distributions et statistiques
- `pandas`: Manipulation des données pour l'analyse et l'export
- `onnx` (optionnel): Export au format ONNX pour la production
- `rasterio`: Accès aux données raster géospatiales

## Exemple d'utilisation

```python
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import nn

from forestgaps.data.normalization import (
    NormalizationStatistics, compute_normalization_statistics,
    NormalizationLayer, AdaptiveNormalization, create_normalization_layer,
    plot_stats_histogram, generate_stats_report
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# 1. Calcul des statistiques à partir des données
file_paths = ["path/to/chm1.tif", "path/to/chm2.tif", "path/to/chm3.tif"]
output_path = "stats/chm_stats.json"

stats = compute_normalization_statistics(
    file_paths=file_paths,
    output_path=output_path,
    method="adaptive",
    sample_ratio=0.2,  # Utilise 20% des pixels pour le calcul
    compute_histogram=True
)

# 2. Visualisation des statistiques
fig = plot_stats_histogram(stats, "stats/histogram.png")
plt.show()

# Génération d'un rapport complet
report_files = generate_stats_report(
    stats=stats,
    output_dir="stats/reports/",
    prefix="chm_adaptive"
)

# 3. Création d'une couche de normalisation pour un modèle
norm_layer = create_normalization_layer(
    method="adaptive",
    stats=stats,
    trainable=False  # Les paramètres ne sont pas apprenables
)

# 4. Intégration dans un modèle PyTorch
class SegmentationModel(nn.Module):
    def __init__(self, norm_layer):
        super().__init__()
        self.normalize = norm_layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # ... autres couches ...
        
    def forward(self, x):
        # Normalise automatiquement les entrées
        x = self.normalize(x)
        x = self.conv1(x)
        x = self.relu(x)
        # ... autres opérations ...
        return x

# Création du modèle avec normalisation intégrée
model = SegmentationModel(norm_layer)

# 5. Sauvegarde des paramètres de normalisation séparément
norm_layer.save("model/normalization_params.json")

# Exemple de normalisation directe d'un batch
input_tensor = torch.randn(4, 1, 256, 256)  # [B, C, H, W]
normalized = norm_layer(input_tensor)
```

## Fonctionnalités avancées

### Normalisation adaptative

Le module offre une fonctionnalité unique de normalisation adaptative qui analyse automatiquement les caractéristiques des données pour choisir la méthode de normalisation la plus appropriée :

- Détection d'asymétrie pour choisir entre Z-score et normalisation robuste
- Analyse des ratios de plage pour détecter les distributions non uniformes
- Détection automatique des valeurs aberrantes pour une normalisation plus robuste

### Exportabilité

La normalisation est conçue pour être exportable avec les modèles :

- Sauvegarde des paramètres au format JSON pour une réutilisation facile
- Intégration avec ONNX pour le déploiement en production
- Modules PyTorch prêts à être intégrés dans n'importe quel modèle

### Visualisation et analyse

Des outils riches de visualisation et d'analyse sont fournis :

- Histogrammes avec superposition des statistiques clés
- Rapports détaillés incluant toutes les mesures importantes
- Outils de comparaison pour analyser plusieurs ensembles de données

## Améliorations futures

- Support de normalisation spécifique aux domaines géospatiaux (ex: normalisation basée sur l'altitude)
- Intégration de techniques de normalisation d'images plus avancées (ex: CLAHE)
- Support multi-GPU pour le calcul de statistiques sur de très grands datasets
- Techniques d'auto-calibration pour ajuster dynamiquement les paramètres de normalisation pendant l'entraînement 