# Organisation des Données sur Google Drive - ForestGaps

Ce guide explique comment organiser vos données DSM/CHM sur Google Drive pour utiliser ForestGaps sur Colab.

## 📁 Structure Recommandée sur Google Drive

```
MyDrive/
└── ForestGaps_DeepLearning/          # Répertoire principal
    ├── raw_data/                     # Vos fichiers DSM/CHM originaux
    │   ├── site1_DSM.tif
    │   ├── site1_CHM.tif
    │   ├── site2_DSM.tif
    │   ├── site2_CHM.tif
    │   ├── site3_DSM.tif
    │   ├── site3_CHM.tif
    │   ├── site4_DSM.tif            # Paire de test
    │   └── site4_CHM.tif
    │
    ├── prepared_data/                # Données préparées (générées)
    │   ├── processed/
    │   ├── masks/
    │   ├── tiles/
    │   └── data_config.yaml
    │
    ├── models/                       # Modèles entraînés
    │   ├── unet_model.pt
    │   └── checkpoints/
    │
    └── outputs/                      # Résultats d'évaluation
        ├── predictions/
        └── metrics/
```

## 📊 Combien de Paires DSM/CHM Faut-il ?

### Configuration Minimale
- **Minimum absolu**: 2 paires (1 train + 1 test)
- **Recommandé**: 4 paires (3 train + 1 test)
- **Optimal**: 6+ paires (5 train + 1 test)

### Pourquoi ?

| Nombre de paires | Entraînement | Test | Qualité attendue |
|------------------|--------------|------|------------------|
| 2 paires         | 1            | 1    | ⚠️ Risque de sur-apprentissage |
| 4 paires (défaut)| 3            | 1    | ✓ Bon pour tests initiaux |
| 6 paires         | 5            | 1    | ✓✓ Bonne généralisation |
| 10+ paires       | 8-9          | 1-2  | ✓✓✓ Excellente généralisation |

**Note importante**: Le script génère **des centaines de tuiles** à partir de chaque paire DSM/CHM, donc même avec 4 paires vous aurez assez de données pour entraîner.

## 📝 Convention de Nommage des Fichiers

### Format Requis

Vos fichiers **doivent contenir** `DSM` ou `CHM` dans leur nom pour être détectés automatiquement:

#### ✅ Noms Valides
```
site1_DSM.tif + site1_CHM.tif
Site1_dsm.tif + Site1_chm.tif
foret_nord_DSM.TIF + foret_nord_CHM.TIF
20230515_DSM_parcelle3.tif + 20230515_CHM_parcelle3.tif
```

#### ❌ Noms Invalides
```
site1_surface.tif  (pas de DSM/CHM)
site1_hauteur.tif  (pas de DSM/CHM)
dsm_site1.tif      (DSM après le nom - peut fonctionner mais non recommandé)
```

### Règles de Correspondance

Le script apparie automatiquement les fichiers par leur **préfixe commun**:

- `site1_DSM.tif` ↔ `site1_CHM.tif` → Paire `site1`
- `foret_nord_DSM.tif` ↔ `foret_nord_CHM.tif` → Paire `foret_nord`
- `A123_DSM_v2.tif` ↔ `A123_CHM_v2.tif` → Paire `a123`

**Astuce**: Utilisez le même préfixe pour les deux fichiers d'une paire.

## 📐 Caractéristiques des Fichiers DSM/CHM

### Format
- **Extension**: `.tif` ou `.tiff` (GeoTIFF)
- **Bandes**: 1 bande (grayscale)
- **Type**: Float32 ou Int16
- **Système de coordonnées**: WGS84, Lambert93, ou tout CRS valide
- **Nodata**: Valeur nodata définie dans les métadonnées

### Résolution
- **Recommandé**: 0.5m - 2m par pixel
- **Minimum**: 0.25m par pixel
- **Maximum**: 5m par pixel

### Taille
- **Minimum**: 512×512 pixels
- **Recommandé**: 2000×2000 pixels ou plus
- **Maximum**: Aucune limite (le script crée des tuiles de 256×256)

### Alignement
- **DSM et CHM doivent couvrir la même zone**
- Si non alignés, le script les aligne automatiquement
- Le DSM est utilisé comme référence pour l'alignement

## 🚀 Utilisation du Script de Préparation

### Sur Google Colab

```python
# 1. Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Définir les chemins
RAW_DATA_DIR = '/content/drive/MyDrive/ForestGaps_DeepLearning/raw_data'
OUTPUT_DIR = '/content/drive/MyDrive/ForestGaps_DeepLearning/prepared_data'

# 3. Exécuter le script de préparation
!python /content/forestgaps/scripts/prepare_training_data.py \
    --data-dir "{RAW_DATA_DIR}" \
    --output-dir "{OUTPUT_DIR}" \
    --thresholds 2.0 5.0 10.0 \
    --tile-size 256 \
    --overlap 0.2 \
    --n-train 3
```

### Paramètres Expliqués

| Paramètre | Description | Valeur par défaut | Recommandation |
|-----------|-------------|-------------------|----------------|
| `--data-dir` | Répertoire des DSM/CHM bruts | **Requis** | Chemin Drive |
| `--output-dir` | Répertoire de sortie | **Requis** | Chemin Drive |
| `--thresholds` | Seuils de hauteur (m) | `2.0 5.0 10.0` | Garder défaut |
| `--tile-size` | Taille tuiles (pixels) | `256` | 256 ou 512 |
| `--overlap` | Chevauchement (0-0.5) | `0.2` | 0.1 à 0.3 |
| `--n-train` | Nombre paires train | `3` | 3 à 5 |

### Exemples de Commandes

#### Configuration standard (4 paires)
```bash
python scripts/prepare_training_data.py \
    --data-dir ./raw_data \
    --output-dir ./prepared_data \
    --n-train 3
```

#### Petites tuiles pour GPU limité
```bash
python scripts/prepare_training_data.py \
    --data-dir ./raw_data \
    --output-dir ./prepared_data \
    --tile-size 128 \
    --n-train 3
```

#### Plus de données d'entraînement (6 paires)
```bash
python scripts/prepare_training_data.py \
    --data-dir ./raw_data \
    --output-dir ./prepared_data \
    --n-train 5
```

## 📤 Étapes de Préparation des Données

Le script automatise **5 étapes** :

### 1️⃣ Détection des Paires
```
Recherche des paires DSM/CHM dans raw_data/
  ✓ Paire trouvée: site1
  ✓ Paire trouvée: site2
  ✓ Paire trouvée: site3
  ✓ Paire trouvée: site4

4 paires trouvées
```

### 2️⃣ Séparation Train/Test
```
Séparation: 3 train, 1 test
Train: site1, site2, site3
Test: site4
```

### 3️⃣ Alignement des Rasters
```
Traitement de site1 (train)
  ✓ DSM et CHM déjà alignés
Traitement de site2 (train)
  ⚙ Alignement nécessaire: reproject, resample
  ✓ Rasters alignés
```

### 4️⃣ Génération des Masques
```
Génération des masques pour site1
  ✓ Masque généré: seuil 2.0m
  ✓ Masque généré: seuil 5.0m
  ✓ Masque généré: seuil 10.0m
```

### 5️⃣ Création des Tuiles
```
Création des tuiles pour site1
  352 tuiles potentielles
  ✓ 298 tuiles créées
```

## 📂 Structure de Sortie Générée

Après exécution du script:

```
prepared_data/
├── processed/                    # Rasters alignés
│   ├── train/
│   │   ├── site1/
│   │   │   ├── site1_DSM.tif
│   │   │   └── site1_CHM.tif
│   │   ├── site2/
│   │   └── site3/
│   └── test/
│       └── site4/
│
├── masks/                        # Masques de trouées
│   ├── train/
│   │   ├── site1/
│   │   │   ├── site1_mask_2.0m.tif
│   │   │   ├── site1_mask_5.0m.tif
│   │   │   └── site1_mask_10.0m.tif
│   │   └── ...
│   └── test/
│
├── tiles/                        # Tuiles pour entraînement
│   ├── train/
│   │   ├── site1_tile_0000_dsm.tif
│   │   ├── site1_tile_0000_mask_2.0m.tif
│   │   ├── site1_tile_0000_mask_5.0m.tif
│   │   ├── site1_tile_0001_dsm.tif
│   │   └── ... (centaines de tuiles)
│   └── test/
│       └── ... (tuiles de test)
│
└── data_config.yaml              # Configuration générée
```

## ⚙️ Fichier de Configuration Généré

Le script crée `data_config.yaml` avec toutes les infos nécessaires:

```yaml
data:
  tiles_dir: /path/to/prepared_data/tiles
  train_dir: /path/to/prepared_data/tiles/train
  test_dir: /path/to/prepared_data/tiles/test
  tile_size: 256
  thresholds: [2.0, 5.0, 10.0]
  n_train_sites: 3
  n_test_sites: 1
  train_sites: [site1, site2, site3]
  test_sites: [site4]

training:
  batch_size: 8
  num_workers: 4
  epochs: 50
  learning_rate: 0.001
```

**Utilisez ce fichier** pour configurer l'entraînement.

## 🔍 Vérification des Données

Après préparation, vérifiez:

### Comptage des Fichiers
```python
import os
from pathlib import Path

prepared_dir = Path('/content/drive/MyDrive/ForestGaps_DeepLearning/prepared_data')

# Compter tuiles train
train_tiles = list((prepared_dir / 'tiles' / 'train').glob('*_dsm.tif'))
print(f"Tuiles d'entraînement: {len(train_tiles)}")

# Compter tuiles test
test_tiles = list((prepared_dir / 'tiles' / 'test').glob('*_dsm.tif'))
print(f"Tuiles de test: {len(test_tiles)}")

# Vérifier config
config_path = prepared_dir / 'data_config.yaml'
print(f"Config existe: {config_path.exists()}")
```

### Visualisation d'une Tuile
```python
import rasterio
import matplotlib.pyplot as plt

# Lire une tuile DSM
with rasterio.open(train_tiles[0]) as src:
    dsm_data = src.read(1)

# Afficher
plt.imshow(dsm_data, cmap='terrain')
plt.colorbar(label='Élévation (m)')
plt.title('Exemple de tuile DSM')
plt.show()
```

## ⚠️ Problèmes Fréquents

### Aucune paire trouvée
**Problème**: Le script ne trouve pas de paires DSM/CHM.

**Solutions**:
1. Vérifiez que les fichiers contiennent `DSM` et `CHM` dans leur nom
2. Vérifiez l'extension (doit être `.tif` ou `.tiff`)
3. Vérifiez que les fichiers sont bien dans `raw_data/`

### Paire incomplète
**Problème**: `⚠ Paire incomplète pour site1: manque CHM`

**Solution**: Assurez-vous d'avoir un fichier DSM **et** un fichier CHM pour chaque site.

### Pas assez de paires
**Problème**: Seulement 2 paires trouvées alors que `--n-train 3`

**Solution**: Le script s'adapte automatiquement et utilise 1 pour train, 1 pour test.

### Erreur d'alignement
**Problème**: Erreur lors de l'alignement des rasters.

**Solutions**:
1. Vérifiez que les fichiers sont des GeoTIFF valides
2. Vérifiez que le CRS est défini
3. Essayez d'ouvrir les fichiers avec QGIS pour validation

### Mémoire insuffisante
**Problème**: `OutOfMemoryError` lors du tuilage.

**Solutions**:
1. Réduire `--tile-size` à 128
2. Traiter les paires une par une
3. Utiliser Colab Pro avec plus de RAM

## 📊 Estimation des Ressources

### Espace Disque Nécessaire

Pour **4 paires DSM/CHM** (chaque paire ~500 MB):

| Étape | Espace | Détails |
|-------|--------|---------|
| Fichiers bruts | ~2 GB | 4 paires × 500 MB |
| Rasters alignés | ~2 GB | Copies alignées |
| Masques | ~600 MB | 3 seuils × 4 paires |
| Tuiles | ~1-2 GB | 1000-1500 tuiles |
| **Total** | **~6 GB** | Estimation |

### Temps de Traitement (Colab)

| Étape | Temps par paire | Total (4 paires) |
|-------|-----------------|------------------|
| Alignement | 1-2 min | 4-8 min |
| Masques | 30 sec | 2 min |
| Tuilage | 2-3 min | 8-12 min |
| **Total** | **4-5 min** | **15-20 min** |

## 🎓 Workflow Complet

### 1. Upload des Données
```python
# Sur Colab, utiliser l'interface ou:
from google.colab import files
uploaded = files.upload()  # Sélectionner vos fichiers DSM/CHM
```

### 2. Préparation
```python
!python scripts/prepare_training_data.py \
    --data-dir /content/drive/MyDrive/ForestGaps_DeepLearning/raw_data \
    --output-dir /content/drive/MyDrive/ForestGaps_DeepLearning/prepared_data
```

### 3. Entraînement
```python
from forestgaps.training import Trainer
from forestgaps.data.loaders import create_data_loaders
from forestgaps.config import load_config

config = load_config('prepared_data/data_config.yaml')
loaders = create_data_loaders(config)
# ... (voir Phase 4 pour la suite)
```

## 📚 Ressources Complémentaires

- [README principal](../README.md)
- [Guide Colab](./COLAB_SETUP.md)
- [Documentation du script](../scripts/prepare_training_data.py)

## 💡 Conseils

1. **Commencez petit**: Testez avec 2-4 paires avant de traiter toutes vos données
2. **Sauvegardez sur Drive**: Toujours travailler depuis Google Drive pour la persistance
3. **Vérifiez visuellement**: Ouvrez quelques tuiles dans QGIS pour validation
4. **Surveillez l'espace**: Google Drive gratuit = 15 GB, surveillez votre usage
5. **Documentez vos sites**: Notez les caractéristiques de chaque site (forêt, saison, etc.)

## 🆘 Support

Si vous rencontrez des problèmes:
1. Vérifiez cette documentation
2. Consultez les logs du script pour les erreurs détaillées
3. Ouvrez une issue GitHub avec les logs
