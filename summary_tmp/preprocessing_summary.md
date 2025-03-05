# Résumé du module `data/preprocessing/`

## Description du module

Le module `data/preprocessing/` est responsable du prétraitement des données raster (DSM/CHM) pour la détection de trouées forestières. Il fournit des fonctionnalités pour l'analyse, l'alignement et la conversion des données avant leur utilisation pour l'entraînement des modèles.

## Fonctionnalités implémentées

### Sous-module `alignment.py`

**Fonctions principales :**
- `check_alignment`: Vérifie l'alignement entre un DSM et un CHM en comparant les métadonnées géospatiales (résolution, système de coordonnées, limites).
- `align_rasters`: Réaligne un raster source sur un raster cible en reprojetant les données pour qu'elles correspondent à la même grille géospatiale.

**Caractéristiques :**
- Vérification complète de la compatibilité des systèmes de coordonnées
- Gestion d'erreurs robuste et messages clairs
- Paramètres configurables pour les seuils de tolérance
- Journalisation détaillée des opérations

### Sous-module `analysis.py`

**Fonctions principales :**
- `verify_raster_integrity`: Vérifie l'intégrité d'un fichier raster en analysant ses métadonnées et données.
- `analyze_raster_pair`: Analyse une paire de rasters (DSM/CHM) pour déterminer leur compatibilité et caractéristiques communes.
- `calculate_raster_statistics`: Calcule des statistiques détaillées sur un raster (min, max, moyenne, médiane, etc.).

**Caractéristiques :**
- Analyse complète de la qualité des données
- Détection des anomalies et valeurs aberrantes
- Génération de rapport statistique détaillé
- Support pour différents formats de raster

### Sous-module `conversion.py`

**Fonctions principales :**
- `convert_to_numpy`: Convertit un raster en tableau NumPy avec options de normalisation et de remplissage des valeurs manquantes.
- `convert_to_geotiff`: Convertit un tableau NumPy en fichier GeoTIFF avec métadonnées géospatiales.
- `extract_raster_window`: Extrait une fenêtre (sous-région) spécifique d'un raster.
- `create_binary_mask`: Crée un masque binaire à partir d'un raster en fonction d'un seuil.

**Caractéristiques :**
- Préservation des métadonnées géospatiales
- Options flexibles de normalisation
- Gestion efficace de la mémoire pour les grands rasters
- Support pour différents types de donnée

## Architecture et design

Le module suit les principes SOLID et une approche modulaire:
- **S** (Responsabilité unique): Chaque fichier a une responsabilité claire et bien définie.
- **O** (Ouvert/fermé): Les fonctions sont conçues pour être extensibles sans modification du code existant.
- **L** (Substitution de Liskov): Les interfaces des fonctions sont cohérentes et suivent des conventions similaires.
- **I** (Ségrégation des interfaces): Les fonctionnalités sont regroupées de manière logique dans différents sous-modules.
- **D** (Inversion des dépendances): Les fonctions dépendent d'abstractions plutôt que d'implémentations concrètes.

## Dépendances externes

Le module dépend des bibliothèques suivantes:
- `rasterio`: Manipulation des données géospatiales raster
- `numpy`: Traitement efficace des tableaux de données
- `pathlib`: Gestion robuste des chemins de fichiers
- `logging`: Journalisation standardisée des événements

## Exemple d'utilisation

```python
import logging
from data.preprocessing import align_rasters, analyze_raster_pair, convert_to_numpy

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Chemins des fichiers DSM et CHM
dsm_path = "path/to/dsm.tif"
chm_path = "path/to/chm.tif"

# Analyse de la paire DSM/CHM
analysis_result = analyze_raster_pair(dsm_path, chm_path, check_data=True)
print(f"Les rasters sont compatibles: {analysis_result['compatible']}")

# Alignement des rasters si nécessaire
if not analysis_result['aligned']:
    aligned_chm_path = "path/to/aligned_chm.tif"
    align_rasters(chm_path, dsm_path, aligned_chm_path)
    chm_path = aligned_chm_path  # Utiliser le CHM aligné

# Conversion en tableaux NumPy
dsm_array, dsm_meta = convert_to_numpy(dsm_path, normalize=True)
chm_array, chm_meta = convert_to_numpy(chm_path, normalize=True)

print(f"Dimensions du DSM: {dsm_array.shape}")
print(f"Dimensions du CHM: {chm_array.shape}")
```

## Améliorations futures

- Optimisation des performances pour le traitement de très grands rasters
- Ajout de fonctionnalités pour la correction automatique des anomalies dans les données
- Support pour des formats de données supplémentaires (LAS, LAZ, etc.)
- Parallélisation des opérations de traitement intensives 