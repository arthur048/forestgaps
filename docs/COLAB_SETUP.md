# Google Colab Setup Guide - ForestGaps

Guide complet pour utiliser ForestGaps sur Google Colab.

**Date:** 2025-12-03  
**Status:** ✅ Ready for testing

---

## Installation Rapide

### Option 1: Installation directe depuis GitHub

```python
!pip install git+https://github.com/arthur048/forestgaps.git
```

### Option 2: Installation avec dépendances complètes

```bash
!pip install torch torchvision rasterio geopandas scikit-learn matplotlib tensorboard pydantic pyyaml tqdm
!pip install git+https://github.com/arthur048/forestgaps.git
```

---

## Dépendances Requises

**Core:** torch, numpy, scipy, scikit-learn  
**Geospatial:** rasterio, geopandas  
**Utils:** pydantic, pyyaml, tqdm, matplotlib, tensorboard

Voir [TEST_Package_ForestGaps.ipynb](../TEST_Package_ForestGaps.ipynb) pour tests complets.

---

## Test Installation

```python
import forestgaps
from forestgaps.models import model_registry

print(f"✅ ForestGaps {forestgaps.__version__}")
print(f"✅ {len(model_registry.list_models())} modèles disponibles")
```

---

## Ressources

- **Notebook test:** [TEST_Package_ForestGaps.ipynb](../TEST_Package_ForestGaps.ipynb)
- **Documentation:** [README.md](../README.md)
- **GitHub:** https://github.com/arthur048/forestgaps
