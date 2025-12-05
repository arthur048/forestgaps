# Testing Guide

Guide complet pour tester le code ForestGaps et vérifier la compatibilité backward.

## Vue d'ensemble

Le projet ForestGaps dispose de plusieurs niveaux de tests:

1. **Unit Tests** - Tests unitaires dans `tests/`
2. **CI/CD** - Tests automatiques via GitHub Actions
3. **Backward Compatibility** - Tests de compatibilité backward
4. **Benchmark Tests** - Tests de performance et stabilité

## 1. Unit Tests

### Structure

```
tests/
├── test_models.py           # Tests des architectures de modèles
├── test_config.py           # Tests de configuration
├── test_training.py         # Tests d'entraînement
├── test_inference.py        # Tests d'inférence
└── test_integration.py      # Tests d'intégration end-to-end
```

### Exécution

```bash
# Tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_models.py -v
pytest tests/test_models.py::TestModelCreation::test_unet_variants -v

# Avec couverture
pytest tests/ --cov=forestgaps --cov-report=html
```

### Couverture Actuelle

**test_models.py**:
- ✅ Création de tous les modèles (UNet, FiLM-UNet, AttentionUNet, DeepLabV3+, ResUNet)
- ✅ Forward pass avec validation de shape
- ✅ Gradient flow
- ✅ Save/load checkpoints

**test_config.py**:
- ✅ Chargement de configs YAML
- ✅ Validation Pydantic
- ✅ Merge de configs
- ✅ Détection d'erreurs

**test_training.py**:
- ✅ Loss functions (BCE, Dice, Focal, Combo)
- ✅ Training loops
- ✅ Callbacks
- ✅ LR scheduling

**test_inference.py**:
- ✅ Tiled processing
- ✅ Batch processing
- ✅ Geospatial metadata preservation

**test_integration.py**:
- ✅ Workflows end-to-end
- ✅ Train → Save → Load → Infer

## 2. CI/CD (GitHub Actions)

### Pipeline Actuel

Fichier: `.github/workflows/tests.yml`

**Tests Automatiques**:
- Multi-Python matrix (3.8, 3.9, 3.10, 3.11, 3.12)
- Installation GDAL pour support géospatial
- Tests pytest complets
- Smoke tests de création de modèles
- Test AttentionUNet forward pass
- Test training step
- Code quality (black, isort, flake8)

**Déclenchement**:
- Push sur `main` ou `develop`
- Pull requests vers `main` ou `develop`
- Manuel via `workflow_dispatch`

### Vérification du Statut

```bash
# Via GitHub CLI
gh run list --workflow=tests.yml

# Via GitHub Web
https://github.com/<user>/forestgaps-dl/actions
```

## 3. Backward Compatibility Tests

### Script Principal

**Fichier**: `scripts/test_backward_compatibility.py`

**Tests Inclus**:
1. **Model Creation** - Vérification que tous les modèles peuvent être créés
2. **Forward Pass** - Validation des shapes d'entrée/sortie
3. **Training Step** - Vérification du training loop (loss, optimizer, backprop)
4. **Config Loading** - Chargement de toutes les configs
5. **AttentionUNet Specific** - Tests spécifiques pour le modèle réparé

### Exécution

```bash
# Dans Docker
docker exec forestgaps-main python scripts/test_backward_compatibility.py

# Localement
python scripts/test_backward_compatibility.py
```

### Output Attendu

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FORESTGAPS BACKWARD COMPATIBILITY TESTS                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
  TEST 1: Model Creation
================================================================================
✓ unet                : 7,765,057 parameters
✓ film_unet           : 7,768,129 parameters
✓ attention_unet      : 8,942,337 parameters
✓ deeplabv3_plus      : 15,234,689 parameters
✓ res_unet            : 12,765,441 parameters
✓ regression_unet     : 7,765,057 parameters

================================================================================
  TEST 2: Model Forward Pass
================================================================================
✓ unet                : torch.Size([2, 1, 256, 256]) → torch.Size([2, 1, 256, 256])
✓ film_unet           : torch.Size([2, 1, 256, 256]) → torch.Size([2, 1, 256, 256])
✓ attention_unet      : torch.Size([2, 1, 256, 256]) → torch.Size([2, 1, 256, 256])
✓ deeplabv3_plus      : torch.Size([2, 1, 256, 256]) → torch.Size([2, 1, 256, 256])
✓ res_unet            : torch.Size([2, 1, 256, 256]) → torch.Size([2, 1, 256, 256])
✓ regression_unet     : torch.Size([2, 1, 256, 256]) → torch.Size([2, 1, 256, 256])

================================================================================
  TEST 3: Training Step
================================================================================
✓ Training step successful
  Loss: 0.8234
  BCE:   0.6934
  Dice:  0.4532
  Focal: 0.3256

================================================================================
  TEST 4: Configuration Loading
================================================================================
✓ configs/test/minimal.yaml   : epochs=2, batch_size=4
✓ configs/test/quick.yaml     : epochs=10, batch_size=8

================================================================================
  TEST 5: AttentionUNet Specific Tests
================================================================================
✓ AttentionUNet depth=3: torch.Size([1, 1, 128, 128]) → torch.Size([1, 1, 128, 128])
✓ AttentionUNet depth=4: torch.Size([1, 1, 128, 128]) → torch.Size([1, 1, 128, 128])
✓ AttentionUNet depth=5: torch.Size([1, 1, 128, 128]) → torch.Size([1, 1, 128, 128])
✓ AttentionUNet gradient flow: OK

================================================================================
SUMMARY
================================================================================

MODEL_CREATION:
  ✓ unet
  ✓ film_unet
  ✓ attention_unet
  ✓ deeplabv3_plus
  ✓ res_unet
  ✓ regression_unet

MODEL_FORWARD:
  ✓ unet
  ✓ film_unet
  ✓ attention_unet
  ✓ deeplabv3_plus
  ✓ res_unet
  ✓ regression_unet

TRAINING_STEP:
  ✓ training_step

CONFIG_LOADING:
  ✓ configs/test/minimal.yaml
  ✓ configs/test/quick.yaml

ATTENTION_UNET:
  ✓ depth_3
  ✓ depth_4
  ✓ depth_5
  ✓ gradient_flow

================================================================================
TOTAL: 20 tests
✓ PASSED: 20
✗ FAILED: 0
================================================================================
```

## 4. Benchmark Tests

### Scripts Disponibles

**`scripts/benchmark_stable.py`** - Production benchmarks
- Fixed random seeds
- Multi-seed runs (mean ± std)
- JSON + CSV output
- Statistiques robustes

**`scripts/benchmark_quick_test.py`** - Quick validation
- Single seed
- Peu d'epochs
- Tests rapides

### Voir aussi

- [docs/BENCHMARKING.md](BENCHMARKING.md) - Guide complet de benchmarking

## Stratégie de Test

### Avant un Commit

```bash
# 1. Tests unitaires rapides
pytest tests/ -v

# 2. Test backward compatibility
python scripts/test_backward_compatibility.py

# 3. Linting
black --check forestgaps/ tests/
isort --check-only forestgaps/ tests/
flake8 forestgaps/ tests/
```

### Avant un Release

```bash
# 1. Tests complets
pytest tests/ -v --cov=forestgaps

# 2. Backward compatibility
python scripts/test_backward_compatibility.py

# 3. Benchmark stability
python scripts/benchmark_stable.py \
  --models unet film_unet attention_unet \
  --config configs/test/quick.yaml \
  --n-seeds 3 \
  --epochs 10

# 4. Code quality
black forestgaps/ tests/
isort forestgaps/ tests/
flake8 forestgaps/ tests/
```

### CI/CD Automatique

Le pipeline GitHub Actions exécute automatiquement:
- ✅ Tests unitaires (multi-Python)
- ✅ Model creation smoke tests
- ✅ AttentionUNet verification
- ✅ Training step validation
- ✅ Code quality checks

## Résolution de Problèmes

### Tests Pytest Échouent

**Vérifier**:
1. Toutes les dépendances sont installées: `pip install -e ".[dev]"`
2. PyTorch version correcte: `pip install torch torchvision`
3. GDAL installé si tests géospatiaux: `conda install gdal`

### Backward Compatibility Fails

**Causes Communes**:
1. **Import Errors** - Vérifier que tous les modules sont importables
2. **Shape Mismatches** - Problème d'architecture de modèle
3. **Gradient Issues** - Problème de backprop

**Debug**:
```python
# Test import
python -c "from forestgaps.models import create_model; print('✓ Import OK')"

# Test model creation
python -c "
from forestgaps.models import create_model
model = create_model('attention_unet', in_channels=1, out_channels=1, init_features=16, depth=4)
print(f'✓ Model created: {sum(p.numel() for p in model.parameters()):,} params')
"

# Test forward
python -c "
import torch
from forestgaps.models import create_model
model = create_model('attention_unet', in_channels=1, out_channels=1, init_features=16, depth=4)
x = torch.randn(1, 1, 128, 128)
y = model(x)
print(f'✓ Forward OK: {x.shape} → {y.shape}')
"
```

### CI/CD Fails

**Vérifier**:
1. GitHub Actions logs: `.github/workflows/tests.yml`
2. Python version compatibility
3. Dependency installation
4. GDAL availability

## Références

- `tests/` - Tests unitaires
- `.github/workflows/tests.yml` - CI/CD pipeline
- `scripts/test_backward_compatibility.py` - Tests de compatibilité
- `docs/BENCHMARKING.md` - Guide de benchmarking
