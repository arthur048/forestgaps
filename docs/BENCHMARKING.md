# Benchmarking Guide

Guide complet pour benchmarker les modèles ForestGaps de manière reproductible et stable.

## Scripts Disponibles

### 1. `benchmark_stable.py` - Benchmark Reproductible ✅ RECOMMANDÉ

Script principal pour benchmarks stables avec seeds fixes et validation statistique.

**Features**:
- ✅ Fixed random seeds → reproductibilité garantie
- ✅ Multi-seed runs → validation statistique (mean ± std)
- ✅ Résultats sauvegardés (JSON + CSV)
- ✅ Comparaison multi-modèles robuste
- ✅ Disable AMP → stabilité maximale

**Usage**:
```bash
# Benchmark rapide (3 seeds, 5 epochs)
python scripts/benchmark_stable.py \
  --models unet film_unet deeplabv3_plus \
  --config configs/test/quick.yaml \
  --data-dir /data \
  --n-seeds 3 \
  --epochs 5 \
  --output-dir ./results/benchmark_quick

# Benchmark complet (5 seeds, 20 epochs)
python scripts/benchmark_stable.py \
  --models unet film_unet attention_unet deeplabv3_plus res_unet \
  --config configs/production/default.yaml \
  --data-dir /data \
  --n-seeds 5 \
  --epochs 20 \
  --output-dir ./results/benchmark_full
```

**Output Files**:
- `benchmark_results.json` - Full results with training history
- `benchmark_summary.csv` - Summary table (best_val_loss, epochs, etc.)

**Example Output**:
```
BENCHMARK SUMMARY (mean ± std across seeds)
============================================================

unet:
  Best val loss: 0.3245 ± 0.0023
  N runs: 3

film_unet:
  Best val loss: 0.3156 ± 0.0018
  N runs: 3

deeplabv3_plus:
  Best val loss: 0.3089 ± 0.0031
  N runs: 3
```

### 2. `benchmark_quick_test.py` - Tests Rapides

Script léger pour validation rapide (1 seed, peu d'epochs).

**Usage**:
```bash
python scripts/benchmark_quick_test.py \
  --experiment-name "test_run" \
  --models unet \
  --epochs 2 \
  --batch-size 4 \
  --max-train-tiles 20 \
  --max-val-tiles 5
```

## Best Practices

### 1. Reproductibilité

**✅ TOUJOURS utiliser `benchmark_stable.py` pour résultats publiables**

Pourquoi ?
- Seeds fixes → même résultat à chaque run
- Multi-seed → validation statistique
- AMP désactivé → pas de variations numériques

### 2. Nombre de Seeds

**Recommandations**:
- **Quick validation**: 3 seeds minimum
- **Paper results**: 5 seeds recommandé
- **High-stakes**: 10 seeds (très long)

**Pourquoi** ?
- N=3 → détection variations grossières (±20%)
- N=5 → estimation fiable (±10%)
- N=10 → haute précision (±5%)

### 3. Configurations

**Pour benchmarking**:

| Config | Epochs | Tiles | Use Case |
|--------|--------|-------|----------|
| `minimal.yaml` | 2 | 10 train, 5 val | Smoke test (30s) |
| `quick.yaml` | 5-10 | 50 train, 10 val | Fast validation (2-5 min/seed) |
| `production/default.yaml` | 20-50 | Full dataset | Publication (hours/seed) |

**Exemple workflow**:
```bash
# 1. Smoke test (verify code works)
python scripts/benchmark_stable.py \
  --models unet \
  --config configs/test/minimal.yaml \
  --n-seeds 1 \
  --epochs 2

# 2. Quick validation (compare models)
python scripts/benchmark_stable.py \
  --models unet film_unet deeplabv3_plus \
  --config configs/test/quick.yaml \
  --n-seeds 3 \
  --epochs 10

# 3. Full benchmark (for paper)
python scripts/benchmark_stable.py \
  --models unet film_unet attention_unet deeplabv3_plus res_unet \
  --config configs/production/default.yaml \
  --n-seeds 5 \
  --epochs 50
```

## Analyse des Résultats

### 1. Chargement des Résultats

```python
import pandas as pd
import json

# Charger summary CSV
df = pd.read_csv("benchmark_results/benchmark_summary.csv")

# Charger full JSON
with open("benchmark_results/benchmark_results.json") as f:
    results = json.load(f)
```

### 2. Analyse Statistique

```python
# Grouper par modèle
summary = df.groupby('model').agg({
    'best_val_loss': ['mean', 'std', 'min', 'max'],
    'best_epoch': 'mean'
})

print(summary)
```

### 3. Visualisation

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='model', y='best_val_loss')
plt.xticks(rotation=45)
plt.ylabel('Best Validation Loss')
plt.title('Model Comparison (N=3 seeds)')
plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=150)
```

## Checklist Publication

Avant de publier des résultats de benchmark:

- [ ] **Seeds fixes** → benchmark_stable.py utilisé
- [ ] **N ≥ 5 seeds** → validation statistique robuste
- [ ] **Config production** → dataset complet
- [ ] **≥ 20 epochs** → convergence atteinte
- [ ] **Résultats sauvegardés** → JSON + CSV disponibles
- [ ] **Statistiques calculées** → mean ± std reportés
- [ ] **Visualisations créées** → boxplots/curves disponibles

## Troubleshooting

### Problème: Résultats non reproductibles

**Solution**:
- Vérifier seeds fixes (utiliser `benchmark_stable.py`)
- Désactiver AMP (`use_amp=False`)
- Utiliser `torch.backends.cudnn.deterministic = True`

### Problème: Variance élevée entre seeds

**Causes possibles**:
- Trop peu d'epochs → pas de convergence
- Dataset trop petit → overfitting instable
- LR trop grand → training instable

**Solutions**:
- Augmenter epochs (≥20)
- Utiliser plus de données
- Réduire learning rate

### Problème: OOM (Out of Memory)

**Solutions**:
- Réduire `batch_size`
- Réduire `init_features` (32 → 16)
- Utiliser gradient accumulation

## Exemples Concrets

### Exemple 1: Comparaison Rapide UNet vs FiLM-UNet

```bash
python scripts/benchmark_stable.py \
  --models unet film_unet \
  --config configs/test/quick.yaml \
  --data-dir /data/Plot137 \
  --n-seeds 3 \
  --epochs 10 \
  --output-dir ./results/unet_vs_film
```

**Résultat attendu** (2-5 minutes):
```
unet:          0.3245 ± 0.0023
film_unet:     0.3156 ± 0.0018  ← Meilleur
```

### Exemple 2: Benchmark Complet Multi-Modèles

```bash
python scripts/benchmark_stable.py \
  --models unet film_unet attention_unet deeplabv3_plus res_unet \
  --config configs/production/default.yaml \
  --data-dir /data \
  --n-seeds 5 \
  --epochs 50 \
  --output-dir ./results/full_benchmark
```

**Résultat attendu** (plusieurs heures):
```
Model             Best Val Loss    Params
-----------------------------------------
unet              0.3245 ± 0.0023  7.8M
film_unet         0.3156 ± 0.0018  7.9M  ← Threshold aware
attention_unet    0.3198 ± 0.0027  8.9M  ← Attention gates
deeplabv3_plus    0.3089 ± 0.0031  15.2M ← Multi-scale
res_unet          0.3134 ± 0.0019  12.7M ← Residual
```

## Intégration CI/CD

Le benchmark stable peut être intégré dans GitHub Actions:

```yaml
# .github/workflows/benchmark.yml
name: Benchmark
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly Sunday 2am

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmark
        run: |
          python scripts/benchmark_stable.py \
            --models unet film_unet \
            --config configs/test/quick.yaml \
            --data-dir ${{ secrets.DATA_DIR }} \
            --n-seeds 3 \
            --epochs 10
```

## Références

- `scripts/benchmark_stable.py` - Script principal
- `scripts/benchmark_quick_test.py` - Tests rapides
- `configs/test/` - Configs de test
- `configs/production/` - Configs production
