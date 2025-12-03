# Scripts ForestGaps

Ce répertoire contient les scripts utilitaires pour l'entraînement, l'évaluation et le benchmarking des modèles ForestGaps.

## 📁 Structure

```
scripts/
├── README.md                    # Ce fichier
├── benchmark_quick_test.py      # Test rapide du benchmarking (5-10 min)
├── benchmark_full.py            # Benchmark complet (plusieurs heures)
├── docker-build.sh              # Construction de l'image Docker
├── docker-run.sh                # Lancement du container Docker
└── docker-test.sh               # Tests dans Docker
```

## 🎯 Scripts de Benchmarking

### `benchmark_quick_test.py`

Script de test rapide pour valider le pipeline de benchmarking.

**Usage :**
```bash
python scripts/benchmark_quick_test.py \
  --experiment-name "test_run" \
  --epochs 5 \
  --models "unet,unet_film"
```

**Paramètres principaux :**
- `--experiment-name` : Nom de l'expérience (requis)
- `--epochs` : Nombre d'époques (défaut: 5)
- `--batch-size` : Taille du batch (défaut: 4)
- `--max-train-tiles` : Nombre de tuiles d'entraînement (défaut: 20)
- `--models` : Modèles à comparer (défaut: "unet,unet_film")
- `--thresholds` : Seuils de hauteur (défaut: "5.0,10.0")

**Cas d'usage :**
- Tester rapidement une nouvelle configuration
- Valider que le pipeline fonctionne
- Débugger avant un long entraînement

**Durée estimée :** 5-10 minutes

---

### `benchmark_full.py`

Script de benchmarking complet pour comparer tous les modèles.

**Usage :**
```bash
python scripts/benchmark_full.py \
  --experiment-name "comparison_all_models" \
  --epochs 50 \
  --batch-size 8 \
  --models "unet,unet_film,deeplabv3_plus,deeplabv3_plus_threshold"
```

**Paramètres principaux :**
- `--experiment-name` : Nom de l'expérience (requis)
- `--epochs` : Nombre d'époques (défaut: 50)
- `--batch-size` : Taille du batch (défaut: 8)
- `--models` : Modèles à comparer (défaut: tous)
- `--thresholds` : Seuils de hauteur (défaut: "2.0,5.0,10.0,15.0")
- `--config` : Fichier de config personnalisé (optionnel)
- `--no-tensorboard` : Désactiver TensorBoard
- `--save-all-checkpoints` : Sauvegarder tous les checkpoints

**Cas d'usage :**
- Benchmark final pour la publication
- Comparaison exhaustive des architectures
- Expériences de recherche

**Durée estimée :** 4-8 heures (selon GPU et données)

---

## 🚀 Modèles disponibles

Les modèles suivants peuvent être spécifiés avec `--models` :

| Nom | Description | Paramètres |
|-----|-------------|------------|
| `unet` | U-Net de base | 32 features init |
| `unet_film` | U-Net avec FiLM | 32 features + FiLM |
| `deeplabv3_plus` | DeepLabV3+ base | ASPP + décodeur |
| `deeplabv3_plus_threshold` | DeepLabV3+ conditionné | ASPP + CBAM + encoding seuil |

## 📊 Outputs générés

Chaque benchmark crée une structure complète dans `outputs/benchmarks/` :

```
YYYYMMDD_HHMMSS_<experiment_name>/
├── config.yaml                    # Configuration complète
├── benchmark_results.json         # Résultats agrégés
├── best_model.pt                  # Meilleur modèle global
├── models/                        # Modèles individuels
│   ├── <ModelName>/
│   │   ├── checkpoints/
│   │   │   ├── best.pt           # Meilleur checkpoint
│   │   │   └── last.pt           # Dernier checkpoint
│   │   ├── metrics.json          # Métriques du modèle
│   │   ├── model_config.json     # Config du modèle
│   │   └── prediction_examples/  # Exemples de prédictions
├── visualizations/                # Graphiques comparatifs
│   ├── metric_comparison_*.png
│   ├── threshold_comparison_*.png
│   ├── training_curves_*.png
│   ├── training_time_comparison.png
│   ├── convergence_speed_*.png
│   └── radar_chart.png
└── reports/                       # Rapports détaillés
    ├── benchmark_report.html      # Rapport principal
    ├── benchmark_report.md        # Version Markdown
    └── benchmark_report.txt       # Version texte
```

## 🔍 Métriques calculées

### Par modèle et par seuil
- **IoU** : Intersection over Union
- **F1-Score** : Harmonic mean Precision/Recall
- **Precision** : Taux de vrais positifs
- **Recall** : Taux de détection

### Métriques globales
- **Moyennes** : Métriques moyennées sur tous les seuils
- **Temps d'entraînement** : Durée totale en secondes
- **Vitesse de convergence** : Époques pour atteindre 90% de perf max

### Classement
- Meilleur modèle par métrique
- Meilleur modèle par seuil
- Modèle le plus rapide
- Modèle le plus stable

## 💡 Exemples d'utilisation

### Test rapide avec 2 modèles
```bash
python scripts/benchmark_quick_test.py \
  --experiment-name "test_unet_vs_film" \
  --models "unet,unet_film" \
  --epochs 5
```

### Benchmark complet avec configuration personnalisée
```bash
python scripts/benchmark_full.py \
  --experiment-name "exp_custom_config" \
  --config config/custom.yaml \
  --epochs 100 \
  --batch-size 16
```

### Benchmark avec seuils spécifiques
```bash
python scripts/benchmark_full.py \
  --experiment-name "seuils_extrêmes" \
  --thresholds "1.0,2.0,20.0,30.0" \
  --epochs 50
```

### Benchmark sans TensorBoard (serveur sans GUI)
```bash
python scripts/benchmark_full.py \
  --experiment-name "server_run" \
  --no-tensorboard \
  --epochs 50
```

## 🐛 Debugging

### Mode verbose
```bash
# Ajouter avant la commande
export LOG_LEVEL=DEBUG
python scripts/benchmark_full.py ...
```

### Profiling mémoire
```bash
# Utiliser le profiler Python
python -m memory_profiler scripts/benchmark_full.py ...
```

### Dry run (vérifier sans exécuter)
Modifier temporairement `epochs=1` et `max_train_tiles=5` pour tester rapidement.

## 📚 Documentation associée

- [QUICK_START_BENCHMARK.md](../QUICK_START_BENCHMARK.md) : Guide de démarrage rapide
- [BENCHMARKING_GUIDE.md](../BENCHMARKING_GUIDE.md) : Guide complet d'organisation
- [forestgaps/benchmarking/README.md](../forestgaps/benchmarking/README.md) : Documentation de l'API

## ⚙️ Scripts Docker

### `docker-build.sh`
Construit l'image Docker ForestGaps.

```bash
bash scripts/docker-build.sh
```

### `docker-run.sh`
Lance le container avec les bons volumes montés.

```bash
bash scripts/docker-run.sh python scripts/benchmark_full.py ...
```

### `docker-test.sh`
Exécute les tests dans le container Docker.

```bash
bash scripts/docker-test.sh
```

## 🔐 Bonnes pratiques

1. **Toujours nommer les expériences** avec `--experiment-name`
2. **Commencer par un test rapide** avant le benchmark complet
3. **Surveiller TensorBoard** pendant l'entraînement
4. **Sauvegarder les configs** pour la reproductibilité
5. **Archiver les bons résultats** dans `models/production/`
6. **Documenter les expériences** dans un fichier EXPERIMENTS.md

## 🆘 Support

En cas de problème :
1. Vérifier les logs : `docker-compose logs forestgaps`
2. Consulter [BENCHMARKING_GUIDE.md](../BENCHMARKING_GUIDE.md) section Troubleshooting
3. Vérifier les issues GitHub du projet
