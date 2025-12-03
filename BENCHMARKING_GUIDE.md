# Guide d'organisation pour le Benchmarking ForestGaps

## 📋 Vue d'ensemble

Ce guide décrit l'organisation des fichiers, logs et outputs pour le benchmarking des modèles de détection de trouées forestières selon les meilleures pratiques du deep learning.

## 🗂️ Structure des répertoires

```
forestgaps-dl/
├── data/                              # Données d'entraînement
│   ├── UTM33S_Plot137_{DSM,CHM}.tif
│   ├── UTM34N_Plot119_{DSM,CHM}.tif
│   ├── ...
│   └── data_external_test/            # Données de test indépendantes
│       ├── SODEFOR_Mini2_DSM.tif
│       └── SODEFOR_Mini2_CHM.tif
│
├── outputs/                           # Tous les outputs d'expériences
│   └── benchmarks/                    # Benchmarks des modèles
│       └── YYYYMMDD_HHMMSS_<name>/   # Timestamp + nom de l'expérience
│           ├── config.yaml           # Configuration complète utilisée
│           ├── benchmark_results.json # Résultats agrégés
│           ├── models/               # Checkpoints par modèle
│           │   ├── UNet_Base/
│           │   │   ├── checkpoints/
│           │   │   │   ├── best.pt
│           │   │   │   ├── epoch_10.pt
│           │   │   │   └── last.pt
│           │   │   ├── metrics.json
│           │   │   ├── model_config.json
│           │   │   └── prediction_examples/
│           │   │       ├── example_0.npy
│           │   │       └── ...
│           │   ├── UNet_FiLM/
│           │   └── DeepLabV3+/
│           ├── visualizations/       # Graphiques comparatifs
│           │   ├── metric_comparison_iou.png
│           │   ├── threshold_comparison_iou.png
│           │   ├── training_curves_iou.png
│           │   ├── training_time_comparison.png
│           │   ├── convergence_speed_iou.png
│           │   └── radar_chart.png
│           └── reports/              # Rapports détaillés
│               ├── benchmark_report.html
│               ├── benchmark_report.md
│               └── benchmark_report.txt
│
├── logs/                              # Logs TensorBoard et training
│   └── benchmarks/                    # Logs de benchmarks
│       └── YYYYMMDD_HHMMSS_<name>/   # Correspondance avec outputs
│           ├── UNet_Base/
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           ├── UNet_FiLM/
│           └── DeepLabV3+/
│
├── models/                            # Modèles finaux sauvegardés
│   ├── production/                    # Modèles en production
│   │   ├── best_unet_film_v1.pt
│   │   └── metadata.json
│   └── archive/                       # Anciens modèles archivés
│       └── YYYYMMDD/
│
├── examples/                          # Scripts d'exemples
│   ├── run_benchmark.py
│   └── ...
│
└── scripts/                           # Scripts utilitaires
    ├── benchmark_quick_test.py        # Test rapide
    └── benchmark_full.py              # Benchmark complet
```

## 🚀 Workflow de benchmarking

### 1. Préparation des données

```bash
# Vérifier la structure des données
ls -lh data/*.tif | head -10
ls -lh data/data_external_test/*.tif
```

**Données disponibles :**
- **Training/Val/Test** : Plots UTM33S et UTM34N dans `data/`
- **Évaluation externe** : SODEFOR_Mini2 dans `data/data_external_test/`

### 2. Lancement du Docker avec TensorBoard

```bash
cd docker/
docker-compose up -d tensorboard

# Vérifier que TensorBoard tourne
# Accès via : http://localhost:6006
```

### 3. Exécution d'un benchmark

#### Option A : Test rapide (recommandé pour débuter)

```bash
# Dans le container Docker
docker-compose run --rm forestgaps python scripts/benchmark_quick_test.py \
  --experiment-name "test_3models" \
  --epochs 5 \
  --quick-mode
```

#### Option B : Benchmark complet

```bash
# Dans le container Docker
docker-compose run --rm forestgaps python scripts/benchmark_full.py \
  --experiment-name "comparison_all_models" \
  --epochs 50 \
  --batch-size 8 \
  --thresholds 2.0 5.0 10.0 15.0
```

### 4. Surveillance de l'entraînement

#### Via TensorBoard
```
Ouvrir : http://localhost:6006
- Logs en temps réel
- Courbes de métriques
- Comparaison entre modèles
```

#### Via logs Docker
```bash
# Suivre les logs
docker-compose logs -f forestgaps

# Logs d'un benchmark spécifique
tail -f logs/benchmarks/20241203_105530_test_3models/UNet_Base/train.log
```

### 5. Analyse des résultats

```bash
# Lister les benchmarks disponibles
ls -lhtr outputs/benchmarks/

# Examiner les résultats d'un benchmark
cd outputs/benchmarks/20241203_105530_test_3models/

# Voir les résultats agrégés
cat benchmark_results.json | jq '.summary'

# Voir le rapport HTML
firefox reports/benchmark_report.html  # ou chrome, edge, etc.
```

## 📊 Convention de nommage

### Expériences
```
YYYYMMDD_HHMMSS_<experiment_name>
Exemple : 20241203_105530_comparison_all_models
```

### Modèles
```
<architecture>_<variant>
Exemples :
  - UNet_Base
  - UNet_FiLM
  - DeepLabV3+_Base
  - DeepLabV3+_Threshold
```

### Checkpoints
```
best.pt              # Meilleur modèle (selon val_iou)
last.pt              # Dernier checkpoint
epoch_<N>.pt         # Checkpoint à l'époque N
```

## 🔍 Métriques suivies

### Métriques principales
- **IoU** (Intersection over Union) : Métrique principale de segmentation
- **F1-Score** : Harmonic mean de précision et recall
- **Precision** : Taux de vrais positifs
- **Recall** : Taux de détection

### Métriques secondaires
- **Training time** : Temps d'entraînement total
- **Convergence speed** : Nombre d'époques pour atteindre 90% de la meilleure performance
- **Inference time** : Temps de prédiction (à ajouter)

### Seuils de hauteur analysés
- **2.0m** : Petites trouées
- **5.0m** : Trouées moyennes
- **10.0m** : Grandes trouées
- **15.0m** : Très grandes trouées

## 🐛 Debugging et monitoring

### Vérifier l'utilisation GPU
```bash
# Dans le container
docker-compose exec forestgaps nvidia-smi

# En continu
docker-compose exec forestgaps watch -n 1 nvidia-smi
```

### Vérifier la mémoire
```bash
docker stats forestgaps-main
```

### Logs d'erreurs
```bash
# Erreurs Python
docker-compose logs forestgaps | grep -i error

# Erreurs CUDA
docker-compose logs forestgaps | grep -i cuda
```

## 📈 Bonnes pratiques

### 1. **Toujours nommer ses expériences**
```python
# BON
benchmark = ModelComparison(..., output_dir="outputs/benchmarks/20241203_comparison_film_variants")

# MAUVAIS
benchmark = ModelComparison(...)  # Nom auto-généré illisible
```

### 2. **Sauvegarder la configuration**
- La config complète est automatiquement sauvegardée dans `config.yaml`
- Permet de reproduire exactement l'expérience

### 3. **Utiliser le mode quick pour tester**
```python
# Test rapide avant un long entraînement
config.data.max_train_tiles = 20
config.data.max_val_tiles = 5
config.training.epochs = 5
```

### 4. **Suivre l'entraînement en temps réel**
- TensorBoard : métriques et courbes
- Docker logs : progression détaillée
- `outputs/<experiment>/models/<model>/metrics.json` : métriques finales

### 5. **Évaluation externe systématique**
Après chaque benchmark, évaluer sur les données externes :
```python
from forestgaps.evaluation import ExternalEvaluator

evaluator = ExternalEvaluator(
    model_path="outputs/benchmarks/.../models/UNet_FiLM/checkpoints/best.pt"
)

results = evaluator.evaluate(
    dsm_path="data/data_external_test/SODEFER_Mini2_DSM.tif",
    chm_path="data/data_external_test/SODEFER_Mini2_CHM.tif",
    output_dir="outputs/external_eval/UNet_FiLM",
    visualize=True
)
```

## 🔄 Workflow complet recommandé

### Phase 1 : Test rapide
1. Lancer `benchmark_quick_test.py` avec 2-3 modèles
2. Vérifier que tout fonctionne (5-10 min)
3. Analyser les premiers résultats

### Phase 2 : Benchmark complet
1. Lancer `benchmark_full.py` avec tous les modèles (plusieurs heures)
2. Surveiller via TensorBoard
3. Sauvegarder les résultats

### Phase 3 : Évaluation externe
1. Évaluer le meilleur modèle sur données externes
2. Générer les visualisations
3. Créer le rapport final

### Phase 4 : Production
1. Copier le meilleur modèle dans `models/production/`
2. Documenter les performances dans `metadata.json`
3. Archiver l'expérience complète

## 📝 Notes importantes

- **Logs TensorBoard** : Partagent le même timestamp que outputs
- **Auto-cleanup** : Les checkpoints intermédiaires peuvent être nettoyés manuellement
- **Reproductibilité** : Seed fixé dans config pour résultats reproductibles
- **Backup** : Sauvegarder régulièrement `outputs/` et `models/production/`

## 🆘 Troubleshooting

### Problème : Out of memory
```python
# Réduire batch_size
config.training.batch_size = 4  # au lieu de 8

# Réduire taille des features
model_params["init_features"] = 16  # au lieu de 32
```

### Problème : TensorBoard ne s'affiche pas
```bash
# Redémarrer le service
docker-compose restart tensorboard

# Vérifier les logs
docker-compose logs tensorboard
```

### Problème : Checkpoint corrompu
```python
# Charger le dernier checkpoint valide
trainer.load_checkpoint("outputs/.../models/<model>/checkpoints/epoch_N.pt")
```
