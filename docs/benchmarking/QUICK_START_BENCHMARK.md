# Quick Start - Benchmarking ForestGaps

Guide de démarrage rapide pour lancer ton premier benchmark.

## ✅ Pré-requis

1. **Docker lancé** avec TensorBoard :
```bash
cd docker/
docker-compose up -d tensorboard
```

2. **Données présentes** :
```bash
# Vérifier les données d'entraînement
ls -lh data/*.tif | wc -l  # Devrait afficher ~16 fichiers

# Vérifier les données externes
ls -lh data/data_external_test/*.tif
```

3. **Accès TensorBoard** : http://localhost:6006

## 🚀 Lancer ton premier benchmark

### Option 1 : Test rapide (5-10 minutes)

```bash
# Entrer dans le container
cd docker/
docker-compose run --rm forestgaps bash

# Dans le container
python scripts/benchmark_quick_test.py \
  --experiment-name "mon_premier_test" \
  --epochs 5 \
  --models "unet,unet_film"
```

**Ce que ça fait :**
- Compare 2 modèles (U-Net et U-Net FiLM)
- 5 époques seulement
- 20 tuiles d'entraînement
- Seuils : 5m et 10m
- **Durée : ~5-10 minutes**

### Option 2 : Benchmark complet (plusieurs heures)

```bash
# Dans le container
python scripts/benchmark_full.py \
  --experiment-name "comparison_all_models" \
  --epochs 50 \
  --models "unet,unet_film,deeplabv3_plus,deeplabv3_plus_threshold" \
  --thresholds "2.0,5.0,10.0,15.0"
```

**Ce que ça fait :**
- Compare 4 modèles
- 50 époques
- Toutes les données
- 4 seuils de hauteur
- **Durée : ~4-8 heures** (selon GPU)

## 📊 Suivre l'entraînement

### TensorBoard (temps réel)
```
1. Ouvrir http://localhost:6006
2. Sélectionner l'expérience en cours
3. Voir les métriques en direct
```

### Logs Docker
```bash
# Logs en temps réel
docker-compose logs -f forestgaps

# Chercher les erreurs
docker-compose logs forestgaps | grep -i error
```

## 📁 Trouver les résultats

Après le benchmark, tout est organisé dans :

```
outputs/benchmarks/YYYYMMDD_HHMMSS_<experiment_name>/
├── benchmark_results.json          # Résultats agrégés
├── best_model.pt                   # Meilleur modèle
├── config.yaml                     # Configuration utilisée
├── models/                         # Détails par modèle
│   ├── UNet_Base/
│   │   ├── checkpoints/
│   │   │   ├── best.pt            # ⭐ Utiliser ce modèle
│   │   │   └── last.pt
│   │   └── metrics.json
│   ├── UNet_FiLM/
│   └── ...
├── visualizations/                 # Graphiques PNG
│   ├── metric_comparison_iou.png
│   ├── training_curves_iou.png
│   └── radar_chart.png
└── reports/                        # Rapports
    ├── benchmark_report.html       # 📄 Ouvrir en premier
    ├── benchmark_report.md
    └── benchmark_report.txt
```

## 🔍 Analyser les résultats

### 1. Rapport HTML (recommandé)
```bash
# Trouver le dernier benchmark
ls -lhtr outputs/benchmarks/ | tail -1

# Ouvrir le rapport
firefox outputs/benchmarks/<experiment_id>/reports/benchmark_report.html
```

### 2. Résultats JSON
```bash
# Voir le résumé
cat outputs/benchmarks/<experiment_id>/benchmark_results.json | jq '.summary'

# Meilleurs modèles
cat outputs/benchmarks/<experiment_id>/benchmark_results.json | jq '.best_models'
```

### 3. TensorBoard (analyse approfondie)
```
http://localhost:6006
- Comparer les courbes d'entraînement
- Voir les distributions de poids
- Analyser la convergence
```

## 🎯 Évaluer sur données externes

Après le benchmark, teste le meilleur modèle sur les données SODEFOR :

```bash
# Dans le container
python -m forestgaps.evaluation.external \
  --model outputs/benchmarks/<experiment_id>/best_model.pt \
  --dsm data/data_external_test/SODEFOR_Mini2_DSM.tif \
  --chm data/data_external_test/SODEFOR_Mini2_CHM.tif \
  --output outputs/external_eval/<experiment_id> \
  --visualize
```

## 🐛 Problèmes courants

### "No module named 'forestgaps'"
```bash
# Dans le container, installer en mode dev
pip install -e .
```

### "Out of memory"
```bash
# Réduire le batch size
python scripts/benchmark_quick_test.py --batch-size 2
```

### "CUDA out of memory"
```bash
# Vérifier l'utilisation GPU
docker-compose exec forestgaps nvidia-smi

# Si un autre process utilise le GPU, le tuer ou réduire batch_size
```

### TensorBoard ne s'affiche pas
```bash
# Redémarrer le service
docker-compose restart tensorboard

# Vérifier qu'il tourne
docker-compose ps tensorboard
```

## 📝 Commandes utiles

```bash
# Lister tous les benchmarks
ls -lhtr outputs/benchmarks/

# Voir la structure d'un benchmark
tree outputs/benchmarks/<experiment_id>/ -L 2

# Copier le meilleur modèle en production
cp outputs/benchmarks/<experiment_id>/best_model.pt models/production/unet_film_v1.pt

# Nettoyer les vieux logs (garder les 3 derniers)
cd logs/benchmarks && ls -t | tail -n +4 | xargs rm -rf

# Archiver une expérience
tar -czf archive_<experiment_id>.tar.gz outputs/benchmarks/<experiment_id>
```

## 💡 Conseils

1. **Commencer par un test rapide** pour valider le setup
2. **Surveiller TensorBoard** pendant l'entraînement
3. **Nommer clairement les expériences** (experiment-name descriptif)
4. **Sauvegarder les bons modèles** dans `models/production/`
5. **Archiver les expériences importantes** (tar.gz)

## 📚 Aller plus loin

- Lire [BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md) pour l'organisation complète
- Consulter [forestgaps/benchmarking/README.md](forestgaps/benchmarking/README.md) pour l'API
- Voir les exemples dans `examples/`

## ⚡ Commande ultime (tout-en-un)

```bash
# Lancer Docker + TensorBoard + Benchmark rapide
cd docker/ && \
docker-compose up -d tensorboard && \
docker-compose run --rm forestgaps python scripts/benchmark_quick_test.py \
  --experiment-name "test_$(date +%Y%m%d)" && \
echo "✅ Résultats dans : outputs/benchmarks/" && \
echo "📊 TensorBoard : http://localhost:6006"
```

Bonne chance ! 🚀
