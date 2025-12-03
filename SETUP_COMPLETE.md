# ✅ Setup Benchmarking Complet - ForestGaps

**Date** : 3 décembre 2024
**Statut** : Prêt pour utilisation

## 📋 Résumé

L'infrastructure de benchmarking pour ForestGaps est maintenant **complètement configurée** selon les meilleures pratiques du deep learning. Tout est prêt pour lancer tes comparaisons de modèles !

## 🎯 Ce qui a été mis en place

### 1. **Structure des répertoires organisée**

```
forestgaps-dl/
├── data/                          ✅ Données d'entraînement (16 plots UTM)
│   └── data_external_test/        ✅ Données externes (SODEFOR_Mini2)
├── outputs/                       ✅ Tous les résultats d'expériences
│   └── benchmarks/                ✅ Benchmarks organisés par timestamp
├── logs/                          ✅ Logs TensorBoard
│   └── benchmarks/                ✅ Logs organisés par expérience
├── models/                        ✅ Modèles sauvegardés
│   ├── production/                ✅ Modèles en production
│   └── archive/                   ✅ Anciens modèles archivés
├── scripts/                       ✅ Scripts de benchmarking
│   ├── benchmark_quick_test.py    ✅ Test rapide (5-10 min)
│   └── benchmark_full.py          ✅ Benchmark complet (4-8h)
├── docker/                        ✅ Configuration Docker
│   └── docker-compose.yml         ✅ TensorBoard + Jupyter + ForestGaps
└── examples/                      ✅ Exemples d'utilisation
```

### 2. **Documentation complète**

| Document | Description | Status |
|----------|-------------|--------|
| [QUICK_START_BENCHMARK.md](QUICK_START_BENCHMARK.md) | Guide de démarrage rapide | ✅ |
| [BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md) | Guide complet d'organisation | ✅ |
| [scripts/README.md](scripts/README.md) | Documentation des scripts | ✅ |
| [forestgaps/benchmarking/README.md](forestgaps/benchmarking/README.md) | API du module | ✅ |

### 3. **Scripts de benchmarking**

#### `benchmark_quick_test.py` ⚡
- Test rapide (5-10 minutes)
- 2 modèles par défaut
- 5 époques, 20 tuiles
- Parfait pour valider le setup

#### `benchmark_full.py` 🚀
- Benchmark complet (4-8 heures)
- Tous les modèles disponibles
- 50 époques par défaut
- Production-ready

### 4. **Infrastructure Docker**

```yaml
Services configurés :
✅ forestgaps      : Container principal
✅ tensorboard     : http://localhost:6006
✅ jupyter         : http://localhost:8888
```

**Volumes montés :**
- `data/` → `/app/data` (lecture/écriture)
- `outputs/` → `/app/outputs` (lecture/écriture)
- `logs/` → `/app/logs` (lecture/écriture)
- `models/` → `/app/models` (lecture/écriture)

### 5. **Système de benchmarking**

**Fonctionnalités :**
- ✅ Comparaison automatique de modèles
- ✅ Métriques complètes (IoU, F1, Precision, Recall)
- ✅ Évaluation multi-seuils (2m, 5m, 10m, 15m)
- ✅ Visualisations automatiques (PNG)
- ✅ Rapports détaillés (HTML, MD, TXT)
- ✅ Logs TensorBoard en temps réel
- ✅ Sauvegarde des meilleurs modèles
- ✅ Exemples de prédictions sauvegardés

**Modèles disponibles :**
1. U-Net Base
2. U-Net FiLM
3. DeepLabV3+ Base
4. DeepLabV3+ Threshold

## 🚀 Comment démarrer MAINTENANT

### Étape 1 : Lancer Docker + TensorBoard (1 minute)

```bash
cd "g:\Mon Drive\forestgaps-dl\docker"
docker-compose up -d tensorboard

# Vérifier que c'est lancé
docker-compose ps
```

✅ Ouvrir TensorBoard : http://localhost:6006

### Étape 2 : Test rapide (5-10 minutes)

```bash
# Dans le même terminal
docker-compose run --rm forestgaps python scripts/benchmark_quick_test.py \
  --experiment-name "premier_test"
```

**Pendant l'exécution :**
- Surveiller les logs dans le terminal
- Voir les métriques en temps réel sur TensorBoard
- Le script affiche la progression

**Quand c'est terminé :**
```bash
# Voir les résultats
ls -lhtr outputs/benchmarks/

# Ouvrir le rapport HTML (remplacer <timestamp> par celui affiché)
explorer.exe "outputs\benchmarks\<timestamp>_premier_test\reports\benchmark_report.html"
```

### Étape 3 : Analyser les résultats

Le rapport HTML contient :
- 📊 Comparaison des métriques
- 📈 Courbes d'apprentissage
- 🏆 Classement des modèles
- ⏱️ Temps d'entraînement
- 📉 Vitesse de convergence
- 🎯 Performance par seuil

## 📊 Structure d'un résultat de benchmark

```
outputs/benchmarks/20241203_105530_premier_test/
├── benchmark_results.json          # Résultats complets (JSON)
├── best_model.pt                   # Meilleur modèle global
├── config.yaml                     # Configuration utilisée
├── models/
│   ├── UNet_Base/
│   │   ├── checkpoints/
│   │   │   ├── best.pt            ⭐ Meilleur checkpoint
│   │   │   └── last.pt
│   │   ├── metrics.json
│   │   └── prediction_examples/
│   └── UNet_FiLM/
│       └── ...
├── visualizations/                 # Tous les graphiques PNG
│   ├── metric_comparison_iou.png
│   ├── training_curves_iou.png
│   └── radar_chart.png
└── reports/
    ├── benchmark_report.html      📄 OUVRIR EN PREMIER
    ├── benchmark_report.md
    └── benchmark_report.txt
```

## 🎓 Workflow recommandé

### Phase 1 : Validation (AUJOURD'HUI)
```bash
# 1. Test rapide (5-10 min)
docker-compose run --rm forestgaps python scripts/benchmark_quick_test.py \
  --experiment-name "test_$(date +%Y%m%d)"

# 2. Vérifier que tout fonctionne
# 3. Analyser les résultats dans le rapport HTML
```

### Phase 2 : Benchmark complet (DEMAIN)
```bash
# Lancer pendant la nuit ou le week-end
docker-compose run --rm forestgaps python scripts/benchmark_full.py \
  --experiment-name "comparison_all_models_v1" \
  --epochs 50 \
  --batch-size 8
```

### Phase 3 : Évaluation externe
```bash
# Évaluer le meilleur modèle sur données SODEFOR
python -m forestgaps.evaluation.external \
  --model outputs/benchmarks/<exp_id>/best_model.pt \
  --dsm data/data_external_test/SODEFOR_Mini2_DSM.tif \
  --chm data/data_external_test/SODEFOR_Mini2_CHM.tif \
  --output outputs/external_eval/<exp_id> \
  --visualize
```

### Phase 4 : Production
```bash
# Copier le meilleur modèle en production
cp outputs/benchmarks/<exp_id>/best_model.pt \
   models/production/unet_film_v1_$(date +%Y%m%d).pt

# Créer un fichier de métadonnées
cat > models/production/metadata.json <<EOF
{
  "model": "UNet_FiLM",
  "experiment": "<exp_id>",
  "date": "$(date -I)",
  "metrics": {
    "iou": 0.XX,
    "f1": 0.XX
  },
  "training": {
    "epochs": 50,
    "data": "UTM plots"
  }
}
EOF
```

## 🛠️ Commandes utiles

### Docker
```bash
# Démarrer TensorBoard
docker-compose up -d tensorboard

# Voir les logs
docker-compose logs -f forestgaps

# Shell dans le container
docker-compose run --rm forestgaps bash

# Arrêter tout
docker-compose down

# Vérifier GPU
docker-compose exec forestgaps nvidia-smi
```

### Benchmarking
```bash
# Lister les benchmarks
ls -lhtr outputs/benchmarks/

# Voir les résultats d'un benchmark
cat outputs/benchmarks/<exp_id>/benchmark_results.json | jq '.summary'

# Comparer deux expériences
diff <(cat outputs/benchmarks/<exp1>/benchmark_results.json | jq '.summary') \
     <(cat outputs/benchmarks/<exp2>/benchmark_results.json | jq '.summary')

# Nettoyer les vieux logs (garder les 5 derniers)
cd logs/benchmarks && ls -t | tail -n +6 | xargs rm -rf
```

### Monitoring
```bash
# Utilisation GPU en temps réel
watch -n 1 'docker-compose exec forestgaps nvidia-smi'

# Utilisation mémoire container
docker stats forestgaps-main

# Espace disque
du -sh outputs/ logs/ models/
```

## 📈 Métriques à surveiller

### TensorBoard (temps réel)
- **Train/Val IoU** : Doit converger vers 0.7-0.9
- **Train/Val Loss** : Doit décroître régulièrement
- **Learning Rate** : Vérifier le schedule

### Rapport final
- **IoU moyen** : >0.75 = bon, >0.85 = excellent
- **F1-Score** : >0.80 = bon, >0.90 = excellent
- **Temps d'entraînement** : Comparer l'efficacité
- **Convergence** : Nombre d'époques pour atteindre 90% de perf max

## 🐛 Troubleshooting rapide

| Problème | Solution |
|----------|----------|
| CUDA out of memory | `--batch-size 2` ou `4` |
| TensorBoard vide | Attendre 1-2 min, refresh |
| Container crash | `docker-compose logs forestgaps` |
| Pas de GPU | Vérifier `nvidia-smi` |
| Import error | `pip install -e .` dans container |

## 📚 Prochaines étapes

1. ✅ **Aujourd'hui** : Lancer le test rapide
2. 🔄 **Demain** : Benchmark complet (4-8h)
3. 📊 **Après** : Analyser les résultats
4. 🎯 **Ensuite** : Évaluation externe
5. 🚀 **Final** : Modèle en production

## 🎉 Tu es prêt !

Tout est configuré selon les **meilleures pratiques du deep learning** :
- ✅ Reproductibilité (seed + config sauvegardée)
- ✅ Traçabilité (timestamps + logs complets)
- ✅ Monitoring (TensorBoard temps réel)
- ✅ Rapports automatiques (HTML + visualisations)
- ✅ Archivage organisé (structure claire)
- ✅ Documentation complète

**Lance ta première commande maintenant** :
```bash
cd "g:\Mon Drive\forestgaps-dl\docker"
docker-compose up -d tensorboard && \
docker-compose run --rm forestgaps python scripts/benchmark_quick_test.py \
  --experiment-name "test_initial"
```

Bon benchmark ! 🚀🌲

---

**Contact et support :**
- Documentation : Voir les fichiers README.md
- Issues : GitHub du projet
- TensorBoard : http://localhost:6006
