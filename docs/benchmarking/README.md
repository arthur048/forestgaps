# Guide Benchmarking ForestGaps

Guide unique et complet pour lancer des benchmarks de modèles.

## 📋 Vue d'ensemble

Le système de benchmarking permet de comparer automatiquement plusieurs architectures de deep learning (U-Net, DeepLabV3+) sur la détection de trouées forestières.

**Outputs automatiques :**
- Métriques détaillées (IoU, F1, Precision, Recall)
- Visualisations comparatives (PNG)
- Rapports HTML/Markdown
- Logs TensorBoard temps réel
- Sauvegarde meilleurs modèles

## 🚀 Démarrage rapide

### 1. Lancer Docker
```bash
cd docker/
docker-compose up -d
```

### 2. Entrer dans le container
```bash
docker exec -it forestgaps-main bash
```

Tu verras : `root@xxxxxx:/app#`

### 3. Test rapide (5-10 min)
```bash
python scripts/benchmark_quick_test.py --experiment-name "test_$(date +%Y%m%d)"
```

### 4. Voir les résultats
```bash
# Dans le container
ls -lhtr outputs/benchmarks/

# Depuis Windows : ouvrir
# outputs\benchmarks\<timestamp>_test\reports\benchmark_report.html
```

## 📊 Scripts disponibles

### `benchmark_quick_test.py` - Test rapide
```bash
python scripts/benchmark_quick_test.py \
  --experiment-name "mon_test" \
  --epochs 5 \
  --models "unet,unet_film" \
  --batch-size 4
```

**Usage :** Valider le setup, tester une config
**Durée :** 5-10 minutes

### `benchmark_full.py` - Benchmark complet
```bash
python scripts/benchmark_full.py \
  --experiment-name "production_v1" \
  --epochs 50 \
  --models "unet,unet_film,deeplabv3_plus,deeplabv3_plus_threshold" \
  --batch-size 8
```

**Usage :** Comparaison complète pour publication
**Durée :** 4-8 heures

## 🎯 Modèles disponibles

| Modèle | Code | Description |
|--------|------|-------------|
| U-Net Base | `unet` | Architecture classique |
| U-Net FiLM | `unet_film` | Avec FiLM conditioning |
| DeepLabV3+ | `deeplabv3_plus` | State-of-the-art seg |
| DeepLabV3+ Threshold | `deeplabv3_plus_threshold` | Avec seuil encoding |

## 📁 Structure des outputs

```
outputs/benchmarks/20241203_105530_mon_test/
├── benchmark_results.json         # Résultats agrégés
├── best_model.pt                  # Meilleur modèle
├── config.yaml                    # Configuration
├── models/                        # Par modèle
│   ├── UNet_Base/
│   │   ├── checkpoints/
│   │   │   ├── best.pt           ⭐ Utiliser celui-ci
│   │   │   └── last.pt
│   │   └── metrics.json
│   └── UNet_FiLM/
│       └── ...
├── visualizations/                # Graphiques PNG
│   ├── metric_comparison_iou.png
│   ├── training_curves_iou.png
│   └── radar_chart.png
└── reports/
    ├── benchmark_report.html     📄 Ouvrir en premier
    ├── benchmark_report.md
    └── benchmark_report.txt
```

## 📈 Monitoring

### TensorBoard (temps réel)
```bash
# Accès depuis Windows
http://localhost:6006
```

Métriques visibles :
- Train/Val Loss
- Train/Val IoU, F1, Precision, Recall
- Learning rate schedule
- Distributions poids

### Logs Docker
```bash
# Suivre les logs
docker-compose logs -f forestgaps
```

## 🔧 Paramètres communs

```bash
--experiment-name "nom"      # Nom de l'expérience (REQUIS pour full)
--epochs N                   # Nombre d'époques (défaut: 5 ou 50)
--batch-size N               # Taille batch (défaut: 4 ou 8)
--models "m1,m2,m3"         # Modèles à comparer
--thresholds "2,5,10,15"    # Seuils hauteur (mètres)
--max-train-tiles N         # Limite tuiles (quick test only)
```

## 💡 Exemples pratiques

### Test ultra-rapide (2 min)
```bash
python scripts/benchmark_quick_test.py \
  --experiment-name "ultra_rapide" \
  --epochs 2 \
  --max-train-tiles 10 \
  --models "unet"
```

### Comparer U-Net vs U-Net+FiLM
```bash
python scripts/benchmark_quick_test.py \
  --experiment-name "unet_comparison" \
  --epochs 10 \
  --models "unet,unet_film"
```

### Benchmark production complet
```bash
python scripts/benchmark_full.py \
  --experiment-name "production_$(date +%Y%m%d)" \
  --epochs 50 \
  --batch-size 8
```

## 🐛 Troubleshooting

| Problème | Solution |
|----------|----------|
| `No module named 'forestgaps'` | `pip install -e .` dans container |
| CUDA out of memory | `--batch-size 2` |
| Container crash | `docker-compose logs forestgaps` |
| TensorBoard vide | Attendre 1-2 min, refresh |
| Pas de GPU visible | `nvidia-smi` dans container |

## 📚 Workflow recommandé

```
1. Test rapide (AUJOURD'HUI)
   └─> Valider setup (5-10 min)

2. Benchmark complet (DEMAIN)
   └─> Lancer pendant la nuit (4-8h)

3. Analyser résultats
   └─> Rapport HTML + TensorBoard

4. Évaluation externe
   └─> Tester sur SODEFOR_Mini2

5. Production
   └─> Sauvegarder dans models/production/
```

## 🔗 Ressources

- **TensorBoard** : http://localhost:6006
- **Jupyter Lab** : http://localhost:8888
- **Outputs** : `outputs/benchmarks/`
- **Logs** : `logs/benchmarks/`
- **API Docs** : `forestgaps/benchmarking/README.md`

## 📝 Notes importantes

- **Timestamp automatique** : Chaque expérience a un ID unique
- **Config sauvegardée** : Permet de reproduire exactement
- **Meilleur modèle** : Sauvegardé automatiquement
- **Rapports multi-formats** : HTML (principal), MD, TXT
- **Métriques par seuil** : 2m, 5m, 10m, 15m analysés séparément

## ⚡ Commande complète (copier-coller)

```bash
# Tout en un : Lancer Docker + Test rapide
cd docker/ && \
docker-compose up -d && \
sleep 5 && \
docker exec -it forestgaps-main bash -c \
  "python scripts/benchmark_quick_test.py --experiment-name test_rapide" && \
echo "✅ Terminé ! Voir outputs/benchmarks/"
```

## 🆘 Besoin d'aide ?

1. Vérifier les logs : `docker-compose logs forestgaps`
2. Consulter la section Troubleshooting ci-dessus
3. Vérifier GPU : `docker exec forestgaps-main nvidia-smi`
4. Vérifier données : `docker exec forestgaps-main ls /app/data/*.tif | wc -l`

---

**Prêt à lancer ton premier benchmark ? Commence par le test rapide ci-dessus ! 🚀**
