# 🚀 START HERE - ForestGaps Benchmarking

Guide ultra-simple pour lancer ton premier benchmark en 3 minutes.

## ✅ Prérequis (2 minutes)

### 1. Lancer Docker
```bash
cd docker/
docker-compose up -d
```

✅ Vérifier que ça tourne :
```bash
docker-compose ps
```

Tu dois voir 3 services "running" :
- `forestgaps-main`
- `forestgaps-tensorboard`
- `forestgaps-jupyter`

### 2. Entrer dans le container
```bash
docker exec -it forestgaps-main bash
```

Tu verras maintenant : `root@xxxxxxxx:/app#`

---

## 🎯 Lancer ton premier test (1 minute)

### Dans le container, copie-colle :
```bash
python scripts/benchmark_quick_test.py --experiment-name "test"
```

**Attendre 5-10 minutes...**

Tu verras :
- ✅ Création des DataLoaders
- ✅ Entraînement des modèles (U-Net, U-Net+FiLM)
- ✅ Évaluation et génération des rapports

---

## 📊 Voir les résultats

### Dans Windows (pas dans le container)

1. **Ouvrir l'explorateur** :
```
G:\Mon Drive\forestgaps-dl\outputs\benchmarks\
```

2. **Trouver ton benchmark** (le plus récent) :
```
20241203_HHMMSS_test\
```

3. **Ouvrir le rapport HTML** :
```
20241203_HHMMSS_test\reports\benchmark_report.html
```

**Double-clic pour ouvrir dans ton navigateur !**

### Sur TensorBoard

Ouvrir : http://localhost:6006

Tu verras :
- Courbes d'entraînement
- Métriques en temps réel
- Comparaison des modèles

---

## 🎓 Prochaines étapes

### Si le test rapide a marché :

#### Option 1 : Benchmark complet (4-8h)
```bash
# Dans le container
python scripts/benchmark_full.py --experiment-name "production"
```

#### Option 2 : Comparer plus de modèles
```bash
# Dans le container
python scripts/benchmark_quick_test.py \
  --experiment-name "comparaison_complete" \
  --epochs 10 \
  --models "unet,unet_film,deeplabv3_plus"
```

#### Option 3 : Personnaliser
```bash
# Dans le container
python scripts/benchmark_quick_test.py \
  --experiment-name "custom" \
  --epochs 20 \
  --batch-size 8 \
  --thresholds "2.0,5.0,10.0,15.0"
```

---

## 📚 Besoin d'aide ?

### Documentation complète
- **Guide benchmarking** : [docs/benchmarking/README.md](docs/benchmarking/README.md)
- **Commandes Docker** : [docs/benchmarking/COMMANDES_DOCKER.md](docs/benchmarking/COMMANDES_DOCKER.md)
- **Index docs** : [docs/README.md](docs/README.md)

### Problèmes courants

| Problème | Solution |
|----------|----------|
| Container n'existe pas | `docker-compose up -d` |
| Module not found | Dans container: `pip install -e .` |
| CUDA out of memory | Ajouter `--batch-size 2` |
| TensorBoard vide | Attendre 1-2 min, refresh |

### Commandes utiles

```bash
# Voir les logs du container
docker-compose logs -f forestgaps

# Vérifier le GPU
docker exec forestgaps-main nvidia-smi

# Lister les benchmarks
docker exec forestgaps-main ls -lhtr /app/outputs/benchmarks/

# Sortir du container
exit
```

---

## 🎉 C'est tout !

Tu sais maintenant :
- ✅ Lancer Docker
- ✅ Exécuter un benchmark
- ✅ Voir les résultats
- ✅ Personnaliser les paramètres

**Prêt pour le benchmarking ? Lance ta première commande ! 🚀**

---

**Questions ?** Consulte [docs/benchmarking/README.md](docs/benchmarking/README.md)
