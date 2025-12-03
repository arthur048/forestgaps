# Documentation ForestGaps

Index central de toute la documentation du projet.

## 📚 Documentation principale

### [README.md](../README.md) - Guide principal
Vue d'ensemble du projet, installation, exemples d'utilisation

## 🔬 Benchmarking

### [Benchmarking Guide](benchmarking/README.md) - **COMMENCER ICI**
Guide complet pour comparer les modèles :
- Démarrage rapide (test en 5-10 min)
- Benchmark complet (production)
- Monitoring TensorBoard
- Analyse des résultats

**Pour plus de détails :**
- [Guide détaillé](benchmarking/BENCHMARKING_GUIDE.md) : Organisation, conventions, best practices
- [Commandes Docker](benchmarking/COMMANDES_DOCKER.md) : Toutes les commandes utiles
- [Quick Start](benchmarking/QUICK_START_BENCHMARK.md) : Démarrage ultra-rapide

## ☁️ Google Colab

### [COLAB_SETUP.md](COLAB_SETUP.md)
Configuration pour utiliser ForestGaps sur Google Colab

### [GOOGLE_DRIVE_SETUP.md](GOOGLE_DRIVE_SETUP.md)
Setup Google Drive pour stockage des données

## 📦 Référence API

### Module benchmarking
[forestgaps/benchmarking/README.md](../forestgaps/benchmarking/README.md)
- API de comparaison de modèles
- Métriques et visualisations
- Génération de rapports

### Modules principaux
- **config** : Gestion de configuration
- **models** : Architectures deep learning
- **training** : Entraînement des modèles
- **evaluation** : Évaluation et métriques
- **inference** : Prédiction sur nouvelles données
- **data** : Chargement et prétraitement

## 🔧 Développement

### Scripts
Voir [scripts/README.md](../scripts/README.md) pour la documentation des scripts.

### Tests
```bash
pytest tests/
```

## 📋 Archives

Documentation obsolète ou archivée : [archive/](archive/)

## 🆘 Troubleshooting

### Problèmes courants

| Problème | Solution | Doc |
|----------|----------|-----|
| Setup benchmarking | Voir guide benchmarking | [benchmarking/README.md](benchmarking/README.md) |
| Docker | Voir commandes Docker | [benchmarking/COMMANDES_DOCKER.md](benchmarking/COMMANDES_DOCKER.md) |
| Google Colab | Voir setup Colab | [COLAB_SETUP.md](COLAB_SETUP.md) |
| CUDA/GPU | Check `nvidia-smi` | - |
| Module not found | `pip install -e .` | [README.md](../README.md#installation) |

## 🔗 Liens rapides

- **TensorBoard** : http://localhost:6006
- **Jupyter Lab** : http://localhost:8888
- **GitHub** : https://github.com/arthur048/forestgaps
- **Issues** : https://github.com/arthur048/forestgaps/issues

## 📝 Convention de documentation

- **README.md** : Documentation principale d'un module/dossier
- **GUIDE.md** : Guides détaillés et tutoriels
- **REFERENCE.md** : Référence API technique
- **SETUP.md** : Instructions de configuration

---

💡 **Nouveau sur le projet ?** Commencez par [README.md](../README.md) puis [benchmarking/README.md](benchmarking/README.md)
