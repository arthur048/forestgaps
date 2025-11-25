# ForestGaps Docker Setup

Configuration Docker complète pour le projet ForestGaps avec support CUDA/GPU.

## 📋 Vue d'Ensemble

Cette configuration Docker résout définitivement les problèmes de compatibilité rasterio/GDAL et fournit un environnement reproductible pour le développement et le déploiement.

### Caractéristiques

- ✅ **Python 3.10** avec PyTorch 2.4.0
- ✅ **CUDA 12.4** + cuDNN 9 pour GPU
- ✅ **GDAL 3.8.0** préinstallé sans conflits
- ✅ **Rasterio 1.3.9** compatible
- ✅ **Multi-core CPU** optimisé pour batch processing
- ✅ **Scripts simplifiés** pour développeurs débutants

## 🚀 Démarrage Rapide

### 1. Prérequis

**Sur votre machine :**
- Docker Desktop installé ([télécharger](https://www.docker.com/products/docker-desktop))
- Pour GPU : NVIDIA Driver ≥ 525.60.13 + nvidia-container-toolkit

**Vérifier Docker :**
```bash
docker --version
docker-compose --version
```

**Vérifier GPU (optionnel) :**
```bash
nvidia-smi
```

### 2. Build de l'Image

**Méthode simple (recommandée) :**
```bash
./scripts/docker-build.sh
```

**Méthode manuelle :**
```bash
docker build -f docker/Dockerfile --target development -t forestgaps:latest .
```

Le build prend environ **10-15 minutes** la première fois (téléchargement des images de base).

### 3. Validation

Vérifiez que tout fonctionne :
```bash
./scripts/docker-test.sh
```

Cela exécute 7 tests automatiques :
1. ✓ Image existe
2. ✓ Container démarre
3. ✓ Imports Python (torch, rasterio, geopandas, forestgaps)
4. ✓ GPU disponible (si présent)
5. ✓ Détection environnement
6. ✓ Compatibilité GDAL/rasterio
7. ✓ Health check

## 💻 Utilisation

### Commandes Principales

#### Ouvrir un Shell Interactif
```bash
./scripts/docker-run.sh shell
```

Vous êtes maintenant dans le container. Essayez :
```bash
python -c "import forestgaps; print(forestgaps.__version__)"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

#### Lancer Jupyter Notebook
```bash
./scripts/docker-run.sh jupyter
```

Accédez à : http://localhost:8888 (token: `forestgaps`)

#### Entraîner un Modèle
```bash
./scripts/docker-run.sh train --data-dir ./data --models-dir ./models
```

#### Inférence sur Nouvelles Données
```bash
./scripts/docker-run.sh inference --data-dir ./data --models-dir ./models
```

#### Exécuter les Tests
```bash
./scripts/docker-run.sh test
```

### Options Avancées

#### Spécifier Répertoires Personnalisés
```bash
./scripts/docker-run.sh train \
  --data-dir /chemin/vers/data \
  --models-dir /chemin/vers/models \
  --outputs-dir /chemin/vers/outputs \
  --logs-dir /chemin/vers/logs
```

#### Mode CPU Uniquement
```bash
./scripts/docker-run.sh shell --gpu disabled
```

#### Utiliser une Image Spécifique
```bash
./scripts/docker-run.sh shell --image forestgaps:v1.0.0
```

## 🏗️ Architecture Docker

### Images de Base

L'image utilise une approche multi-stage :

```
Stage 1: GDAL Builder (osgeo/gdal:ubuntu-small-3.8.0)
         └─> Fournit GDAL 3.8.0 + bibliothèques système

Stage 2: PyTorch Base (pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel)
         └─> Python 3.10 + CUDA 12.4 + PyTorch 2.4.0
         └─> Copie GDAL depuis Stage 1

Stage 3: Dependencies
         └─> Installation ordonnée des dépendances Python

Stage 4: Development
         └─> Installation ForestGaps + outils dev
         └─> Image finale ~4.5 GB
```

### Points de Montage (Volumes)

| Volume | Mode | Usage |
|--------|------|-------|
| `./data` → `/app/data` | ro | Données d'entrée (DSM, CHM) |
| `./models` → `/app/models` | rw | Checkpoints modèles |
| `./outputs` → `/app/outputs` | rw | Résultats prédictions |
| `./logs` → `/app/logs` | rw | Logs TensorBoard |

**Note :** `ro` = read-only, `rw` = read-write

## 🔧 Configuration GPU

### Installation NVIDIA Container Toolkit (Windows WSL2)

Si vous avez un GPU NVIDIA et Docker Desktop sur Windows :

```bash
# Dans WSL2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Vérifier GPU dans Container

```bash
./scripts/docker-run.sh shell
# Dans le container:
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 🐛 Troubleshooting

### Problème : Image Build Échoue

**Symptôme :** Erreur pendant `docker build`

**Solutions :**
1. Rebuild sans cache :
   ```bash
   ./scripts/docker-build.sh --no-cache
   ```

2. Vérifier espace disque disponible :
   ```bash
   df -h
   ```

3. Vérifier logs détaillés :
   ```bash
   docker build -f docker/Dockerfile --target development --progress=plain -t forestgaps:latest .
   ```

### Problème : GPU Non Détecté

**Symptôme :** `torch.cuda.is_available()` retourne `False`

**Solutions :**
1. Vérifier driver NVIDIA sur host :
   ```bash
   nvidia-smi
   ```

2. Vérifier nvidia-docker :
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

3. Relancer Docker Desktop

### Problème : "No Space Left on Device"

**Symptôme :** Erreur lors du training avec DataLoader

**Solution :**
Le container utilise déjà `shm-size: 8gb`. Si insuffisant, éditer `docker-compose.yml` :
```yaml
shm_size: '16gb'  # Augmenter à 16 GB
```

### Problème : GDAL Version Mismatch

**Symptôme :** Erreur "GDAL API version must be specified"

**Solution :**
Ceci ne devrait PAS arriver grâce à notre Dockerfile. Si cela se produit :
```bash
./scripts/docker-run.sh shell
# Dans le container:
python -c "from osgeo import gdal; print(gdal.__version__)"
python -c "import rasterio; print(rasterio.__version__)"
```

Les versions doivent être :
- GDAL : 3.8.0
- Rasterio : 1.3.9

### Problème : Permissions Denied

**Symptôme :** Impossible d'écrire dans `/app/models` ou `/app/outputs`

**Solution :**
Le container s'exécute avec l'utilisateur `forestgaps` (UID 1000). Vérifier permissions sur host :
```bash
sudo chown -R 1000:1000 ./models ./outputs ./logs
```

## 📦 Docker Compose

### Démarrer avec Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Voir les Logs

```bash
docker-compose -f docker/docker-compose.yml logs -f
```

### Arrêter

```bash
docker-compose -f docker/docker-compose.yml down
```

### Rebuild

```bash
docker-compose -f docker/docker-compose.yml up --build
```

## 📊 Optimisations Performance

### Multi-Core CPU (Batch Processing)

Le container est configuré pour utiliser **8 CPU cores** pour le preprocessing parallèle des données pendant le training.

Variables d'environnement (déjà configurées) :
```yaml
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

### GPU Memory Management

Configuration automatique pour éviter les OOM (Out Of Memory) :
```yaml
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🧪 Développement

### Live Code Editing

Pour développer sans rebuilder :

1. Décommenter dans `docker-compose.yml` :
   ```yaml
   volumes:
     - ../forestgaps:/app/forestgaps:rw
     - ../tests:/app/tests:rw
   ```

2. Relancer :
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

Vos modifications dans `forestgaps/` sont maintenant live !

### Ajouter des Dépendances

1. Ajouter dans `requirements/requirements.txt`
2. Rebuild l'image :
   ```bash
   ./scripts/docker-build.sh
   ```

## 📝 Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `docker/Dockerfile` | Build multi-stage complet |
| `docker/docker-compose.yml` | Orchestration Docker |
| `docker/.dockerignore` | Exclusions build context |
| `docker/healthcheck.py` | Health check container |
| `scripts/docker-build.sh` | Script de build |
| `scripts/docker-run.sh` | Script d'exécution |
| `scripts/docker-test.sh` | Script de validation |
| `requirements/requirements.txt` | Dépendances production |

## 🎯 Compatibilité Colab

Le code reste 100% compatible avec Google Colab !

**Détection automatique de l'environnement :**
```python
from forestgaps.environment import setup_environment

env = setup_environment()
# Détecte automatiquement : Docker, Colab, ou Local
```

**Résultat :**
- Dans Docker : `DockerEnvironment`
- Dans Colab : `ColabEnvironment`
- En local : `LocalEnvironment`

Aucune modification de code nécessaire ! 🎉

## 🔒 Sécurité

- ✅ Container s'exécute en **non-root** (utilisateur `forestgaps`)
- ✅ Volumes data en **read-only** pour éviter modifications accidentelles
- ✅ **Pas de secrets** dans l'image (utiliser variables d'environnement)
- ✅ Dépendances **pinnées** pour éviter supply chain attacks

### Scanner Vulnérabilités

```bash
# Installer trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Scanner l'image
trivy image forestgaps:latest
```

## 📚 Ressources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch/tags)
- [OSGeo GDAL Docker](https://hub.docker.com/r/osgeo/gdal)

## 🆘 Support

**Problème avec Docker ?**
1. Consulter la section Troubleshooting ci-dessus
2. Vérifier les logs : `docker logs <container_id>`
3. Ouvrir une issue sur GitHub avec :
   - Version Docker : `docker --version`
   - OS : `uname -a` ou `ver` (Windows)
   - Logs complets de l'erreur

## 📄 License

Ce projet est sous licence MIT. Voir [LICENSE](../LICENSE) pour plus de détails.

---

**Prêt à démarrer ? 🚀**

```bash
# 1. Build l'image
./scripts/docker-build.sh

# 2. Vérifier que tout fonctionne
./scripts/docker-test.sh

# 3. Ouvrir un shell
./scripts/docker-run.sh shell

# 4. Commencer à coder !
```

**Des questions ?** Consultez le [README principal](../README.md) ou ouvrez une issue !
