# ForestGaps - Guide de l'Environnement de Développement

## 🎯 Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TON PC WINDOWS                               │
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────────────────────────────┐│
│  │    VS Code      │    │           Docker Desktop                ││
│  │                 │    │                                         ││
│  │  - Code source  │    │  ┌─────────────────────────────────┐   ││
│  │  - Claude Code  │◄──►│  │    Conteneur forestgaps-main   │   ││
│  │  - Terminal     │    │  │    (Ubuntu + PyTorch + CUDA)    │   ││
│  │                 │    │  │                                 │   ││
│  └─────────────────┘    │  │  /app/forestgaps ◄─────────────┼───┼─► G:\...\forestgaps\
│                         │  │  /app/data      ◄─────────────┼───┼─► G:\...\data\
│         ▲               │  │  /app/models    ◄─────────────┼───┼─► G:\...\models\
│         │               │  │  /app/logs      ◄─────────────┼───┼─► G:\...\logs\
│         │               │  └─────────────────────────────────┘   ││
│         │               │                                         ││
│    GPU NVIDIA ◄─────────┼──► Accès GPU via nvidia-docker          ││
│    (RTX/etc.)           │                                         ││
│                         └─────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Comprendre les Concepts Docker

### Qu'est-ce qu'un Conteneur ?

Imagine une **machine virtuelle ultra-légère** qui contient :
- Ubuntu 24.04
- Python 3.12
- PyTorch + CUDA
- Toutes tes librairies (rasterio, geopandas, etc.)

**Avantage** : Ton environnement est identique partout. Plus de "ça marche sur ma machine".

### Les 3 Conteneurs de ForestGaps

```
┌─────────────────────────────────────────────────────────────────┐
│                    docker-compose.yml                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  forestgaps-main      Le conteneur de base                      │
│  ─────────────────    - Pour lancer des scripts ponctuels       │
│                       - Pour le shell interactif                │
│                       - Pour les tests                          │
│                                                                 │
│  forestgaps-jupyter   Interface web interactive                 │
│  ─────────────────    - Notebooks pour exploration              │
│                       - Visualisation de données                │
│                       - http://localhost:8888                   │
│                                                                 │
│  forestgaps-tensorboard   Monitoring entraînement               │
│  ─────────────────────    - Courbes de loss                     │
│                           - Métriques en temps réel             │
│                           - http://localhost:6006               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Le Network `docker_default`

C'est un **réseau virtuel** qui permet aux conteneurs de communiquer entre eux.
Par exemple, Jupyter pourrait appeler TensorBoard en interne via `tensorboard:6006`.

Tu n'as pas à t'en soucier — Docker le gère automatiquement.

### Le Healthcheck

```yaml
healthcheck:
  test: ["CMD", "python", "/app/healthcheck.py"]
  interval: 30s
```

C'est un **check de santé** : Docker exécute ce script toutes les 30s pour vérifier 
que le conteneur fonctionne. Si ça échoue 3 fois, Docker peut redémarrer le conteneur.

Utile pour la production, optionnel pour le dev.

---

## 🔗 Les Volumes : Le Pont entre Windows et Docker

```yaml
volumes:
  - ../forestgaps:/app/forestgaps:rw   # Code source (lecture/écriture)
  - ../data:/app/data:ro                # Données (lecture seule)
  - ../models:/app/models:rw            # Checkpoints (lecture/écriture)
  - ../logs:/app/logs:rw                # Logs TensorBoard (lecture/écriture)
```

**Ce que ça signifie :**

| Chemin Windows | Chemin dans Docker | Mode |
|----------------|-------------------|------|
| `G:\Mon Drive\forestgaps-dl\forestgaps\` | `/app/forestgaps/` | rw |
| `G:\Mon Drive\forestgaps-dl\data\` | `/app/data/` | ro |
| `G:\Mon Drive\forestgaps-dl\models\` | `/app/models/` | rw |
| `G:\Mon Drive\forestgaps-dl\logs\` | `/app/logs/` | rw |

**Conséquence magique** : Tu édites le code dans VS Code sur Windows, et les 
modifications sont **instantanément** visibles dans le conteneur Docker !

---

## 💻 Workflow Quotidien

### Structure de ton projet

```
G:\Mon Drive\forestgaps-dl\
├── forestgaps/              # 📦 Package Python (ton code)
│   ├── __init__.py
│   ├── models/              # Architectures U-Net, etc.
│   ├── data/                # DataLoaders, augmentations
│   ├── training/            # Boucle d'entraînement
│   ├── inference/           # Prédiction
│   └── cli/                 # Scripts CLI
│       ├── train.py
│       └── predict.py
│
├── data/                    # 📊 Données (GeoTIFF, etc.)
│   ├── raw/                 # Données brutes
│   ├── processed/           # Données prétraitées
│   └── splits/              # Train/val/test
│
├── models/                  # 🧠 Checkpoints sauvegardés
│   └── experiment_001/
│       ├── best_model.pt
│       └── config.yaml
│
├── logs/                    # 📈 Logs TensorBoard
│   └── experiment_001/
│
├── outputs/                 # 🗺️ Prédictions, visualisations
│
├── tests/                   # ✅ Tests unitaires
│
├── notebooks/               # 📓 Jupyter notebooks
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── setup.py
└── README.md
```

### Démarrage de journée typique

```powershell
# 1. Ouvrir VS Code dans le projet
cd "G:\Mon Drive\forestgaps-dl"
code .

# 2. Lancer l'environnement Docker
cd docker
docker-compose up jupyter tensorboard -d   # -d = en arrière-plan

# 3. Ouvrir les interfaces
# → http://localhost:8888  (Jupyter)
# → http://localhost:6006  (TensorBoard)
```

### Fin de journée

```powershell
# Arrêter les conteneurs
docker-compose down
```

---

## 🛠️ Commandes Essentielles

### Développement quotidien

```powershell
# Lancer Jupyter + TensorBoard
docker-compose up jupyter tensorboard -d

# Shell interactif dans le conteneur
docker-compose run --rm forestgaps bash

# Lancer un script Python
docker-compose run --rm forestgaps python -m forestgaps.cli.train --config config.yaml

# Lancer les tests
docker-compose run --rm forestgaps pytest

# Lancer un test spécifique
docker-compose run --rm forestgaps pytest tests/test_model.py -v

# Voir les logs d'un conteneur
docker-compose logs -f jupyter
```

### Gestion Docker

```powershell
# Voir les conteneurs en cours
docker ps

# Voir tous les conteneurs (même arrêtés)
docker ps -a

# Arrêter tout proprement
docker-compose down

# Rebuild après modification du Dockerfile
docker-compose build

# Rebuild sans cache (en cas de problème)
docker-compose build --no-cache

# Nettoyer les images/conteneurs inutilisés
docker system prune -f
```

---

## 🔧 VS Code + Docker

### Extensions recommandées

1. **Docker** (Microsoft) — Gestion visuelle des conteneurs
2. **Dev Containers** (Microsoft) — Développer DANS le conteneur
3. **Python** (Microsoft)
4. **Jupyter** (Microsoft)

### Option 1 : Éditer sur Windows, exécuter dans Docker (Recommandé)

```
┌──────────────────┐         ┌──────────────────┐
│     VS Code      │         │     Docker       │
│                  │         │                  │
│  Édition code    │──────►  │  Exécution       │
│  IntelliSense    │ volumes │  GPU             │
│  Git             │         │  PyTorch         │
└──────────────────┘         └──────────────────┘
```

**Workflow** :
1. Tu édites dans VS Code (Windows)
2. Les volumes synchronisent automatiquement
3. Tu exécutes dans Docker via terminal

### Option 2 : Dev Container (tout dans Docker)

VS Code peut s'attacher directement au conteneur :

1. `Ctrl+Shift+P` → "Dev Containers: Attach to Running Container"
2. Sélectionner `forestgaps-jupyter`
3. VS Code s'ouvre DANS le conteneur

**Avantage** : IntelliSense parfait (même environnement Python)
**Inconvénient** : Plus lent, extensions à réinstaller

---

## 🤖 Claude Code + Docker

Claude Code fonctionne sur ton système Windows. Pour exécuter du code dans Docker :

### Méthode 1 : Demander à Claude de générer la commande

```
"Claude, lance l'entraînement avec le config experiment_001.yaml"
→ Claude génère : docker-compose run --rm forestgaps python -m forestgaps.cli.train --config configs/experiment_001.yaml
```

### Méthode 2 : Script wrapper

Crée un fichier `run.ps1` :

```powershell
# run.ps1 - Wrapper pour exécuter dans Docker
param(
    [Parameter(Mandatory=$true)]
    [string]$Command
)

docker-compose -f docker/docker-compose.yml run --rm forestgaps $Command
```

Usage :
```powershell
.\run.ps1 "python -m forestgaps.cli.train"
.\run.ps1 "pytest tests/"
```

---

## 🧪 Tests et Qualité de Code

### Structure des tests

```
tests/
├── conftest.py              # Fixtures pytest partagées
├── test_models.py           # Tests des architectures
├── test_data.py             # Tests des DataLoaders
├── test_training.py         # Tests de la boucle d'entraînement
└── test_inference.py        # Tests de prédiction
```

### Lancer les tests

```powershell
# Tous les tests
docker-compose run --rm forestgaps pytest

# Avec couverture
docker-compose run --rm forestgaps pytest --cov=forestgaps

# Tests rapides seulement (marqués)
docker-compose run --rm forestgaps pytest -m "not slow"

# Un fichier spécifique
docker-compose run --rm forestgaps pytest tests/test_models.py -v
```

### Formatage et linting

```powershell
# Formater le code
docker-compose run --rm forestgaps black forestgaps/

# Trier les imports
docker-compose run --rm forestgaps isort forestgaps/

# Vérifier le style
docker-compose run --rm forestgaps flake8 forestgaps/
```

---

## 🚀 Workflow d'Entraînement Deep Learning

### 1. Préparation des données

```python
# Dans un notebook Jupyter (http://localhost:8888)

import rasterio
import geopandas as gpd
from forestgaps.data import ForestGapsDataset

# Explorer tes données LiDAR
with rasterio.open('/app/data/raw/yangambi_chm.tif') as src:
    chm = src.read(1)
    print(f"Shape: {chm.shape}, CRS: {src.crs}")
```

### 2. Configuration d'expérience

```yaml
# configs/experiment_001.yaml
experiment:
  name: "unet_baseline"
  
data:
  train_path: "/app/data/splits/train"
  val_path: "/app/data/splits/val"
  batch_size: 16
  
model:
  architecture: "unet"
  encoder: "resnet34"
  
training:
  epochs: 100
  lr: 0.001
  
logging:
  tensorboard_dir: "/app/logs/experiment_001"
  checkpoint_dir: "/app/models/experiment_001"
```

### 3. Lancement de l'entraînement

```powershell
# Terminal PowerShell
docker-compose run --rm forestgaps python -m forestgaps.cli.train --config configs/experiment_001.yaml
```

### 4. Monitoring avec TensorBoard

Ouvre http://localhost:6006 et observe :
- **Scalars** : Loss, metrics par epoch
- **Images** : Prédictions vs ground truth
- **Histograms** : Distribution des poids

### 5. Reprise d'entraînement

```powershell
docker-compose run --rm forestgaps python -m forestgaps.cli.train \
  --config configs/experiment_001.yaml \
  --resume /app/models/experiment_001/checkpoint_epoch_50.pt
```

---

## 📊 Exemple de Session Complète

```powershell
# === MATIN : Setup ===
cd "G:\Mon Drive\forestgaps-dl\docker"
docker-compose up jupyter tensorboard -d

# === DÉVELOPPEMENT ===
# 1. Ouvrir VS Code, éditer forestgaps/models/unet.py
# 2. Tester rapidement
docker-compose run --rm forestgaps pytest tests/test_models.py -v

# === EXPLORATION ===
# Ouvrir http://localhost:8888
# Créer un notebook pour explorer les données

# === ENTRAÎNEMENT ===
docker-compose run --rm forestgaps python -m forestgaps.cli.train --config configs/exp001.yaml

# Surveiller sur http://localhost:6006

# === INFÉRENCE ===
docker-compose run --rm forestgaps python -m forestgaps.cli.predict \
  --model /app/models/exp001/best.pt \
  --input /app/data/test/tile_001.tif \
  --output /app/outputs/pred_001.tif

# === FIN DE JOURNÉE ===
docker-compose down
```

---

## 🐛 Dépannage Courant

### "No space left on device"
```powershell
# Augmenter shm_size dans docker-compose.yml
shm_size: '16gb'

# Ou nettoyer Docker
docker system prune -a
```

### "CUDA out of memory"
```python
# Réduire batch_size dans config
# Ou activer gradient checkpointing
model.gradient_checkpointing_enable()
```

### Les modifications de code ne sont pas prises en compte
```powershell
# Le hot-reload devrait fonctionner grâce aux volumes
# Si problème, relancer le conteneur
docker-compose restart jupyter
```

### Port déjà utilisé
```powershell
# Trouver le processus
netstat -ano | findstr :8888
# Tuer le processus
taskkill /PID <numero> /F
```

---

## ✅ Checklist Bonnes Pratiques

- [ ] **Versionner** : `git commit` régulièrement
- [ ] **Configurer** : Un fichier YAML par expérience, jamais de magic numbers
- [ ] **Logger** : Tout dans TensorBoard (loss, lr, exemples visuels)
- [ ] **Tester** : Au moins les fonctions critiques
- [ ] **Documenter** : Docstrings, README à jour
- [ ] **Sauvegarder** : Checkpoints réguliers + config associée
- [ ] **Reproduire** : `environment.yml` ou `requirements.txt` figé

---

## 🎓 Pour Aller Plus Loin

1. **MLflow** : Tracking d'expériences plus avancé que TensorBoard
2. **DVC** : Versioning des données et modèles
3. **Weights & Biases** : Alternative cloud à TensorBoard
4. **Hydra** : Gestion avancée des configurations
5. **PyTorch Lightning** : Abstraction de la boucle d'entraînement

Bon courage pour ForestGaps ! 🌴🛰️