# ForestGaps-DL Docker

Documentation pour l'utilisation de ForestGaps-DL avec Docker.

## Table des matières

- [Prérequis](#prérequis)
- [Architecture Docker](#architecture-docker)
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [Commandes de base](#commandes-de-base)
  - [Entraînement de modèles](#entraînement-de-modèles)
  - [Inférence](#inférence)
  - [Prétraitement des données](#prétraitement-des-données)
  - [Évaluation des modèles](#évaluation-des-modèles)
- [Volumes et persistance des données](#volumes-et-persistance-des-données)
- [Configuration avancée](#configuration-avancée)
- [Résolution des problèmes](#résolution-des-problèmes)

## Prérequis

Pour utiliser ForestGaps-DL avec Docker, vous aurez besoin de :

- Docker Engine (>= 19.03)
- Docker Compose (>= 1.27.0)
- Pour l'accélération GPU : NVIDIA Container Toolkit (nvidia-docker2)

### Installation de Docker

```bash
# Pour Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Pour Windows/macOS
# Téléchargez et installez Docker Desktop depuis le site officiel
```

### Installation de NVIDIA Docker (pour GPU)

```bash
# Pour Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Architecture Docker

ForestGaps-DL est conteneurisé avec deux images Docker principales :

1. **Image GPU (forestgaps-dl:latest)** : Basée sur PyTorch avec support CUDA, optimisée pour l'entraînement et l'inférence sur GPU.

2. **Image CPU (forestgaps-dl:cpu)** : Version légère sans dépendances CUDA, adaptée aux environnements sans GPU ou pour le prétraitement des données.

Le système utilise également Docker Compose pour définir différents services selon les besoins :

- **forestgaps-dl** : Service principal avec support GPU
- **forestgaps-dl-cpu** : Service alternatif sans GPU
- **training** : Configuration spécifique pour l'entraînement
- **inference** : Configuration pour l'inférence et la prédiction
- **preprocessing** : Configuration pour le prétraitement des données
- **evaluation** : Configuration pour l'évaluation des modèles

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/arthur048/forestgaps-dl.git
cd forestgaps-dl
```

### 2. Générer le fichier requirements.txt

```bash
# Installer les dépendances pour l'analyseur
bash scripts/install_dep.sh

# Générer requirements.txt
python utils/dependency_analyzer.py
```

### 3. Construire les images Docker

```bash
# Rendre les scripts exécutables
chmod +x scripts/*.sh

# Construire les images Docker (GPU et CPU)
bash scripts/docker-build.sh
```

### 4. Vérifier l'installation

```bash
# Tester les images Docker
bash scripts/docker-test.sh
```

## Utilisation

### Commandes de base

Le script `docker-run.sh` facilite l'exécution des conteneurs avec les configurations appropriées :

```bash
# Afficher l'aide
bash scripts/docker-run.sh --help

# Lancer un shell interactif dans le conteneur
bash scripts/docker-run.sh shell

# Forcer l'utilisation de l'image CPU
bash scripts/docker-run.sh --cpu shell

# Forcer l'utilisation de l'image GPU
bash scripts/docker-run.sh --gpu shell
```

### Entraînement de modèles

```bash
# Entraîner un modèle avec docker-compose
docker-compose up training

# Ou avec le script docker-run.sh (crée automatiquement les volumes nécessaires)
bash scripts/docker-run.sh train --config /app/config/defaults/training.yml

# En spécifiant un fichier de configuration personnalisé
bash scripts/docker-run.sh train --config /app/config/mon_entrainement.yml
```

### Inférence

```bash
# Prédiction avec un modèle préentraîné
bash scripts/docker-run.sh predict --model /app/models/model.pth --input /app/data/input.tif --output /app/outputs/prediction.tif

# Version CPU (pour les machines sans GPU)
bash scripts/docker-run.sh --cpu predict --model /app/models/model.pth --input /app/data/input.tif --output /app/outputs/prediction.tif
```

### Prétraitement des données

```bash
# Prétraitement avec la configuration par défaut
bash scripts/docker-run.sh preprocess --config /app/config/defaults/preprocessing.yml

# Avec des paramètres personnalisés
bash scripts/docker-run.sh preprocess --input /app/data/raw --output /app/data/processed
```

### Évaluation des modèles

```bash
# Évaluer un modèle
bash scripts/docker-run.sh evaluate --model /app/models/model.pth --data /app/data/validation --output /app/reports/evaluation.json
```

## Volumes et persistance des données

Par défaut, les volumes suivants sont montés dans les conteneurs :

- `./data:/app/data` : Données d'entrée et de sortie
- `./models:/app/models` : Modèles entraînés
- `./config:/app/config` : Fichiers de configuration
- `./logs:/app/logs` : Logs d'entraînement et TensorBoard (pour training)
- `./outputs:/app/outputs` : Résultats de prédiction (pour inference)
- `./reports:/app/reports` : Rapports d'évaluation (pour evaluation)

Pour monter des volumes supplémentaires :

```bash
bash scripts/docker-run.sh -v "/chemin/local:/chemin/conteneur" train --config /app/config/training.yml
```

## Configuration avancée

### Utilisation de la mémoire et des GPUs

Par défaut, Docker utilisera toutes les ressources disponibles. Pour limiter l'utilisation :

```yaml
# Dans docker-compose.yml
services:
  training:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Personnalisation des images Docker

Pour personnaliser les images Docker, modifiez les fichiers `Dockerfile` et `Dockerfile.cpu` selon vos besoins, puis reconstruisez :

```bash
docker build -t forestgaps-dl:custom -f Dockerfile.custom .
```

## Résolution des problèmes

### Problèmes courants

#### Erreur "NVIDIA CUDA not found"

Vérifiez que NVIDIA Docker est correctement installé :

```bash
# Vérifier l'installation de NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

Si cette commande échoue, réinstallez NVIDIA Docker.

#### Erreur "Out of memory"

Limitez la mémoire utilisée par PyTorch dans votre script ou configuration :

```python
# Dans votre script Python
torch.cuda.set_per_process_memory_fraction(0.7)  # Utilise 70% de la mémoire GPU
```

#### Problèmes de permissions

Si vous rencontrez des problèmes de permissions avec les volumes Docker :

```bash
# Ajustez les permissions des dossiers montés
sudo chown -R 1000:1000 ./data ./models ./outputs
```

### Obtenir de l'aide

Pour des questions spécifiques, veuillez ouvrir une issue sur le dépôt GitHub ou contacter directement les mainteneurs. 