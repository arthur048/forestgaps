# Docker pour ForestGaps

ForestGaps est désormais disponible en tant que conteneur Docker, facilitant son déploiement et son utilisation sur différentes plateformes.

## Fonctionnalités principales

- **Double image** : Version GPU (CUDA) et CPU
- **Services préconfigurés** : Entraînement, inférence, prétraitement, évaluation
- **Scripts utilitaires** : Construction, exécution et test des conteneurs
- **Isolation des dépendances** : Environnement reproductible et portable

## Utilisation rapide

```bash
# Construire les images
bash scripts/docker-build.sh

# Exécuter un shell interactif
bash scripts/docker-run.sh shell

# Entraîner un modèle
bash scripts/docker-run.sh train --config /app/config/defaults/training.yml

# Faire des prédictions
bash scripts/docker-run.sh predict --model /app/models/model.pth --input /app/data/input.tif
```

Pour des instructions détaillées, consultez [la documentation Docker complète](docker/README.md). 