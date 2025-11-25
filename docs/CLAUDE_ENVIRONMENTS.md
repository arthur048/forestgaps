# Guide Claude Code : Gestion des Environnements

Ce guide explique comment Claude Code gère les différents environnements d'exécution (Docker, Conda, GDAL) pour le projet ForestGaps.

## État des Installations

✅ **Docker** : v28.5.2 - Installé et fonctionnel
✅ **Conda** : v24.5.0 - Installé et fonctionnel
❌ **GDAL** : Non installé dans l'environnement base Python
✅ **Python** : v3.12.4
✅ **R** : v4.5.2

### Environnements Conda Disponibles
- `base` (actif) : C:\ProgramData\miniconda3
- `transcription` : C:\Users\Arthur\.conda\envs\transcription
- `webgis` : C:\Users\Arthur\.conda\envs\webgis

---

## 1. Exécution avec Conda

### Activation d'un environnement spécifique

Claude Code peut exécuter des scripts dans n'importe quel environnement conda :

```bash
# Activer un environnement et exécuter un script
conda run -n webgis python mon_script.py

# Ou avec activation explicite
conda activate webgis && python mon_script.py

# Installer des packages dans un environnement
conda install -n webgis gdal rasterio geopandas
```

### Exemple : Vérifier GDAL dans l'environnement webgis

```bash
conda run -n webgis python -c "from osgeo import gdal; print(gdal.__version__)"
```

### Créer un nouvel environnement pour ForestGaps

```bash
# Créer un environnement avec les dépendances géospatiales
conda create -n forestgaps python=3.12 gdal rasterio geopandas pytorch torchvision -c conda-forge

# Activer et installer forestgaps
conda activate forestgaps
pip install -e .
```

---

## 2. Exécution avec Docker

### Utilisation du Dockerfile existant

Le projet ForestGaps inclut déjà une configuration Docker dans `docker/`.

```bash
# Build de l'image Docker
docker build -t forestgaps:latest -f docker/Dockerfile .

# Exécuter un script dans Docker
docker run --rm -v "$(pwd):/app" forestgaps:latest python /app/scripts/mon_script.py

# Mode interactif
docker run -it --rm -v "$(pwd):/app" forestgaps:latest bash
```

### Exécution de commandes dans un conteneur existant

```bash
# Lancer un conteneur en arrière-plan
docker run -d --name forestgaps-dev -v "$(pwd):/app" forestgaps:latest tail -f /dev/null

# Exécuter des commandes dans le conteneur
docker exec forestgaps-dev python /app/scripts/preprocess.py
docker exec forestgaps-dev python /app/scripts/train.py

# Arrêter et supprimer le conteneur
docker stop forestgaps-dev
docker rm forestgaps-dev
```

### Docker Compose

Si vous avez un `docker-compose.yml` :

```bash
# Démarrer les services
docker-compose up -d

# Exécuter une commande
docker-compose exec app python scripts/train.py

# Arrêter les services
docker-compose down
```

---

## 3. Gestion de GDAL

### Option A : Installation via Conda (Recommandé pour ForestGaps)

```bash
# Dans l'environnement webgis (probablement déjà présent)
conda activate webgis
python -c "from osgeo import gdal; print(gdal.__version__)"

# Ou créer un nouvel environnement
conda create -n forestgaps-gdal python=3.12 gdal rasterio geopandas numpy -c conda-forge
conda activate forestgaps-gdal
pip install -e .
```

### Option B : Installation via pip (moins fiable sur Windows)

```bash
pip install GDAL==$(gdal-config --version)
```

### Option C : Utiliser GDAL dans Docker (le plus fiable)

Le Dockerfile de ForestGaps devrait déjà inclure GDAL :

```dockerfile
FROM python:3.12-slim

# Installation de GDAL
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install GDAL==$(gdal-config --version)
RUN pip install -r requirements.txt
```

---

## 4. Workflows Claude Code pour ForestGaps

### Workflow 1 : Développement Local avec Conda

```bash
# 1. Activer l'environnement
conda activate webgis  # ou forestgaps

# 2. Exécuter le preprocessing
python -m forestgaps.cli.preprocessing_cli \
    --dsm data/raw/site1_dsm.tif \
    --chm data/raw/site1_chm.tif \
    --output data/processed/

# 3. Entraîner un modèle
python -m forestgaps.cli.training_cli \
    --config configs/unet_film.yaml \
    --data data/processed/ \
    --output models/

# 4. Évaluer
python -m forestgaps.cli.evaluate \
    --model models/unet_film.pt \
    --data data/test/
```

### Workflow 2 : Entraînement dans Docker

```bash
# 1. Build de l'image avec toutes les dépendances
docker build -t forestgaps:latest .

# 2. Monter les données et lancer l'entraînement
docker run --gpus all --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/models:/app/models" \
    forestgaps:latest \
    python -m forestgaps.cli.training_cli \
        --config /app/configs/unet_film.yaml \
        --data /app/data/processed/ \
        --output /app/models/
```

### Workflow 3 : Inference sur de nouvelles données

```bash
# Avec conda
conda run -n webgis python -c "
from forestgaps.inference import run_inference
result = run_inference(
    model_path='models/unet_film.pt',
    dsm_path='data/new/site2_dsm.tif',
    output_path='results/site2_prediction.tif',
    threshold=5.0
)
print(f'Inference terminée : {result}')
"

# Avec Docker
docker run --rm \
    -v "$(pwd):/app" \
    forestgaps:latest \
    python -m forestgaps.inference.core \
        --model /app/models/unet_film.pt \
        --input /app/data/new/site2_dsm.tif \
        --output /app/results/site2_prediction.tif
```

---

## 5. Exemples d'Exécution par Claude Code

### Exemple 1 : Script Python avec bibliothèques géospatiales

Quand vous demandez : "Exécute le preprocessing sur les données du site 1"

Claude Code exécutera :
```bash
conda run -n webgis python -m forestgaps.cli.preprocessing_cli \
    --dsm data/raw/site1_dsm.tif \
    --chm data/raw/site1_chm.tif \
    --output data/processed/site1/
```

### Exemple 2 : Entraînement avec GPU dans Docker

Quand vous demandez : "Lance l'entraînement du modèle U-Net FiLM avec GPU"

Claude Code exécutera :
```bash
docker run --gpus all --rm \
    -v "G:/Mon Drive/forestgaps-dl/data:/app/data" \
    -v "G:/Mon Drive/forestgaps-dl/models:/app/models" \
    -v "G:/Mon Drive/forestgaps-dl/configs:/app/configs" \
    forestgaps:latest \
    python -m forestgaps.cli.training_cli \
        --config /app/configs/unet_film.yaml \
        --epochs 50 \
        --batch-size 16
```

### Exemple 3 : Installation de nouvelles dépendances

Quand vous demandez : "Installe la bibliothèque rioxarray"

Claude Code peut :
```bash
# Dans conda
conda install -n webgis rioxarray -c conda-forge

# Ou dans le package
pip install rioxarray

# Ou dans Docker (rebuild)
docker build --build-arg EXTRA_DEPS="rioxarray" -t forestgaps:latest .
```

---

## 6. Gestion des Dépendances du Projet

### Fichiers de dépendances

Le projet ForestGaps utilise plusieurs fichiers :
- `setup.py` : Dépendances pip principales
- `requirements.txt` : Dépendances exactes (générées)
- `environment.yml` : Environnement conda recommandé
- `docker/requirements.txt` : Dépendances Docker

### Mettre à jour les dépendances

```bash
# Générer requirements.txt depuis l'environnement actuel
pip freeze > requirements.txt

# Créer environment.yml pour conda
conda env export > environment.yml

# Ou manuellement (recommandé)
conda env export --from-history > environment.yml
```

---

## 7. Bonnes Pratiques

### ✅ Recommandations

1. **Utiliser Conda pour le développement local**
   - Plus facile à gérer sur Windows
   - GDAL et bibliothèques géospatiales bien supportées
   - Environnements isolés

2. **Utiliser Docker pour la production et CI/CD**
   - Reproductibilité garantie
   - Facilite le déploiement
   - Idéal pour les serveurs de calcul

3. **Documenter l'environnement utilisé**
   - Indiquer dans les README quels environnements fonctionnent
   - Fournir des exemples pour chaque méthode

### ❌ Pièges à éviter

1. **Ne pas mélanger pip et conda** pour GDAL
   - Toujours utiliser conda pour GDAL sur Windows
   - Ou utiliser Docker

2. **Attention aux chemins Windows dans Docker**
   - Utiliser des chemins absolus ou relatifs Unix
   - Monter les volumes correctement

3. **Vérifier les versions de GDAL**
   - Les versions de GDAL Python doivent correspondre à GDAL système
   - Utiliser `gdal-config --version` et `python -c "from osgeo import gdal; print(gdal.__version__)"`

---

## 8. Tests Rapides

### Tester GDAL dans différents environnements

```bash
# Test conda base
python -c "from osgeo import gdal; print(f'GDAL base: {gdal.__version__}')"

# Test conda webgis
conda run -n webgis python -c "from osgeo import gdal; print(f'GDAL webgis: {gdal.__version__}')"

# Test Docker
docker run --rm forestgaps:latest python -c "from osgeo import gdal; print(f'GDAL Docker: {gdal.__version__}')"
```

### Tester rasterio (dépend de GDAL)

```bash
conda run -n webgis python -c "import rasterio; print(f'Rasterio: {rasterio.__version__}')"
```

---

## Résumé : Comment Claude Code Choisit l'Environnement

Quand vous demandez d'exécuter un script, Claude Code :

1. **Analyse le script** pour détecter les dépendances
2. **Vérifie les imports** (rasterio, gdal, pytorch, etc.)
3. **Choisit l'environnement approprié** :
   - Scripts avec GDAL/rasterio → `conda run -n webgis`
   - Scripts PyTorch GPU → Docker avec `--gpus all`
   - Scripts simples → Environnement Python base
4. **Exécute et capture l'output** complet

Vous pouvez aussi **spécifier explicitement** l'environnement :
- "Exécute dans l'environnement webgis"
- "Utilise Docker pour cette commande"
- "Lance ça dans conda base"

---

## Questions Fréquentes

**Q : GDAL ne fonctionne pas dans mon environnement base ?**
R : Normal, installez-le avec `conda install -c conda-forge gdal` ou utilisez l'environnement `webgis`.

**Q : Puis-je utiliser plusieurs environnements en même temps ?**
R : Oui ! Claude Code peut exécuter des commandes dans différents environnements selon les besoins.

**Q : Comment savoir quel environnement est utilisé ?**
R : Claude Code affiche toujours la commande complète exécutée, incluant `conda run -n ...` ou `docker run ...`.

**Q : Peut-on créer un environnement forestgaps dédié ?**
R : Oui, recommandé ! Voir la section "Créer un nouvel environnement pour ForestGaps".

---

## Prochaines Étapes

Voulez-vous que Claude Code :
1. Crée un environnement conda dédié `forestgaps` avec toutes les dépendances ?
2. Vérifie si GDAL est disponible dans l'environnement `webgis` ?
3. Build l'image Docker avec les dépendances complètes ?
4. Teste l'exécution d'un script de preprocessing ?
