# Dockerfile pour ForestGaps-DL
# Image de base avec PyTorch et dépendances CUDA
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="Arthur <arthurvdl048@email.com>"
LABEL description="ForestGaps-DL: Bibliothèque PyTorch pour la détection et l'analyse des trouées forestières"

# Définition des arguments et variables d'environnement
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libgdal-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Dépendances GDAL pour rasterio
RUN pip install --no-cache-dir GDAL==$(gdal-config --version)

# Installation du package en mode développement
COPY . /app/
RUN pip install -e .

# Mise en place des volumes pour les données et les modèles
VOLUME ["/app/data", "/app/models"]

# Configuration pour l'accès au GPU
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Point d'entrée par défaut
ENTRYPOINT ["python", "-m", "forestgaps_dl"]
CMD ["--help"] 