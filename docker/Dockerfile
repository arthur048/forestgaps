# ============================================
# ForestGaps Docker Image - GDAL Full + PyTorch
# ============================================
# Approche robuste : GDAL complet + PyTorch via conda
# Tout installé via conda pour cohérence maximale
# ============================================

# Image de base avec GDAL COMPLET (toutes dépendances incluses)
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.8.5

LABEL maintainer="vdlinden.arthur@gmail.com"
LABEL description="ForestGaps - Forest canopy gap detection using deep learning"
LABEL version="0.1.1"
LABEL gdal.version="3.8.5"
LABEL pytorch.version="2.4.0"
LABEL cuda.version="12.4"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ============================================
# Install System Dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Install Miniconda
# ============================================
# L'image GDAL n'a pas conda, on l'installe
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init bash

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Configure conda
RUN conda config --set channel_priority flexible && \
    conda config --prepend channels conda-forge && \
    conda config --prepend channels pytorch && \
    conda config --set auto_update_conda false

# ============================================
# Install ALL Dependencies via Conda
# ============================================
# TOUT via conda pour cohérence maximale
# PyTorch depuis canal pytorch (officiel)
# Reste depuis conda-forge
RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda install -y \
    # PyTorch + CUDA (canal pytorch officiel)
    pytorch=2.4.0 \
    torchvision=0.19.0 \
    pytorch-cuda=12.4 \
    -c pytorch -c nvidia && \
    # Geospatial (déjà dans l'image mais on force les versions)
    conda install -y -c conda-forge \
    gdal=3.8.5 \
    rasterio=1.3.9 \
    geopandas=0.14.2 \
    fiona=1.9.5 \
    shapely=2.0.2 && \
    # ML & Scientific
    conda install -y -c conda-forge \
    scikit-learn=1.3.2 \
    scikit-image=0.22.0 \
    numpy=1.26.3 \
    pandas=2.1.4 \
    scipy=1.11.4 && \
    # Visualization
    conda install -y -c conda-forge \
    matplotlib=3.8.2 \
    seaborn=0.13.0 \
    pillow=10.1.0 && \
    # Utils
    conda install -y -c conda-forge \
    pydantic=2.5.0 \
    pyyaml=6.0.1 \
    tensorboard=2.15.1 \
    tqdm=4.66.1 \
    tabulate=0.9.0 && \
    # Dev tools
    conda install -y -c conda-forge \
    pytest=7.4.3 \
    black=23.12.1 \
    isort=5.13.2 \
    flake8=6.1.0 && \
    conda clean -afy

# Upgrade pip (pour rares packages non sur conda)
RUN pip install --no-cache-dir --upgrade pip

# ============================================
# Verify Critical Imports
# ============================================
RUN python -c "from osgeo import gdal; print(f'✓ GDAL {gdal.__version__}')" && \
    python -c "import rasterio; print(f'✓ Rasterio {rasterio.__version__}')" && \
    python -c "import geopandas as gpd; print(f'✓ GeoPandas {gpd.__version__}')" && \
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" && \
    python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

# ============================================
# Install ForestGaps Package
# ============================================
WORKDIR /app

# Copy project files
COPY forestgaps/ /app/forestgaps/
COPY setup.py README.md MANIFEST.in pytest.ini ./

# Install in editable mode
RUN pip install --no-cache-dir -e .

# Create directories
RUN mkdir -p /app/data /app/models /app/outputs /app/logs /app/tests

# Copy Docker files
COPY docker/healthcheck.py /app/healthcheck.py
COPY tests/ /app/tests/

# ============================================
# Security: Non-Root User
# ============================================
RUN useradd -m -u 1000 -s /bin/bash forestgaps && \
    chown -R forestgaps:forestgaps /app

USER forestgaps
WORKDIR /app

# ============================================
# Environment Variables
# ============================================
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_HOME=/app/.torch \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# ============================================
# Health Check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

EXPOSE 6006 8888

CMD ["python", "-c", "from forestgaps.environment import Environment; env = Environment.detect(); import pprint; pprint.pprint(env.get_environment_info())"]

# ============================================
# Build Instructions:
# ============================================
# Build image:
#   docker build -f docker/Dockerfile -t forestgaps:latest .
#
# Run with GPU:
#   docker run --gpus all --rm -it forestgaps:latest
#
# Run with volumes:
#   docker run --gpus all --rm -it \
#     -v $(pwd)/data:/app/data:ro \
#     -v $(pwd)/models:/app/models:rw \
#     -v $(pwd)/outputs:/app/outputs:rw \
#     forestgaps:latest
#
# Open shell:
#   docker run --gpus all --rm -it forestgaps:latest /bin/bash
# ============================================
