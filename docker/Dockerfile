# ============================================
# ForestGaps Docker Image - CUDA + Miniforge
# ============================================
# Approche propre : Image CUDA pure + tout via mamba
# Mamba résout les dépendances 10-100x plus vite que conda classic
# Pas de conflit avec GDAL pré-installé
# ============================================

# Image de base CUDA pure (pas de pollution avec apt packages)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="vdlinden.arthur@gmail.com"
LABEL description="ForestGaps - Forest canopy gap detection using deep learning"
LABEL version="0.1.1"
LABEL gdal.version="3.8.x"
LABEL pytorch.version="2.4.x"
LABEL cuda.version="12.4"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ============================================
# Install System Dependencies
# ============================================
# Force fast mirrors and minimal dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Install Miniforge (includes mamba - much faster than conda)
# ============================================
RUN curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

# Add conda/mamba to PATH
ENV PATH="/opt/conda/bin:$PATH"

# ============================================
# Install ALL Dependencies via Mamba (ONE command for global resolution)
# ============================================
# Mamba résout tout d'un coup - résolution en secondes au lieu de minutes/heures
# Ordre des channels : pytorch > nvidia > conda-forge
# Python 3.10 pour compatibilité avec Google Colab
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba install -y \
    -c pytorch -c nvidia -c conda-forge \
    # Python (3.10 = Colab version)
    python=3.10 \
    # PyTorch + CUDA
    "pytorch>=2.4.0,<2.5" \
    "torchvision>=0.19.0,<0.20" \
    pytorch-cuda=12.4 \
    # Geospatial
    "gdal>=3.8,<3.9" \
    "rasterio>=1.3,<1.4" \
    "geopandas>=0.14,<0.15" \
    "fiona>=1.9,<2.0" \
    "shapely>=2.0,<3.0" \
    # ML & Scientific
    "scikit-learn>=1.3,<1.5" \
    "scikit-image>=0.22,<0.23" \
    "numpy>=1.26,<2.0" \
    "pandas>=2.1,<3.0" \
    "scipy>=1.11,<1.12" \
    # Visualization
    "matplotlib>=3.8,<3.9" \
    "seaborn>=0.13,<0.14" \
    "pillow>=10.1,<11.0" \
    "opencv>=4.8,<5.0" \
    # Utils
    "pydantic>=2.5,<3.0" \
    "pyyaml>=6.0,<7.0" \
    "tensorboard>=2.15,<3.0" \
    "tqdm>=4.66,<5.0" \
    "tabulate>=0.9,<1.0" \
    # Dev tools
    "pytest>=7.4,<8.0" \
    "black>=23.12,<25.0" \
    "isort>=5.13,<6.0" \
    "flake8>=6.1,<7.0" \
    && mamba clean -afy

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
