# ============================================
# ForestGaps Docker Image - NGC PyTorch 25.03
# ============================================
# 
# POURQUOI 25.03 ?
# - Inclut /etc/pip/constraint.txt natif (empêche les upgrades NumPy)
# - PyTorch 2.7 compatible NumPy 2.x si besoin
# - Python 3.12, CUDA 12.8, cuDNN 9.8
# - Ubuntu 24.04
#
# ATTENTION: Volta (V100) n'est plus supporté depuis 25.01
# Pour Volta, utiliser NGC 24.12 ou NVIDIA AI Enterprise
# ============================================

FROM nvcr.io/nvidia/pytorch:25.03-py3

LABEL maintainer="vdlinden.arthur@gmail.com"
LABEL description="ForestGaps - Forest canopy gap detection using deep learning"
LABEL version="0.2.0"

# ============================================
# Environment Variables
# ============================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ============================================
# Vérifier les contraintes NGC existantes
# ============================================
# NGC 25.03 a déjà /etc/pip/constraint.txt
# On peut voir son contenu ou l'étendre si besoin
RUN echo "=== NGC pip constraints ===" && \
    head -50 /etc/pip/constraint.txt || echo "No constraint file found"

# ============================================
# Install Geospatial Packages
# ============================================
# Les contraintes NGC empêchent les upgrades de NumPy
RUN pip install \
    rasterio \
    geopandas \
    fiona \
    shapely \
    pyproj \
    rioxarray \
    xarray

# ============================================
# Install Additional ML/Science Packages
# ============================================
# opencv-python-headless évite les conflits avec DALI
RUN pip install \
    scikit-image \
    opencv-python-headless \
    pydantic \
    pyyaml \
    tqdm \
    tabulate \
    seaborn

# ============================================
# Install Dev Tools
# ============================================
RUN pip install \
    pytest \
    black \
    isort \
    flake8

# ============================================
# Setup Working Directory
# ============================================
WORKDIR /app
RUN mkdir -p /app/data /app/models /app/outputs /app/logs /app/tests

# ============================================
# Install ForestGaps Package
# ============================================
COPY forestgaps/ /app/forestgaps/
COPY setup.py README.md ./
COPY pytest.ini ./
COPY MANIFEST.in* ./

RUN pip install -e .

COPY tests/ /app/tests/
COPY scripts/ /app/scripts/
COPY docker/healthcheck.py /app/healthcheck.py

# ============================================
# Verify Installation
# ============================================
RUN python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" && \
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" && \
    python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import rasterio; print(f'✓ Rasterio {rasterio.__version__}')" && \
    python -c "import geopandas; print(f'✓ GeoPandas {geopandas.__version__}')" && \
    python -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')" && \
    python -c "import torch; t = torch.tensor([1.0]).numpy(); print('✓ torch.numpy() works')"

# ============================================
# Runtime Configuration
# ============================================
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_HOME=/app/.torch \
    # Architectures: Turing -> Hopper (Volta retiré en 25.01)
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

EXPOSE 6006 8888

CMD ["python", "-c", "from forestgaps.environment import Environment; env = Environment.detect(); import pprint; pprint.pprint(env.get_environment_info())"]