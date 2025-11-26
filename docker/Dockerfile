# ============================================
# ForestGaps Docker Image - NGC PyTorch Base
# ============================================
# Approche optimale : Image NGC pré-construite + wheels pip
# Build time : ~2-3 min (vs 30+ min avec conda)
# 
# L'image NGC inclut déjà :
#   - PyTorch 2.5 + CUDA 12.6 + cuDNN 9.3
#   - NumPy, SciPy, Pandas
#   - TensorBoard, JupyterLab
#   - OpenCV (via DALI)
#   - NCCL, TensorRT, etc.
# ============================================

FROM nvcr.io/nvidia/pytorch:24.08-py3

LABEL maintainer="vdlinden.arthur@gmail.com"
LABEL description="ForestGaps - Forest canopy gap detection using deep learning"
LABEL version="0.1.1"

# ============================================
# Environment Variables
# ============================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ============================================
# Install Geospatial Packages (wheels only, no compilation)
# ============================================
# Les wheels manylinux incluent leurs propres libs (GDAL, GEOS, PROJ)
# Rasterio installe automatiquement GDAL comme dépendance
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
# Certains sont déjà dans NGC mais on s'assure des versions
RUN pip install \
    scikit-image \
    pydantic \
    pyyaml \
    tqdm \
    tabulate \
    seaborn

# ============================================
# Install Dev Tools (optional, comment out for production)
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

# Create directories
RUN mkdir -p /app/data /app/models /app/outputs /app/logs /app/tests

# ============================================
# Install ForestGaps Package
# ============================================
# Copy project files
COPY forestgaps/ /app/forestgaps/
COPY setup.py README.md ./
COPY pytest.ini ./

# Copy optional files if they exist
COPY MANIFEST.in* ./

# Install in editable mode
RUN pip install -e .

# Copy tests and health check
COPY tests/ /app/tests/
COPY docker/healthcheck.py /app/healthcheck.py

# ============================================
# Verify Installation
# ============================================
RUN python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" && \
    python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import rasterio; print(f'✓ Rasterio {rasterio.__version__}')" && \
    python -c "import geopandas; print(f'✓ GeoPandas {geopandas.__version__}')" && \
    python -c "from osgeo import gdal; print(f'✓ GDAL {gdal.__version__}')"

# ============================================
# Security: Non-Root User
# ============================================
# Note: Commenté car l'image NGC utilise root par défaut
# et certaines opérations CUDA peuvent nécessiter root
# Décommente si tu veux un user non-root :
#
# RUN useradd -m -u 1000 -s /bin/bash forestgaps && \
#     chown -R forestgaps:forestgaps /app
# USER forestgaps

# ============================================
# Runtime Configuration
# ============================================
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_HOME=/app/.torch \
    # Architectures GPU supportées (Volta -> Hopper)
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# ============================================
# Health Check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

# ============================================
# Expose Ports
# ============================================
EXPOSE 6006 8888

# ============================================
# Default Command
# ============================================
CMD ["python", "-c", "from forestgaps.environment import Environment; env = Environment.detect(); import pprint; pprint.pprint(env.get_environment_info())"]