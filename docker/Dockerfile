# ============================================
# ForestGaps Docker Image - Conda Edition
# ============================================
# Simplified Dockerfile using conda-forge for GDAL + geospatial dependencies
# Combines PyTorch (CUDA 12.4) + GDAL 3.8.5 for geospatial deep learning
#
# Build: docker build -f docker/Dockerfile -t forestgaps:latest .
# Run:   docker run --gpus all -v ./data:/app/data forestgaps:latest
# ============================================

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

LABEL maintainer="vdlinden.arthur@gmail.com"
LABEL description="ForestGaps - Forest canopy gap detection using deep learning"
LABEL version="0.1.1"
LABEL python.version="3.11"
LABEL cuda.version="12.4"
LABEL pytorch.version="2.4.0"
LABEL gdal.version="3.8.5"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Install GDAL + Geospatial Stack via Conda-Forge
# ============================================
# This single command handles ALL dependencies automatically:
# - GDAL with all its system libraries (proj, geos, xerces-c, etc.)
# - Rasterio, GeoPandas, Fiona, Shapely
# - No manual library copying needed!
RUN conda install -c conda-forge -y \
    gdal=3.8.5 \
    rasterio=1.3.9 \
    geopandas=0.14.2 \
    fiona=1.9.5 \
    shapely=2.0.2 \
    && conda clean -afy

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================
# Install Python Dependencies
# ============================================
WORKDIR /tmp/build

# Copy requirements files
COPY requirements/requirements.txt requirements.txt
COPY requirements/requirements-gpu.txt requirements-gpu.txt
COPY requirements/requirements-dev.txt requirements-dev.txt

# Install remaining dependencies
# (GDAL, rasterio, geopandas already installed via conda)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-gpu.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# ============================================
# Verify Critical Imports
# ============================================
RUN python -c "from osgeo import gdal; print(f'✓ GDAL {gdal.__version__}')" && \
    python -c "import rasterio; print(f'✓ Rasterio {rasterio.__version__}')" && \
    python -c "import geopandas; print(f'✓ GeoPandas {geopandas.__version__}')" && \
    python -c "import torch; print(f'✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ============================================
# Install ForestGaps Package
# ============================================
WORKDIR /app

# Copy project files
COPY forestgaps/ /app/forestgaps/
COPY setup.py /app/
COPY README.md /app/
COPY MANIFEST.in /app/
COPY pytest.ini /app/

# Install in editable mode
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/outputs /app/logs /app/tests

# Copy Docker-specific files
COPY docker/healthcheck.py /app/healthcheck.py

# Copy tests
COPY tests/ /app/tests/

# ============================================
# Security: Non-Root User
# ============================================
RUN useradd -m -u 1000 -s /bin/bash forestgaps && \
    chown -R forestgaps:forestgaps /app

USER forestgaps
WORKDIR /app

# ============================================
# GPU Optimization
# ============================================
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_HOME=/app/.torch \
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
