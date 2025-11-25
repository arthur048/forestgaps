# ============================================
# ForestGaps Docker Image
# ============================================
# Multi-stage Dockerfile for forest canopy gap detection
# Combines PyTorch (CUDA 12.4) + GDAL 3.8.5 for geospatial deep learning
#
# Build: docker build -f docker/Dockerfile -t forestgaps:latest .
# Run:   docker run --gpus all -v ./data:/app/data forestgaps:latest
# ============================================

# ============================================
# Stage 1: GDAL Builder
# ============================================
# Provides pre-compiled GDAL 3.8.5 with system libraries
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.5 AS gdal-builder

LABEL stage=builder
LABEL description="GDAL system libraries builder"

# Install Python 3.10 and development headers
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: PyTorch Base with GDAL
# ============================================
# Python 3.10 + CUDA 12.4 + cuDNN 9 + PyTorch 2.4.0
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS base

LABEL maintainer="vdlinden.arthur@gmail.com"
LABEL description="ForestGaps - Forest canopy gap detection using deep learning"
LABEL version="0.1.1"
LABEL python.version="3.10"
LABEL cuda.version="12.4"
LABEL pytorch.version="2.4.0"
LABEL gdal.version="3.8.5"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy GDAL libraries and binaries from builder stage
COPY --from=gdal-builder /usr/lib/x86_64-linux-gnu/libgdal* /usr/lib/x86_64-linux-gnu/
COPY --from=gdal-builder /usr/lib/x86_64-linux-gnu/libproj* /usr/lib/x86_64-linux-gnu/
COPY --from=gdal-builder /usr/lib/x86_64-linux-gnu/libgeos* /usr/lib/x86_64-linux-gnu/
COPY --from=gdal-builder /usr/lib/x86_64-linux-gnu/libspatialite* /usr/lib/x86_64-linux-gnu/
COPY --from=gdal-builder /usr/lib/x86_64-linux-gnu/libspatialindex* /usr/lib/x86_64-linux-gnu/
COPY --from=gdal-builder /usr/lib/x86_64-linux-gnu/libgeotiff* /usr/lib/x86_64-linux-gnu/
COPY --from=gdal-builder /usr/share/gdal /usr/share/gdal
COPY --from=gdal-builder /usr/share/proj /usr/share/proj
COPY --from=gdal-builder /usr/bin/gdal* /usr/bin/
COPY --from=gdal-builder /usr/bin/ogr* /usr/bin/
COPY --from=gdal-builder /usr/bin/proj* /usr/bin/

# Copy ALL GDAL development headers (needed for pip install GDAL)
# Using multiple patterns to ensure we get all GDAL-related headers
COPY --from=gdal-builder /usr/include/gdal*.h /usr/include/
COPY --from=gdal-builder /usr/include/cpl*.h /usr/include/
COPY --from=gdal-builder /usr/include/ogr*.h /usr/include/
COPY --from=gdal-builder /usr/include/gnm*.h /usr/include/
COPY --from=gdal-builder /usr/include/*dataset.h /usr/include/

# Set GDAL environment variables (CRITICAL for rasterio)
ENV GDAL_DATA=/usr/share/gdal \
    PROJ_LIB=/usr/share/proj \
    GDAL_CONFIG=/usr/bin/gdal-config \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Install essential system dependencies including build tools for GDAL Python bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libspatialindex-dev \
    git \
    curl \
    ca-certificates \
    build-essential \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 3: Dependencies Installation
# ============================================
FROM base AS dependencies

LABEL stage=dependencies
LABEL description="Python dependencies installation"

WORKDIR /tmp/build

# Copy requirements files
COPY requirements/requirements.txt requirements.txt
COPY requirements/requirements-gpu.txt requirements-gpu.txt
COPY requirements/requirements-dev.txt requirements-dev.txt

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ============================================
# CRITICAL: Install dependencies in THIS ORDER to avoid conflicts
# ============================================

# 1. Install GDAL Python bindings (MUST match system GDAL version)
RUN GDAL_VERSION=$(gdal-config --version) && \
    echo "Installing GDAL Python bindings version: $GDAL_VERSION" && \
    pip install --no-cache-dir GDAL==$GDAL_VERSION

# 2. Install geospatial stack (depends on GDAL)
# Note: Using versions compatible with GDAL 3.8.5
RUN pip install --no-cache-dir \
    rasterio==1.3.9 \
    shapely==2.0.2 \
    fiona==1.9.5 \
    geopandas==0.14.2

# 3. Install remaining production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 4. Install GPU-specific dependencies (if any)
RUN pip install --no-cache-dir -r requirements-gpu.txt

# 5. Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Verify critical imports work
RUN python -c "from osgeo import gdal; print(f'GDAL version: {gdal.__version__}')" && \
    python -c "import rasterio; print(f'Rasterio version: {rasterio.__version__}')" && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ============================================
# Stage 4: Development Image
# ============================================
FROM dependencies AS development

LABEL stage=development
LABEL description="ForestGaps development image with all tools"

# Create application directory
WORKDIR /app

# Copy project files
COPY forestgaps/ /app/forestgaps/
COPY setup.py /app/
COPY README.md /app/
COPY MANIFEST.in /app/
COPY pytest.ini /app/

# Install ForestGaps package in editable mode
RUN pip install --no-cache-dir -e .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/models /app/outputs /app/logs /app/tests

# Copy Docker-specific files
COPY docker/healthcheck.py /app/healthcheck.py

# Copy tests for development
COPY tests/ /app/tests/

# Set up non-root user for security
RUN useradd -m -u 1000 -s /bin/bash forestgaps && \
    chown -R forestgaps:forestgaps /app

# Switch to non-root user
USER forestgaps

# Set working directory
WORKDIR /app

# Configure GPU optimization
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_HOME=/app/.torch \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python /app/healthcheck.py || exit 1

# Expose ports
EXPOSE 6006 8888

# Default command: show environment info
CMD ["python", "-c", "from forestgaps.environment import Environment; env = Environment.detect(); import pprint; pprint.pprint(env.get_environment_info())"]

# ============================================
# Build Instructions:
# ============================================
# Build development image:
#   docker build -f docker/Dockerfile --target development -t forestgaps:latest .
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
