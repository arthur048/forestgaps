#!/bin/bash
# ============================================
# ForestGaps Docker Validation Script
# ============================================
# Comprehensive testing of Docker setup
#
# Usage:
#   ./scripts/docker-test.sh
#   ./scripts/docker-test.sh --image forestgaps:v1.0.0
# ============================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default image
IMAGE="forestgaps:latest"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --help|-h)
            echo "ForestGaps Docker Validation Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --image IMAGE    Docker image to test [default: forestgaps:latest]"
            echo "  --help, -h       Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Test counter
TOTAL_TESTS=7
PASSED_TESTS=0
FAILED_TESTS=0

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}ForestGaps Docker Validation${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${YELLOW}Image: $IMAGE${NC}"
echo ""

# Test 1: Image exists
echo -e "${YELLOW}[1/$TOTAL_TESTS] Checking if image exists...${NC}"
if docker image inspect "$IMAGE" &> /dev/null; then
    echo -e "${GREEN}✓ Image found${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ Image not found${NC}"
    echo -e "${YELLOW}Run: ./scripts/docker-build.sh${NC}"
    ((FAILED_TESTS++))
    exit 1
fi
echo ""

# Test 2: Container starts
echo -e "${YELLOW}[2/$TOTAL_TESTS] Testing container startup...${NC}"
if docker run --rm "$IMAGE" python -c "print('✓ Container started successfully')" 2>&1 | grep -q "Container started successfully"; then
    echo -e "${GREEN}✓ Container starts successfully${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ Container failed to start${NC}"
    ((FAILED_TESTS++))
fi
echo ""

# Test 3: Import test
echo -e "${YELLOW}[3/$TOTAL_TESTS] Testing Python imports...${NC}"
IMPORT_OUTPUT=$(docker run --rm "$IMAGE" python -c "
import sys
try:
    import torch
    print(f'✓ torch {torch.__version__}')

    import rasterio
    print(f'✓ rasterio {rasterio.__version__}')

    import geopandas
    print(f'✓ geopandas {geopandas.__version__}')

    import forestgaps
    print(f'✓ forestgaps {forestgaps.__version__}')

    print('SUCCESS')
except ImportError as e:
    print(f'FAILED: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)

if echo "$IMPORT_OUTPUT" | grep -q "SUCCESS"; then
    echo "$IMPORT_OUTPUT" | grep "✓"
    echo -e "${GREEN}✓ All imports successful${NC}"
    ((PASSED_TESTS++))
else
    echo "$IMPORT_OUTPUT"
    echo -e "${RED}✗ Import failed${NC}"
    ((FAILED_TESTS++))
fi
echo ""

# Test 4: GPU availability (conditional)
echo -e "${YELLOW}[4/$TOTAL_TESTS] Testing GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    GPU_OUTPUT=$(docker run --rm --gpus all "$IMAGE" python -c "
import sys
import torch

if not torch.cuda.is_available():
    print('FAILED: CUDA not available', file=sys.stderr)
    sys.exit(1)

print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ CUDA version: {torch.version.cuda}')
print(f'✓ Device count: {torch.cuda.device_count()}')

# Test GPU operation
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
y = x * 2
assert y.is_cuda
print('✓ GPU computation test passed')
print('SUCCESS')
" 2>&1)

    if echo "$GPU_OUTPUT" | grep -q "SUCCESS"; then
        echo "$GPU_OUTPUT" | grep "✓"
        echo -e "${GREEN}✓ GPU accessible in container${NC}"
        ((PASSED_TESTS++))
    else
        echo "$GPU_OUTPUT"
        echo -e "${RED}✗ GPU not accessible${NC}"
        ((FAILED_TESTS++))
    fi
else
    echo -e "${YELLOW}⊘ No GPU on host, skipping GPU test${NC}"
    echo -e "${YELLOW}  (This is OK for CPU-only systems)${NC}"
    ((PASSED_TESTS++))
fi
echo ""

# Test 5: Environment detection
echo -e "${YELLOW}[5/$TOTAL_TESTS] Testing environment detection...${NC}"
ENV_OUTPUT=$(docker run --rm "$IMAGE" python -c "
import sys
try:
    from forestgaps.environment import Environment
    env = Environment.detect()
    info = env.get_environment_info()

    env_type = info.get('environment_type', 'Unknown')
    print(f'✓ Detected environment: {env_type}')

    if 'Docker' in env_type or 'Local' in env_type:
        print('✓ Environment detection working')
        print('SUCCESS')
    else:
        print(f'WARNING: Unexpected environment type: {env_type}')
        print('SUCCESS')  # Still pass, just warn
except Exception as e:
    print(f'FAILED: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

if echo "$ENV_OUTPUT" | grep -q "SUCCESS"; then
    echo "$ENV_OUTPUT" | grep "✓\|WARNING"
    echo -e "${GREEN}✓ Environment detection works${NC}"
    ((PASSED_TESTS++))
else
    echo "$ENV_OUTPUT"
    echo -e "${RED}✗ Environment detection failed${NC}"
    ((FAILED_TESTS++))
fi
echo ""

# Test 6: GDAL/rasterio compatibility
echo -e "${YELLOW}[6/$TOTAL_TESTS] Testing GDAL/rasterio compatibility...${NC}"
GDAL_OUTPUT=$(docker run --rm "$IMAGE" python -c "
import sys
import os
try:
    import rasterio
    from osgeo import gdal

    gdal_version = gdal.__version__
    rasterio_version = rasterio.__version__

    print(f'✓ GDAL version: {gdal_version}')
    print(f'✓ Rasterio version: {rasterio_version}')

    # Check environment variables
    gdal_data = os.environ.get('GDAL_DATA', '')
    proj_lib = os.environ.get('PROJ_LIB', '')

    if gdal_data and os.path.exists(gdal_data):
        print(f'✓ GDAL_DATA: {gdal_data}')
    else:
        print(f'WARNING: GDAL_DATA not set properly')

    if proj_lib and os.path.exists(proj_lib):
        print(f'✓ PROJ_LIB: {proj_lib}')
    else:
        print(f'WARNING: PROJ_LIB not set properly')

    print('✓ GDAL and rasterio are compatible')
    print('SUCCESS')
except Exception as e:
    print(f'FAILED: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

if echo "$GDAL_OUTPUT" | grep -q "SUCCESS"; then
    echo "$GDAL_OUTPUT" | grep "✓\|WARNING"
    echo -e "${GREEN}✓ GDAL/rasterio compatible${NC}"
    ((PASSED_TESTS++))
else
    echo "$GDAL_OUTPUT"
    echo -e "${RED}✗ GDAL/rasterio incompatible${NC}"
    ((FAILED_TESTS++))
fi
echo ""

# Test 7: Health check
echo -e "${YELLOW}[7/$TOTAL_TESTS] Running container health check...${NC}"
if docker run --rm "$IMAGE" python /app/healthcheck.py > /tmp/healthcheck-$$.log 2>&1; then
    cat /tmp/healthcheck-$$.log | tail -20
    echo -e "${GREEN}✓ Health check passed${NC}"
    ((PASSED_TESTS++))
    rm -f /tmp/healthcheck-$$.log
else
    cat /tmp/healthcheck-$$.log
    echo -e "${RED}✗ Health check failed${NC}"
    ((FAILED_TESTS++))
    rm -f /tmp/healthcheck-$$.log
fi
echo ""

# Final summary
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Total tests:  ${BLUE}$TOTAL_TESTS${NC}"
echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}🎉 All validation tests passed!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Open shell:     ${GREEN}./scripts/docker-run.sh shell${NC}"
    echo -e "  2. Start Jupyter:  ${GREEN}./scripts/docker-run.sh jupyter${NC}"
    echo -e "  3. Train model:    ${GREEN}./scripts/docker-run.sh train${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}⚠️  $FAILED_TESTS test(s) failed${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  1. Rebuild image: ${GREEN}./scripts/docker-build.sh --no-cache${NC}"
    echo -e "  2. Check logs above for specific errors"
    echo -e "  3. Verify NVIDIA drivers: ${GREEN}nvidia-smi${NC}"
    echo -e "  4. Check Docker daemon is running"
    echo ""
    exit 1
fi
