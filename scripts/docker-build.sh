#!/bin/bash
# ============================================
# ForestGaps Docker Build Script
# ============================================
# Builds Docker images for production and development
#
# Usage:
#   ./scripts/docker-build.sh
#   ./scripts/docker-build.sh --tag v1.0.0
# ============================================

set -e  # Exit on error

# Determine script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TARGET="development"
TAG="latest"
PLATFORM="linux/amd64"
NO_CACHE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --help|-h)
            echo "ForestGaps Docker Build Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --target TARGET     Build target (development) [default: development]"
            echo "  --tag TAG          Image tag [default: latest]"
            echo "  --platform PLATFORM Build platform [default: linux/amd64]"
            echo "  --no-cache         Build without using cache"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Build latest development image"
            echo "  $0 --tag v1.0.0              # Build with specific tag"
            echo "  $0 --no-cache                # Force rebuild without cache"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate target
if [[ "$TARGET" != "development" ]]; then
    echo -e "${RED}Invalid target: $TARGET${NC}"
    echo "Valid targets: development"
    exit 1
fi

# Set image name
IMAGE_NAME="forestgaps:$TAG"

# Print build configuration
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}ForestGaps Docker Build${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Target:   ${GREEN}$TARGET${NC}"
echo -e "  Tag:      ${GREEN}$TAG${NC}"
echo -e "  Platform: ${GREEN}$PLATFORM${NC}"
echo -e "  No Cache: ${GREEN}$NO_CACHE${NC}"
echo -e "  Image:    ${GREEN}$IMAGE_NAME${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Build cache flags
CACHE_FLAGS=""
if [ "$NO_CACHE" = true ]; then
    CACHE_FLAGS="--no-cache"
    echo -e "${YELLOW}Building without cache...${NC}"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Build the image
echo -e "${YELLOW}Building Docker image: $IMAGE_NAME${NC}"
echo ""

BUILD_START=$(date +%s)

DOCKER_BUILDKIT=1 docker build \
    --platform "$PLATFORM" \
    --target "$TARGET" \
    -t "$IMAGE_NAME" \
    -f docker/Dockerfile \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    $CACHE_FLAGS \
    . 2>&1 | tee /tmp/docker-build-$$.log

BUILD_EXIT_CODE=${PIPESTATUS[0]}
BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))

# Check build result
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo -e "Image: ${GREEN}$IMAGE_NAME${NC}"
    echo -e "Build duration: ${GREEN}${BUILD_DURATION}s${NC}"
    echo ""

    # Show image details
    echo -e "${YELLOW}Image details:${NC}"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""

    # Show image layers (top 10)
    echo -e "${YELLOW}Image layers (top 10):${NC}"
    docker history "$IMAGE_NAME" --format "table {{.CreatedBy}}\t{{.Size}}" --no-trunc | head -11
    echo ""

    # Quick verification
    echo -e "${YELLOW}Quick verification:${NC}"
    if docker run --rm "$IMAGE_NAME" python -c "import torch, rasterio, geopandas, forestgaps; print('✅ All imports OK')"; then
        echo -e "${GREEN}✅ Quick import test passed${NC}"
    else
        echo -e "${RED}⚠️ Quick import test failed${NC}"
    fi
    echo ""

    # Suggest next steps
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "1. Run tests:    ${GREEN}./scripts/docker-test.sh${NC}"
    echo -e "2. Run shell:    ${GREEN}./scripts/docker-run.sh shell${NC}"
    echo -e "3. Start Jupyter: ${GREEN}./scripts/docker-run.sh jupyter${NC}"
    echo -e "4. Train model:  ${GREEN}./scripts/docker-run.sh train${NC}"
    echo ""

    # Clean up log
    rm -f /tmp/docker-build-$$.log

    exit 0
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}❌ Build failed${NC}"
    echo -e "${RED}============================================${NC}"
    echo -e "Build duration: ${RED}${BUILD_DURATION}s${NC}"
    echo ""
    echo -e "${YELLOW}Check the build log for details:${NC}"
    echo -e "  cat /tmp/docker-build-$$.log"
    echo ""
    echo -e "${YELLOW}Common issues:${NC}"
    echo -e "  - Missing dependencies in requirements files"
    echo -e "  - GDAL version mismatch"
    echo -e "  - Network issues during package download"
    echo -e "  - Insufficient disk space"
    echo ""
    echo -e "${YELLOW}Try rebuilding without cache:${NC}"
    echo -e "  $0 --no-cache"
    echo ""

    exit 1
fi
