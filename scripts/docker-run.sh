#!/bin/bash
# ============================================
# ForestGaps Docker Run Script
# ============================================
# Provides convenient commands for different use cases
#
# Usage:
#   ./scripts/docker-run.sh shell
#   ./scripts/docker-run.sh train --data-dir ./data
#   ./scripts/docker-run.sh jupyter
# ============================================

set -e

# Determine script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE="forestgaps:latest"
GPU_MODE="auto"  # auto, enabled, disabled
DATA_DIR=""
MODELS_DIR=""
OUTPUTS_DIR=""
LOGS_DIR=""

show_help() {
    cat << EOF
${BLUE}ForestGaps Docker Run Script${NC}

${YELLOW}Usage:${NC}
  $0 <mode> [OPTIONS] [ARGS...]

${YELLOW}Modes:${NC}
  shell         Open interactive shell in container
  jupyter       Start Jupyter notebook server
  train         Train a model
  inference     Run inference on new data
  evaluate      Evaluate a trained model
  preprocess    Preprocess raster data
  test          Run test suite
  healthcheck   Run health check

${YELLOW}Options:${NC}
  --image IMAGE       Docker image to use [default: forestgaps:latest]
  --gpu MODE         GPU mode: auto|enabled|disabled [default: auto]
  --data-dir DIR     Mount data directory
  --models-dir DIR   Mount models directory
  --outputs-dir DIR  Mount outputs directory
  --logs-dir DIR     Mount logs directory
  --help, -h         Show this help

${YELLOW}Examples:${NC}
  $0 shell
  $0 shell --gpu disabled
  $0 jupyter
  $0 train --data-dir ./data --models-dir ./models
  $0 inference --data-dir ./data --models-dir ./models
  $0 test
  $0 healthcheck

${YELLOW}Notes:${NC}
  - GPU is auto-detected by default (uses GPU if available)
  - Default directories are created in project root if not specified
  - Use Ctrl+C to stop Jupyter or training
EOF
}

# Parse mode
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

MODE="$1"
shift

# Parse options
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --gpu)
            GPU_MODE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --outputs-dir)
            OUTPUTS_DIR="$2"
            shift 2
            ;;
        --logs-dir)
            LOGS_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Determine GPU flags
GPU_FLAGS=""
if [ "$GPU_MODE" == "enabled" ]; then
    GPU_FLAGS="--gpus all --runtime nvidia"
    echo -e "${GREEN}✓ GPU mode: enabled${NC}"
elif [ "$GPU_MODE" == "disabled" ]; then
    echo -e "${YELLOW}✓ GPU mode: disabled${NC}"
elif [ "$GPU_MODE" == "auto" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        GPU_FLAGS="--gpus all --runtime nvidia"
        echo -e "${GREEN}✓ GPU mode: auto (GPU detected and enabled)${NC}"
    else
        echo -e "${YELLOW}✓ GPU mode: auto (no GPU detected, using CPU)${NC}"
    fi
else
    echo -e "${RED}Invalid GPU mode: $GPU_MODE${NC}"
    exit 1
fi

# Build volume mounts
VOLUME_FLAGS=""

# Data directory
if [ -n "$DATA_DIR" ]; then
    DATA_DIR_ABS=$(realpath "$DATA_DIR" 2>/dev/null || echo "$DATA_DIR")
    VOLUME_FLAGS="$VOLUME_FLAGS -v $DATA_DIR_ABS:/app/data:ro"
    echo -e "${GREEN}✓ Data dir:    $DATA_DIR_ABS${NC}"
else
    mkdir -p "$PROJECT_ROOT/data"
    VOLUME_FLAGS="$VOLUME_FLAGS -v $PROJECT_ROOT/data:/app/data:ro"
    echo -e "${YELLOW}✓ Data dir:    $PROJECT_ROOT/data (default)${NC}"
fi

# Models directory
if [ -n "$MODELS_DIR" ]; then
    MODELS_DIR_ABS=$(realpath "$MODELS_DIR" 2>/dev/null || echo "$MODELS_DIR")
    VOLUME_FLAGS="$VOLUME_FLAGS -v $MODELS_DIR_ABS:/app/models:rw"
    echo -e "${GREEN}✓ Models dir:  $MODELS_DIR_ABS${NC}"
else
    mkdir -p "$PROJECT_ROOT/models"
    VOLUME_FLAGS="$VOLUME_FLAGS -v $PROJECT_ROOT/models:/app/models:rw"
    echo -e "${YELLOW}✓ Models dir:  $PROJECT_ROOT/models (default)${NC}"
fi

# Outputs directory
if [ -n "$OUTPUTS_DIR" ]; then
    OUTPUTS_DIR_ABS=$(realpath "$OUTPUTS_DIR" 2>/dev/null || echo "$OUTPUTS_DIR")
    VOLUME_FLAGS="$VOLUME_FLAGS -v $OUTPUTS_DIR_ABS:/app/outputs:rw"
    echo -e "${GREEN}✓ Outputs dir: $OUTPUTS_DIR_ABS${NC}"
else
    mkdir -p "$PROJECT_ROOT/outputs"
    VOLUME_FLAGS="$VOLUME_FLAGS -v $PROJECT_ROOT/outputs:/app/outputs:rw"
    echo -e "${YELLOW}✓ Outputs dir: $PROJECT_ROOT/outputs (default)${NC}"
fi

# Logs directory
if [ -n "$LOGS_DIR" ]; then
    LOGS_DIR_ABS=$(realpath "$LOGS_DIR" 2>/dev/null || echo "$LOGS_DIR")
    VOLUME_FLAGS="$VOLUME_FLAGS -v $LOGS_DIR_ABS:/app/logs:rw"
    echo -e "${GREEN}✓ Logs dir:    $LOGS_DIR_ABS${NC}"
else
    mkdir -p "$PROJECT_ROOT/logs"
    VOLUME_FLAGS="$VOLUME_FLAGS -v $PROJECT_ROOT/logs:/app/logs:rw"
    echo -e "${YELLOW}✓ Logs dir:    $PROJECT_ROOT/logs (default)${NC}"
fi

# Common docker run flags
COMMON_FLAGS="--rm -it $GPU_FLAGS $VOLUME_FLAGS --shm-size=8g"

# Execute based on mode
echo ""
case $MODE in
    shell)
        echo -e "${BLUE}Opening interactive shell...${NC}"
        docker run $COMMON_FLAGS \
            "$IMAGE" \
            /bin/bash
        ;;

    jupyter)
        PORT="${EXTRA_ARGS[0]:-8888}"
        TOKEN="${EXTRA_ARGS[1]:-forestgaps}"
        echo -e "${BLUE}Starting Jupyter notebook server...${NC}"
        echo -e "${YELLOW}Access at: http://localhost:$PORT${NC}"
        echo -e "${YELLOW}Token: $TOKEN${NC}"
        echo ""
        docker run $COMMON_FLAGS \
            -p "$PORT:$PORT" \
            -v "$PROJECT_ROOT:/app/workspace:rw" \
            "$IMAGE" \
            jupyter notebook \
                --ip=0.0.0.0 \
                --port=$PORT \
                --no-browser \
                --allow-root \
                --NotebookApp.token=$TOKEN
        ;;

    train)
        echo -e "${BLUE}Training model...${NC}"
        docker run $COMMON_FLAGS \
            "$IMAGE" \
            python -m forestgaps.cli.training_cli "${EXTRA_ARGS[@]}"
        ;;

    inference)
        echo -e "${BLUE}Running inference...${NC}"
        docker run $COMMON_FLAGS \
            "$IMAGE" \
            python -m forestgaps.inference.core "${EXTRA_ARGS[@]}"
        ;;

    evaluate)
        echo -e "${BLUE}Evaluating model...${NC}"
        docker run $COMMON_FLAGS \
            "$IMAGE" \
            python -m forestgaps.evaluation.core "${EXTRA_ARGS[@]}"
        ;;

    preprocess)
        echo -e "${BLUE}Preprocessing data...${NC}"
        docker run $COMMON_FLAGS \
            "$IMAGE" \
            python -m forestgaps.cli.preprocessing_cli "${EXTRA_ARGS[@]}"
        ;;

    test)
        echo -e "${BLUE}Running test suite...${NC}"
        docker run $COMMON_FLAGS \
            -v "$PROJECT_ROOT/tests:/app/tests:ro" \
            "$IMAGE" \
            pytest tests/ -v "${EXTRA_ARGS[@]}"
        ;;

    healthcheck)
        echo -e "${BLUE}Running health check...${NC}"
        docker run $COMMON_FLAGS \
            "$IMAGE" \
            python /app/healthcheck.py
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Command completed successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Command failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
