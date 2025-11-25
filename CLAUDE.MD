# ForestGaps - Claude Code Context

This file provides essential context for Claude Code when working with the ForestGaps project.

## Project Overview

**ForestGaps** is a Python library for automatic detection and analysis of forest canopy gaps using deep learning. The library processes Digital Surface Models (DSM) and Canopy Height Models (CHM) to identify forest gaps, estimate canopy heights, and compare different deep learning approaches.

### Key Features
- Forest gap segmentation using multiple models (U-Net, DeepLabV3+, U-Net with FiLM)
- Canopy height estimation through regression
- Optimized geospatial data preprocessing
- Comprehensive model evaluation with specialized metrics
- Inference on new forest areas
- Model benchmarking capabilities
- Compatible with both local environments and Google Colab

## Project Structure

```
forestgaps/
├── __init__.py              # Package entry point
├── __version__.py           # Version definition
├── config/                  # Configuration management
│   ├── base.py             # Base configuration classes
│   └── schema.py           # Configuration schemas
├── environment/             # Environment detection and setup (Colab/local)
│   ├── base.py             # Abstract Environment class
│   ├── colab.py            # Google Colab environment
│   └── local.py            # Local environment
├── data/                    # Data processing and loading
│   ├── datasets/           # PyTorch datasets
│   ├── preprocessing/      # Raster preprocessing
│   ├── generation/         # Mask and tile generation
│   └── loaders/            # DataLoader creation
├── models/                  # Neural network architectures
│   ├── base.py             # Base model classes
│   ├── unet.py             # U-Net implementation
│   ├── unet_film.py        # U-Net with FiLM
│   ├── deeplabv3plus.py    # DeepLabV3+ implementation
│   └── registry.py         # Model registry pattern
├── training/                # Training logic
│   ├── trainer.py          # Main Trainer class
│   ├── losses.py           # Loss functions
│   ├── metrics.py          # Evaluation metrics
│   └── callbacks.py        # Training callbacks
├── evaluation/              # Model evaluation
│   ├── core.py             # Evaluation logic
│   └── metrics.py          # Evaluation metrics
├── inference/               # Model inference
│   ├── core.py             # Inference logic
│   └── postprocessing.py   # Post-processing
├── utils/                   # Utility functions
│   ├── visualization.py    # Visualization tools
│   ├── io.py               # I/O operations
│   ├── profiling.py        # Profiling tools
│   └── errors.py           # Error handling
├── cli/                     # Command-line interface
│   ├── preprocessing_cli.py
│   ├── training_cli.py
│   └── evaluate.py
├── benchmarking/            # Model comparison
│   ├── comparison.py       # Model comparison logic
│   ├── metrics.py          # Benchmark metrics
│   └── reporting.py        # Report generation
├── examples/                # Usage examples
├── tests/                   # Unit and integration tests
├── docker/                  # Docker configurations
└── scripts/                 # Utility scripts
```

## Core Concepts

### 1. Configuration System
- Centralized configuration via YAML files
- Validation using Pydantic schemas
- Default configurations for all modules
- Config inheritance and overrides

Key files:
- `config/base.py`: Base configuration classes
- `config/schema.py`: Configuration schemas

### 2. Environment Management
- Automatic detection of execution environment (Colab vs local)
- Environment-specific setup (GPU, paths, dependencies)
- Google Drive mounting for Colab
- Unified interface via abstract `Environment` class

Key files:
- `environment/base.py`: Abstract Environment class
- `environment/colab.py`: Colab-specific implementation
- `environment/local.py`: Local environment implementation

### 3. Data Processing Pipeline
1. Load DSM/CHM raster files
2. Preprocess (filtering, alignment, normalization)
3. Generate gap masks at different height thresholds
4. Create tiles for training
5. Build PyTorch datasets with augmentations

Key files:
- `data/preprocessing/`: Raster preprocessing functions
- `data/generation/`: Mask and tile generation
- `data/datasets/`: PyTorch dataset implementations
- `data/loaders/`: DataLoader creation

### 4. Model Architecture
Base classes:
- `BaseModel`: Abstract base for all models
- `SegmentationModel`: For gap segmentation tasks
- `RegressionModel`: For height estimation tasks

Implemented models:
- **UNet**: Standard U-Net architecture
- **UNetFiLM**: U-Net with Feature-wise Linear Modulation
- **DeepLabV3Plus**: DeepLabV3+ for segmentation
- **UNetRegression**: U-Net for regression tasks

Pattern: Registry pattern for model creation
```python
model = create_model("unet_film", **params)
```

Key files:
- `models/base.py`: Base model classes
- `models/unet.py`, `models/unet_film.py`, etc.: Model implementations
- `models/registry.py`: Model registration system

### 5. Training System
- Flexible Trainer class
- Customizable loss functions for gap detection
- Comprehensive metrics (IoU, F1, precision, recall)
- Callback system for monitoring (TensorBoard, checkpointing, early stopping)

Key files:
- `training/trainer.py`: Main Trainer class
- `training/losses.py`: Loss function implementations
- `training/metrics.py`: Metric implementations
- `training/callbacks.py`: Training callbacks

### 6. Evaluation and Inference
**Evaluation**: Test models on independent DSM/CHM pairs
- Single file evaluation
- Site-wide evaluation
- Multi-site evaluation
- Model comparison

**Inference**: Apply trained models to new data
- Tiled processing for large images
- Preserves geospatial metadata
- Batch inference support

Key files:
- `evaluation/core.py`: Evaluation logic
- `inference/core.py`: Inference logic

## Important Design Patterns

1. **Factory Pattern**: Model, optimizer, and dataset creation
2. **Registry Pattern**: Dynamic model registration
3. **Strategy Pattern**: Different training strategies
4. **Observer Pattern**: Training callbacks
5. **Template Method**: Base classes define algorithm skeleton

## Coding Standards

### Principles
- **SOLID principles**: Single responsibility, open/closed, etc.
- **PEP 8 compliance**: Python style guide
- **Type hints**: Static typing throughout
- **Comprehensive documentation**: Docstrings for all public APIs

### Error Handling
- Hierarchical error system in `utils/errors.py`
- Custom exceptions for different error types
- Graceful degradation where possible

## Common Workflows

### 1. Environment Setup
```python
from forestgaps.environment import setup_environment
env = setup_environment()  # Auto-detects Colab or local
```

### 2. Data Preprocessing
```python
from forestgaps.config import load_default_config
from forestgaps.data.preprocessing import process_raster_pair_robustly
from forestgaps.data.generation import create_gap_masks

config = load_default_config()
result = process_raster_pair_robustly(dsm_path, chm_path, site_name, config)
mask_paths = create_gap_masks(result["aligned_chm"], thresholds, output_dir, site_name)
```

### 3. Model Training
```python
from forestgaps.models import create_model
from forestgaps.data.loaders import create_data_loaders
from forestgaps.training import Trainer

config = load_default_config()
data_loaders = create_data_loaders(config)
model = create_model("unet_film")
trainer = Trainer(model, config, data_loaders['train'], data_loaders['val'], data_loaders['test'])
results = trainer.train(epochs=50)
```

### 4. Inference
```python
from forestgaps.inference import run_inference

result = run_inference(
    model_path="path/to/model.pt",
    dsm_path="path/to/new_dsm.tif",
    output_path="path/to/prediction.tif",
    threshold=5.0
)
```

### 5. Model Evaluation
```python
from forestgaps.evaluation import compare_models

models = {
    "unet": "path/to/unet.pt",
    "unet_film": "path/to/unet_film.pt"
}
results = compare_models(models, dsm_path, chm_path, output_dir, thresholds=[2.0, 5.0, 10.0])
```

## Development Guidelines

### When Adding New Models
1. Inherit from `BaseModel` or `SegmentationModel`/`RegressionModel`
2. Implement required methods: `forward()`, `predict()`
3. Register with the model registry using decorator
4. Add tests in `tests/models/`

### When Modifying Data Processing
1. Maintain geospatial metadata preservation
2. Ensure compatibility with both Colab and local environments
3. Add appropriate error handling
4. Update preprocessing pipeline documentation

### When Adding New Features
1. Follow SOLID principles
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests
5. Update relevant README files
6. Check backward compatibility

## Testing

Run tests with:
```bash
pytest tests/
```

Test structure mirrors package structure:
- `tests/models/`: Model tests
- `tests/data/`: Data processing tests
- `tests/training/`: Training tests
- `tests/evaluation/`: Evaluation tests

## Docker Support

Build and run with Docker:
```bash
bash scripts/docker-build.sh
bash scripts/docker-run.sh predict --model /app/models/model.pt --input /app/data/input.tif
```

## Dependencies

Core dependencies:
- **PyTorch** (>=1.8.0): Deep learning framework
- **Rasterio** (>=1.2.0): Geospatial raster I/O
- **GeoPandas** (>=0.10.0): Geospatial data handling
- **NumPy** (>=1.19.0): Numerical operations
- **TensorBoard** (>=2.8.0): Training visualization
- **Pydantic** (>=1.8.0): Configuration validation

See `setup.py` for complete list.

## Key Files Reference

### Configuration
- `config/base.py`: ConfigBase class with validation
- Default configs in each module's directory

### Models
- `models/base.py:BaseModel`: Abstract base class for all models
- `models/registry.py:MODEL_REGISTRY`: Registry for model registration
- `models/unet_film.py:UNetFiLM`: Recommended architecture

### Training
- `training/trainer.py:Trainer`: Main training class
- `training/callbacks.py`: TensorBoard, checkpointing, early stopping

### Data
- `data/preprocessing/`: Raster preprocessing functions
- `data/datasets/gap_dataset.py:ForestGapDataset`: Main dataset class
- `data/loaders/`: DataLoader creation utilities

### Evaluation & Inference
- `evaluation/core.py:ExternalEvaluator`: Evaluation on external data
- `inference/core.py:InferenceManager`: Inference on new data

## Additional Documentation

- `README.md`: User-facing documentation (French/English)
- `context_llm.md`: Comprehensive technical documentation for LLMs
- `environment/README.md`: Environment module documentation
- `evaluation/README.md`: Evaluation module documentation
- `inference/README.md`: Inference module documentation
- `COLAB_README.md`: Google Colab specific instructions

## Git Workflow

This is a worktree setup:
- Main repository: `G:\Mon Drive\forestgaps-dl`
- Current worktree: `C:\Users\Arthur\.claude-worktrees\forestgaps-dl\affectionate-merkle`
- Worktree branch: `affectionate-merkle`

When making changes:
1. Work in this worktree directory
2. Commit changes to the `affectionate-merkle` branch
3. Changes are tracked in the main repository

## Notes for Claude Code

### When asked to modify code:
- Preserve geospatial metadata handling
- Maintain compatibility with both Colab and local environments
- Follow the existing error handling hierarchy
- Add type hints to new functions
- Update tests when changing functionality

### When asked to add features:
- Check if similar functionality exists in other modules
- Follow the registry pattern for extensible components
- Use configuration system for new parameters
- Consider both segmentation and regression use cases

### When debugging:
- Check environment setup first (Colab vs local)
- Verify configuration loading and validation
- Look for geospatial metadata issues (CRS, transform)
- Check tensor shapes in data pipeline
- Review TensorBoard logs for training issues

### Project-specific terminology:
- **DSM**: Digital Surface Model (elevation data)
- **CHM**: Canopy Height Model (vegetation height)
- **Gap/Trouée**: Opening in forest canopy
- **Threshold**: Height threshold for gap detection (e.g., 2m, 5m, 10m)
- **FiLM**: Feature-wise Linear Modulation (conditioning technique)
- **Tile**: Image patch for training/inference
