# ForestGaps - Configuration System

This directory contains all configuration files for ForestGaps training, using the new YAML + Pydantic system.

## 📁 Directory Structure

```
configs/
├── defaults/          # Default configurations (baseline)
│   ├── training.yaml  # Default training config
│   ├── data.yaml      # Default data config
│   └── model.yaml     # Default model config (UNet-FiLM)
│
├── test/              # Test configurations (for CI/CD and validation)
│   ├── minimal.yaml         # Ultra-fast smoke test (2 epochs, 10 tiles)
│   ├── data_minimal.yaml    # Minimal data config
│   ├── model_minimal.yaml   # Minimal model (small UNet)
│   ├── quick.yaml           # Quick test with all features (5 epochs)
│   ├── data_quick.yaml      # Quick data config
│   └── model_quick.yaml     # Quick model (UNet-FiLM)
│
├── production/        # Production configurations
│   ├── default.yaml         # Production training (50 epochs, all features)
│   └── data_default.yaml    # Production data config
│
└── experiments/       # User experiments (gitignored)
    └── ...
```

## 🚀 Quick Start

### Using Default Configs

```python
from forestgaps.config import load_training_config, load_data_config, load_model_config

# Load defaults
training_config = load_training_config()  # Uses configs/defaults/training.yaml
data_config = load_data_config()          # Uses configs/defaults/data.yaml
model_config = load_model_config()        # Uses configs/defaults/model.yaml
```

### Using Test Configs

```python
# Minimal test (ultra-fast)
training_config = load_training_config("configs/test/minimal.yaml")
data_config = load_data_config("configs/test/data_minimal.yaml")
model_config = load_model_config("configs/test/model_minimal.yaml")

# Quick test (with all features)
training_config = load_training_config("configs/test/quick.yaml")
data_config = load_data_config("configs/test/data_quick.yaml")
model_config = load_model_config("configs/test/model_quick.yaml")
```

### Using Production Config

```python
training_config = load_training_config("configs/production/default.yaml")
data_config = load_data_config("configs/production/data_default.yaml")
```

### With Overrides

```python
training_config = load_training_config(
    "configs/production/default.yaml",
    overrides={
        "epochs": 100,
        "batch_size": 32,
        "optimizer": {
            "lr": 0.0005
        }
    }
)
```

## 📋 Configuration Levels

### Minimal (configs/test/minimal.yaml)
**Purpose**: Ultra-fast smoke test for CI/CD

- **Epochs**: 2
- **Data**: 10 train tiles, 5 val tiles
- **Features**: Minimal (BCE loss, no scheduler, no AMP, no callbacks)
- **Use case**: Quick validation that code runs without errors
- **Duration**: < 1 minute

### Quick (configs/test/quick.yaml)
**Purpose**: Fast testing with all features enabled

- **Epochs**: 5
- **Data**: 50 train tiles, 20 val tiles
- **Features**: Full (Combo Loss, OneCycleLR, AMP, callbacks, etc.)
- **Use case**: Validate new features work correctly
- **Duration**: ~ 2-5 minutes

### Production (configs/production/default.yaml)
**Purpose**: Full production training

- **Epochs**: 50
- **Data**: All available data
- **Features**: All best practices enabled
- **Use case**: Real training runs for deployment
- **Duration**: Hours (depends on data size)

## 🎯 Feature Matrix

| Feature | Minimal | Quick | Production |
|---------|---------|-------|------------|
| **Loss** | BCE | Combo | Combo |
| **Scheduler** | None | OneCycleLR | OneCycleLR |
| **AMP** | ❌ | ✅ | ✅ |
| **Gradient Clipping** | ❌ | ✅ | ✅ |
| **Early Stopping** | ❌ | ✅ (patience=3) | ✅ (patience=10) |
| **Checkpointing** | ✅ | ✅ | ✅ |
| **TensorBoard** | ❌ | ✅ | ✅ |
| **Progress Bar** | ✅ | ✅ | ✅ |
| **Augmentation** | ❌ | ✅ | ✅ |

## 📝 Configuration Files

### Training Config (training.yaml)

Main training parameters:
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `optimizer`: Optimizer configuration (type, lr, weight_decay, etc.)
- `scheduler`: LR scheduler configuration
- `loss`: Loss function configuration (type, weights for Combo Loss)
- `callbacks`: Callback configuration (early stopping, checkpointing, etc.)
- `optimization`: Training optimizations (AMP, gradient clipping, etc.)

### Data Config (data.yaml)

Data pipeline parameters:
- `preprocessing`: Tile generation, normalization
- `augmentation`: Data augmentation settings
- `dataset`: Split ratios, thresholds
- `num_workers`: DataLoader workers

### Model Config (model.yaml)

Model architecture parameters:
- `model_type`: Architecture (unet, unet_film, deeplabv3_plus, etc.)
- `in_channels`, `out_channels`: Input/output channels
- `base_channels`: Base number of feature channels
- `depth`: Network depth
- `num_conditions`: FiLM conditions (for unet_film)

## 🧪 Testing Your Config

Test any configuration with the test script:

```bash
# Test minimal config
python scripts/test_complete_workflow.py --config minimal

# Test quick config
python scripts/test_complete_workflow.py --config quick

# Test production config
python scripts/test_complete_workflow.py --config production
```

## 🔧 Creating Custom Configs

1. **Copy a base config**:
   ```bash
   cp configs/production/default.yaml configs/experiments/my_experiment.yaml
   ```

2. **Modify parameters** as needed

3. **Load your config**:
   ```python
   config = load_training_config("configs/experiments/my_experiment.yaml")
   ```

## 📚 Documentation

For detailed documentation on the configuration system, see:
- [IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Complete implementation guide
- [examples/training_with_new_features.py](../examples/training_with_new_features.py) - Usage examples

## ✅ Validation

All configs are validated using Pydantic schemas:
- Type checking
- Value range validation
- Required fields verification
- Automatic error messages

Example validation error:
```
ValidationError: 1 validation error for TrainingConfig
epochs
  ensure this value is greater than or equal to 1
```

## 🔥 Best Practices

1. **Start with test configs** to validate your setup
2. **Use minimal for CI/CD** (fast feedback)
3. **Use quick for feature testing** (comprehensive but fast)
4. **Use production for final training** (all best practices)
5. **Override carefully** (prefer new config files over overrides)
6. **Version control your experiments** (in configs/experiments/)

## 🎯 Compliance

All configurations follow the recommendations from:
- **Document 1**: Combo Loss (PRIORITÉ MAX)
- **Document 2**: OneCycleLR, AMP, gradient clipping, callbacks
- **Document 3**: Early stopping (patience=10), per-tile normalization, 70/15/15 splits

---

**Status**: ✅ Fully implemented
**Version**: Phase 1 & 2 Complete
**Last Update**: December 4, 2024
