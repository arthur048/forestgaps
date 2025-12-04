"""Complete workflow test with new config system.

Tests the entire training pipeline with new features:
- Config loading (YAML + Pydantic)
- Combo Loss
- LR Scheduling
- Callbacks
- AMP + Gradient Clipping

Usage:
    python scripts/test_complete_workflow.py --config minimal  # Ultra-fast smoke test
    python scripts/test_complete_workflow.py --config quick    # Quick test with features
    python scripts/test_complete_workflow.py --config production  # Full production test
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from forestgaps.config import (
    load_training_config,
    load_data_config,
    load_model_config,
)
from forestgaps.models import create_model
from forestgaps.training.losses import ComboLoss
from forestgaps.training.optimization import create_scheduler, TrainingOptimizer
from forestgaps.training.callbacks import (
    CallbackList,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    TensorBoardCallback,
    ProgressBarCallback,
)


def create_dummy_data(num_samples: int, tile_size: int = 256):
    """Create dummy data for testing."""
    print(f"Creating dummy dataset: {num_samples} samples of {tile_size}x{tile_size}")

    # Create random DSM tiles and gap masks
    dsm_tiles = torch.randn(num_samples, 1, tile_size, tile_size)
    gap_masks = torch.randint(0, 2, (num_samples, 1, tile_size, tile_size)).float()

    return TensorDataset(dsm_tiles, gap_masks)


def create_data_loaders(data_config, training_config, max_train=100, max_val=20):
    """Create data loaders with dummy data."""
    tile_size = data_config.preprocessing.tile_size

    # Create datasets
    train_dataset = create_dummy_data(max_train, tile_size)
    val_dataset = create_dummy_data(max_val, tile_size)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.val_batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    return train_loader, val_loader


class DummyTrainer:
    """Minimal trainer for callback testing."""
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.stop_training = False


def test_workflow(config_name: str):
    """Test complete workflow with specified config."""
    print("\n" + "=" * 80)
    print(f"Testing Complete Workflow: {config_name.upper()} configuration")
    print("=" * 80 + "\n")

    # 1. Load configurations
    print("Step 1: Loading configurations...")
    training_config = load_training_config(
        config_path=f"configs/test/{config_name}.yaml" if config_name in ["minimal", "quick"]
                    else "configs/production/default.yaml"
    )
    data_config = load_data_config(
        config_path=f"configs/test/data_{config_name}.yaml" if config_name in ["minimal", "quick"]
                    else "configs/production/data_default.yaml"
    )
    model_config = load_model_config(
        config_path=f"configs/test/model_{config_name}.yaml" if config_name in ["minimal", "quick"]
                    else "configs/defaults/model.yaml"
    )
    print(f"✓ Configs loaded successfully")
    print(f"  - Training: {training_config.epochs} epochs, batch {training_config.batch_size}")
    print(f"  - Data: {data_config.preprocessing.tile_size}x{data_config.preprocessing.tile_size} tiles")
    print(f"  - Model: {model_config.model_type}, {model_config.base_channels} channels")
    print()

    # 2. Create model
    print("Step 2: Creating model...")
    # Use appropriate kwargs based on model type
    model_kwargs = {
        "in_channels": model_config.in_channels,
        "out_channels": model_config.out_channels,
    }

    # Add model-specific params
    if model_config.model_type == "unet":
        model_kwargs["init_features"] = model_config.base_channels
    elif model_config.model_type == "unet_film":
        model_kwargs["init_features"] = model_config.base_channels
        model_kwargs["num_conditions"] = model_config.num_conditions
    else:
        # Other models might use base_channels
        model_kwargs["base_channels"] = model_config.base_channels

    model = create_model(model_config.model_type, **model_kwargs)
    device = torch.device(
        training_config.device if training_config.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)
    print(f"✓ Model created: {model_config.model_type}")
    print(f"  - Device: {device}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # 3. Create loss function
    print("Step 3: Creating loss function...")
    if training_config.loss.type == "combo":
        criterion = ComboLoss(
            bce_weight=training_config.loss.bce_weight,
            dice_weight=training_config.loss.dice_weight,
            focal_weight=training_config.loss.focal_weight,
            focal_alpha=training_config.loss.focal_alpha,
            focal_gamma=training_config.loss.focal_gamma,
        )
        print(f"✓ Combo Loss created (BCE={training_config.loss.bce_weight}, "
              f"Dice={training_config.loss.dice_weight}, "
              f"Focal={training_config.loss.focal_weight})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"✓ BCE Loss created")
    print()

    # 4. Create optimizer
    print("Step 4: Creating optimizer...")
    if training_config.optimizer.type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.optimizer.lr,
            weight_decay=training_config.optimizer.weight_decay,
            betas=training_config.optimizer.betas,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config.optimizer.lr,
        )
    print(f"✓ Optimizer: {training_config.optimizer.type} (lr={training_config.optimizer.lr})")
    print()

    # 5. Create data loaders
    print("Step 5: Creating data loaders...")
    max_train = getattr(training_config, 'max_train_tiles', 100)
    max_val = getattr(training_config, 'max_val_tiles', 20)
    train_loader, val_loader = create_data_loaders(
        data_config, training_config, max_train, max_val
    )
    print()

    # 6. Create scheduler
    print("Step 6: Creating LR scheduler...")
    scheduler = create_scheduler(
        optimizer,
        training_config.scheduler.dict(),
        steps_per_epoch=len(train_loader),
        epochs=training_config.epochs,
    )
    print(f"✓ Scheduler: {training_config.scheduler.type}")
    print()

    # 7. Create training optimizer (AMP + gradient clipping)
    print("Step 7: Setting up training optimizations...")
    training_opt = TrainingOptimizer(
        gradient_clip_value=training_config.optimization.gradient_clip_value,
        gradient_clip_norm=training_config.optimization.gradient_clip_norm,
        use_amp=training_config.optimization.use_amp,
        accumulate_grad_batches=training_config.optimization.accumulate_grad_batches,
        device=str(device),
    )
    print(f"✓ Training optimizer configured")
    print(f"  - AMP: {training_config.optimization.use_amp}")
    print(f"  - Gradient clipping: {training_config.optimization.gradient_clip_norm}")
    print(f"  - Gradient accumulation: {training_config.optimization.accumulate_grad_batches}")
    print()

    # 8. Create callbacks
    print("Step 8: Setting up callbacks...")
    callbacks_list = []

    if training_config.callbacks.early_stopping:
        callbacks_list.append(
            EarlyStoppingCallback(
                monitor=training_config.callbacks.early_stopping_monitor,
                patience=training_config.callbacks.early_stopping_patience,
                mode=training_config.callbacks.early_stopping_mode,
            )
        )

    callbacks_list.append(
        ModelCheckpointCallback(
            save_dir=training_config.checkpoint_dir,
            save_best_only=training_config.callbacks.checkpoint_save_best_only,
            monitor=training_config.callbacks.checkpoint_monitor,
            mode=training_config.callbacks.checkpoint_mode,
            save_frequency=training_config.callbacks.checkpoint_save_frequency,
        )
    )

    if training_config.callbacks.tensorboard_enabled:
        callbacks_list.append(
            TensorBoardCallback(
                log_dir=training_config.callbacks.tensorboard_log_dir,
                comment=training_config.callbacks.tensorboard_comment,
            )
        )

    if training_config.callbacks.progress_bar:
        callbacks_list.append(
            ProgressBarCallback(total_epochs=training_config.epochs)
        )

    callbacks = CallbackList(callbacks_list)
    print(f"✓ Callbacks configured: {len(callbacks_list)} active")
    for cb in callbacks_list:
        print(f"  - {cb.__class__.__name__}")
    print()

    # 9. Training loop
    print("Step 9: Running training loop...")
    print("-" * 80)

    trainer = DummyTrainer(model, optimizer)
    callbacks.on_train_begin(trainer)

    for epoch in range(training_config.epochs):
        if trainer.stop_training:
            print(f"\nTraining stopped early at epoch {epoch}")
            break

        callbacks.on_epoch_begin(epoch, trainer)

        # Training
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            callbacks.on_batch_begin(batch_idx, trainer)

            # Forward pass with AMP
            with training_opt.forward_context():
                outputs = model(inputs)
                if training_config.loss.type == "combo":
                    loss, loss_breakdown = criterion(outputs, targets)
                    epoch_metrics.update(loss_breakdown)
                else:
                    loss = criterion(outputs, targets)

            # Backward pass with optimizations
            step_info = training_opt.backward_step(
                loss, optimizer, model.parameters()
            )

            epoch_loss += loss.item()

            # Scheduler step (if per-batch)
            if hasattr(scheduler, 'step') and training_config.scheduler.type == "onecycle":
                scheduler.step()

            batch_logs = {"loss": loss.item()}
            if "grad_norm" in step_info:
                batch_logs["grad_norm"] = step_info["grad_norm"]
            callbacks.on_batch_end(batch_idx, trainer, logs=batch_logs)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                if training_config.loss.type == "combo":
                    loss, _ = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Scheduler step (if per-epoch)
        if hasattr(scheduler, 'step') and training_config.scheduler.type != "onecycle":
            if training_config.scheduler.type == "reduce_on_plateau":
                scheduler.step(val_loss / len(val_loader))
            else:
                scheduler.step()

        # Log epoch metrics
        epoch_logs = {
            "loss": epoch_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "lr": optimizer.param_groups[0]["lr"],
        }
        epoch_logs.update(epoch_metrics)

        callbacks.on_epoch_end(epoch, trainer, logs=epoch_logs)

    callbacks.on_train_end(trainer)

    print("-" * 80)
    print(f"\n✓ Training completed successfully!")
    print()

    # 10. Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Configuration: {config_name}")
    print(f"Status: ✅ PASSED")
    print(f"Epochs completed: {epoch + 1}/{training_config.epochs}")
    print(f"Final training loss: {epoch_logs['loss']:.4f}")
    print(f"Final validation loss: {epoch_logs['val_loss']:.4f}")
    print(f"All features tested:")
    print(f"  ✓ Config system (YAML + Pydantic)")
    print(f"  ✓ Model creation ({model_config.model_type})")
    print(f"  ✓ Loss function ({training_config.loss.type})")
    print(f"  ✓ Optimizer ({training_config.optimizer.type})")
    print(f"  ✓ LR Scheduler ({training_config.scheduler.type})")
    print(f"  ✓ Training optimizations (AMP, gradient clipping)")
    print(f"  ✓ Callbacks ({len(callbacks_list)} active)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test complete workflow")
    parser.add_argument(
        "--config",
        type=str,
        choices=["minimal", "quick", "production"],
        default="minimal",
        help="Configuration to test (minimal=ultra-fast, quick=fast+features, production=full)"
    )
    args = parser.parse_args()

    try:
        test_workflow(args.config)
        print("\n✅ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
