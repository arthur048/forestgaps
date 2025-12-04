"""Example: Training with new features.

This example demonstrates how to use all the new features implemented
in Phase 1 and Phase 2:
- Configuration system (YAML + Pydantic)
- Combo Loss (BCE + Dice + Focal)
- LR Scheduling (OneCycleLR, CosineAnnealing, etc.)
- Callback system (Early stopping, checkpointing, TensorBoard)
- Optimizations (AMP, gradient clipping, gradient accumulation)

Conforme Documents 1, 2, 3: best practices pour l'entraînement.
"""

import torch
import torch.nn as nn
from pathlib import Path

# Configuration imports
from forestgaps.config import (
    load_training_config,
    load_data_config,
    load_model_config,
    load_complete_config,
)

# Loss imports
from forestgaps.training.losses import ComboLoss, DiceLoss, FocalLoss

# Scheduler imports
from forestgaps.training.optimization import create_scheduler

# Callback imports
from forestgaps.training.callbacks import (
    CallbackList,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LRSchedulerCallback,
    TensorBoardCallback,
    ProgressBarCallback,
)

# Optimization imports
from forestgaps.training.optimization import (
    TrainingOptimizer,
    enable_gradient_checkpointing,
    compile_model,
)

# Model imports (example with UNet)
from forestgaps.models import create_model


def example_1_basic_config_loading():
    """Example 1: Load default configurations."""
    print("=" * 60)
    print("Example 1: Loading Default Configurations")
    print("=" * 60)

    # Load default training config
    training_config = load_training_config()
    print(f"✓ Training config loaded")
    print(f"  - Epochs: {training_config.epochs}")
    print(f"  - Batch size: {training_config.batch_size}")
    print(f"  - Optimizer: {training_config.optimizer.type}")
    print(f"  - Scheduler: {training_config.scheduler.type}")
    print(f"  - Loss: {training_config.loss.type}")
    print(f"  - AMP enabled: {training_config.optimization.use_amp}")
    print()

    # Load default data config
    data_config = load_data_config()
    print(f"✓ Data config loaded")
    print(f"  - Tile size: {data_config.preprocessing.tile_size}")
    print(f"  - Normalization: {data_config.preprocessing.normalize_method}")
    print(f"  - Augmentation: {data_config.augmentation.enabled}")
    print()

    # Load default model config
    model_config = load_model_config()
    print(f"✓ Model config loaded")
    print(f"  - Model type: {model_config.model_type}")
    print(f"  - Task: {model_config.task}")
    print(f"  - Base channels: {model_config.base_channels}")
    print()


def example_2_config_with_overrides():
    """Example 2: Load config with overrides."""
    print("=" * 60)
    print("Example 2: Configuration with Overrides")
    print("=" * 60)

    # Load with overrides
    training_config = load_training_config(
        overrides={
            "epochs": 100,
            "batch_size": 32,
            "optimizer": {
                "type": "adamw",
                "lr": 0.001,
            },
            "scheduler": {
                "type": "cosine",
            },
            "loss": {
                "type": "combo",
                "bce_weight": 0.4,
                "dice_weight": 0.4,
                "focal_weight": 0.2,
            },
        }
    )

    print(f"✓ Training config loaded with overrides")
    print(f"  - Epochs: {training_config.epochs}")
    print(f"  - Batch size: {training_config.batch_size}")
    print(f"  - LR: {training_config.optimizer.lr}")
    print(f"  - Scheduler: {training_config.scheduler.type}")
    print(f"  - Loss weights: BCE={training_config.loss.bce_weight}, "
          f"Dice={training_config.loss.dice_weight}, "
          f"Focal={training_config.loss.focal_weight}")
    print()


def example_3_combo_loss():
    """Example 3: Using Combo Loss."""
    print("=" * 60)
    print("Example 3: Combo Loss (BCE + Dice + Focal)")
    print("=" * 60)

    # Create Combo Loss
    loss_fn = ComboLoss(
        bce_weight=0.5,
        dice_weight=0.3,
        focal_weight=0.2,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )

    # Example forward pass
    batch_size, height, width = 4, 256, 256
    pred = torch.randn(batch_size, 1, height, width)
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()

    total_loss, loss_breakdown = loss_fn(pred, target)

    print(f"✓ Combo Loss computed")
    print(f"  - Total loss: {total_loss.item():.4f}")
    print(f"  - BCE loss: {loss_breakdown['bce_loss']:.4f}")
    print(f"  - Dice loss: {loss_breakdown['dice_loss']:.4f}")
    print(f"  - Focal loss: {loss_breakdown['focal_loss']:.4f}")
    print()


def example_4_lr_scheduler():
    """Example 4: Using LR Schedulers."""
    print("=" * 60)
    print("Example 4: LR Scheduling")
    print("=" * 60)

    # Create a dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Create OneCycleLR scheduler
    scheduler_config = {
        "type": "onecycle",
        "max_lr": 0.01,
        "pct_start": 0.3,
    }

    scheduler = create_scheduler(
        optimizer,
        scheduler_config,
        steps_per_epoch=100,
        epochs=50
    )

    print(f"✓ OneCycleLR scheduler created")
    print(f"  - Type: {scheduler_config['type']}")
    print(f"  - Max LR: {scheduler_config['max_lr']}")
    print(f"  - Initial LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Simulate a few steps
    for step in range(10):
        scheduler.step()

    print(f"  - LR after 10 steps: {optimizer.param_groups[0]['lr']:.6f}")
    print()


def example_5_callbacks():
    """Example 5: Using Callbacks."""
    print("=" * 60)
    print("Example 5: Callback System")
    print("=" * 60)

    # Create callbacks
    callbacks = CallbackList([
        EarlyStoppingCallback(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
        ModelCheckpointCallback(
            save_dir="checkpoints",
            monitor="val_loss",
            mode="min",
        ),
        TensorBoardCallback(
            log_dir="runs",
            comment="example_run",
        ),
        ProgressBarCallback(total_epochs=50),
    ])

    print(f"✓ Callback system created")
    print(f"  - Number of callbacks: {len(callbacks.callbacks)}")
    print(f"  - Early stopping patience: 10 epochs")
    print(f"  - Model checkpointing enabled")
    print(f"  - TensorBoard logging enabled")
    print(f"  - Progress bar enabled")
    print()


def example_6_training_optimizer():
    """Example 6: Using Training Optimizer (AMP + gradient clipping)."""
    print("=" * 60)
    print("Example 6: Training Optimizations")
    print("=" * 60)

    # Create training optimizer
    training_opt = TrainingOptimizer(
        gradient_clip_value=None,
        gradient_clip_norm=1.0,
        use_amp=True,
        accumulate_grad_batches=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"✓ Training optimizer created")
    print(f"  - AMP enabled: {training_opt.amp_manager.enabled}")
    print(f"  - Gradient clipping (norm): 1.0")
    print(f"  - Gradient accumulation: 4 batches")
    print()

    # Simulated training step
    print("Simulated training step:")
    model = nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    # Forward pass with AMP
    with training_opt.forward_context():
        inputs = torch.randn(4, 10)
        outputs = model(inputs)
        loss = outputs.mean()

    # Backward pass with all optimizations
    step_info = training_opt.backward_step(loss, optimizer, model.parameters())

    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Optimizer stepped: {step_info.get('stepped', False)}")
    if 'grad_norm' in step_info:
        print(f"  - Gradient norm: {step_info['grad_norm']:.4f}")
    print()


def example_7_complete_training_setup():
    """Example 7: Complete training setup."""
    print("=" * 60)
    print("Example 7: Complete Training Setup")
    print("=" * 60)

    # Load complete configuration
    config = load_complete_config()

    training_cfg = config["training"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    # Create model
    print("Setting up model...")
    model = create_model(
        model_cfg.model_type,
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        base_channels=model_cfg.base_channels,
        depth=model_cfg.depth,
    )
    print(f"✓ Model created: {model_cfg.model_type}")

    # Enable gradient checkpointing if configured
    if training_cfg.optimization.use_gradient_checkpointing:
        model = enable_gradient_checkpointing(model)
        print(f"✓ Gradient checkpointing enabled")

    # Compile model if configured (PyTorch 2.0+)
    if training_cfg.optimization.use_torch_compile:
        model = compile_model(model, mode=training_cfg.optimization.compile_mode)
        print(f"✓ Model compiled with torch.compile()")

    # Create loss function
    print("Setting up loss function...")
    if training_cfg.loss.type == "combo":
        criterion = ComboLoss(
            bce_weight=training_cfg.loss.bce_weight,
            dice_weight=training_cfg.loss.dice_weight,
            focal_weight=training_cfg.loss.focal_weight,
            focal_alpha=training_cfg.loss.focal_alpha,
            focal_gamma=training_cfg.loss.focal_gamma,
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    print(f"✓ Loss function: {training_cfg.loss.type}")

    # Create optimizer
    print("Setting up optimizer...")
    if training_cfg.optimizer.type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_cfg.optimizer.lr,
            weight_decay=training_cfg.optimizer.weight_decay,
            betas=training_cfg.optimizer.betas,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg.optimizer.lr)
    print(f"✓ Optimizer: {training_cfg.optimizer.type} (lr={training_cfg.optimizer.lr})")

    # Create scheduler
    print("Setting up scheduler...")
    scheduler = create_scheduler(
        optimizer,
        training_cfg.scheduler.dict(),
        steps_per_epoch=100,  # Would come from DataLoader
        epochs=training_cfg.epochs,
    )
    print(f"✓ Scheduler: {training_cfg.scheduler.type}")

    # Create training optimizer (AMP + gradient clipping)
    print("Setting up training optimizations...")
    training_opt = TrainingOptimizer(
        gradient_clip_value=training_cfg.optimization.gradient_clip_value,
        gradient_clip_norm=training_cfg.optimization.gradient_clip_norm,
        use_amp=training_cfg.optimization.use_amp,
        accumulate_grad_batches=training_cfg.optimization.accumulate_grad_batches,
        device=training_cfg.device,
    )
    print(f"✓ AMP: {training_cfg.optimization.use_amp}")
    print(f"✓ Gradient clipping: {training_cfg.optimization.gradient_clip_norm}")

    # Create callbacks
    print("Setting up callbacks...")
    callbacks = CallbackList([
        EarlyStoppingCallback(
            monitor=training_cfg.callbacks.early_stopping_monitor,
            patience=training_cfg.callbacks.early_stopping_patience,
            mode=training_cfg.callbacks.early_stopping_mode,
        ) if training_cfg.callbacks.early_stopping else None,
        ModelCheckpointCallback(
            save_dir=training_cfg.checkpoint_dir,
            monitor=training_cfg.callbacks.checkpoint_monitor,
            mode=training_cfg.callbacks.checkpoint_mode,
        ),
        TensorBoardCallback(
            log_dir=training_cfg.callbacks.tensorboard_log_dir,
            comment=training_cfg.callbacks.tensorboard_comment,
        ) if training_cfg.callbacks.tensorboard_enabled else None,
        ProgressBarCallback(total_epochs=training_cfg.epochs) if training_cfg.callbacks.progress_bar else None,
    ])
    # Remove None callbacks
    callbacks.callbacks = [cb for cb in callbacks.callbacks if cb is not None]
    print(f"✓ Callbacks: {len(callbacks.callbacks)} active")

    print()
    print("=" * 60)
    print("Setup complete! Ready for training.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ForestGaps: New Features Examples")
    print("Phase 1 & 2 Implementation")
    print("=" * 60 + "\n")

    # Run examples
    example_1_basic_config_loading()
    example_2_config_with_overrides()
    example_3_combo_loss()
    example_4_lr_scheduler()
    example_5_callbacks()
    example_6_training_optimizer()
    example_7_complete_training_setup()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
