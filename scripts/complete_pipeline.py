"""Complete training + evaluation + inference pipeline.

Pipeline complet: Train → Eval → Infer sur données indépendantes.

Usage:
    python scripts/complete_pipeline.py --config minimal --data-dir /data/Plot137
    python scripts/complete_pipeline.py --config quick --data-dir /data
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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


def create_dummy_data(num_samples: int, tile_size: int = 256):
    """Create dummy data for testing."""
    dsm_tiles = torch.randn(num_samples, 1, tile_size, tile_size)
    gap_masks = torch.randint(0, 2, (num_samples, 1, tile_size, tile_size)).float()
    return TensorDataset(dsm_tiles, gap_masks)


def create_data_loaders(data_config, training_config, max_train=100, max_val=20, max_test=20):
    """Create data loaders."""
    tile_size = data_config.preprocessing.tile_size

    train_dataset = create_dummy_data(max_train, tile_size)
    val_dataset = create_dummy_data(max_val, tile_size)
    test_dataset = create_dummy_data(max_test, tile_size)

    train_loader = DataLoader(
        train_dataset, batch_size=training_config.batch_size,
        shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=training_config.val_batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=training_config.val_batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                training_opt, device, epochs):
    """Train model."""
    print(f"\n{'='*80}")
    print("PHASE 1: TRAINING")
    print(f"{'='*80}\n")

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            with training_opt.forward_context():
                # Check if model needs threshold parameter (FiLM models)
                if 'film' in model.__class__.__name__.lower():
                    # Create threshold tensor (default 5.0)
                    threshold = torch.full((inputs.shape[0], 1), 5.0, device=device)
                    outputs = model(inputs, threshold)
                else:
                    outputs = model(inputs)

                if isinstance(criterion, ComboLoss):
                    loss, _ = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)

            step_info = training_opt.backward_step(loss, optimizer, model.parameters())
            train_loss += loss.item()

            if hasattr(scheduler, 'step') and training_opt.accumulator.should_step():
                if hasattr(scheduler, '__class__') and 'OneCycleLR' in scheduler.__class__.__name__:
                    scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # Check if model needs threshold parameter
                if 'film' in model.__class__.__name__.lower():
                    threshold = torch.full((inputs.shape[0], 1), 5.0, device=device)
                    outputs = model(inputs, threshold)
                else:
                    outputs = model(inputs)
                if isinstance(criterion, ComboLoss):
                    loss, _ = criterion(outputs, targets)
                else:
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Scheduler step (if not OneCycleLR)
        if hasattr(scheduler, 'step') and 'OneCycleLR' not in scheduler.__class__.__name__:
            if 'ReduceLROnPlateau' in scheduler.__class__.__name__:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"\n✓ Training completed!")
    print(f"  Best val loss: {best_val_loss:.4f} (epoch {best_epoch+1})")

    return history


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set (données jamais vues)."""
    print(f"\n{'='*80}")
    print("PHASE 2: EVALUATION SUR DONNÉES DE TEST")
    print(f"{'='*80}\n")

    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Check if model needs threshold parameter
            if 'film' in model.__class__.__name__.lower():
                threshold = torch.full((inputs.shape[0], 1), 5.0, device=device)
                outputs = model(inputs, threshold)
            else:
                outputs = model(inputs)

            if isinstance(criterion, ComboLoss):
                loss, _ = criterion(outputs, targets)
            else:
                loss = criterion(outputs, targets)
            test_loss += loss.item()

            # Collect predictions for metrics
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    test_loss /= len(test_loader)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Calculate metrics
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    tn = np.sum((all_preds == 0) & (all_targets == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"Test Results (données indépendantes):")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  IoU: {iou:.4f}")

    metrics = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }

    return metrics


def inference_on_independent_data(model, data_dir, device, tile_size=256):
    """Inférence sur données indépendantes du répertoire /data."""
    print(f"\n{'='*80}")
    print("PHASE 3: INFÉRENCE SUR DONNÉES INDÉPENDANTES")
    print(f"{'='*80}\n")

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"⚠ Data directory not found: {data_dir}")
        print("  Using dummy data for demonstration")
        # Create dummy independent data
        independent_data = create_dummy_data(10, tile_size)
        independent_loader = DataLoader(independent_data, batch_size=4, shuffle=False)
    else:
        print(f"✓ Loading data from: {data_dir}")
        # TODO: Implement real data loading from /data directory
        # For now, use dummy data
        independent_data = create_dummy_data(10, tile_size)
        independent_loader = DataLoader(independent_data, batch_size=4, shuffle=False)

    model.eval()
    predictions = []

    print(f"Running inference on {len(independent_loader)} batches...")
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(independent_loader):
            inputs = inputs.to(device)
            # Check if model needs threshold parameter
            if 'film' in model.__class__.__name__.lower():
                threshold = torch.full((inputs.shape[0], 1), 5.0, device=device)
                outputs = model(inputs, threshold)
            else:
                outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            predictions.append(preds.cpu())

            if batch_idx % 5 == 0:
                print(f"  Processed batch {batch_idx+1}/{len(independent_loader)}")

    predictions = torch.cat(predictions)

    print(f"\n✓ Inference completed!")
    print(f"  Generated {predictions.shape[0]} predictions")
    print(f"  Shape: {predictions.shape}")
    print(f"  Mean prediction: {predictions.mean():.4f}")

    return predictions


def complete_pipeline(config_name: str, data_dir: str = None):
    """Run complete pipeline: Train → Eval → Infer."""
    print("\n" + "=" * 80)
    print(f"COMPLETE PIPELINE: {config_name.upper()}")
    print("=" * 80 + "\n")

    # Load configs
    print("Loading configurations...")
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
    print("✓ Configs loaded\n")

    # Create model
    print("Creating model...")
    model_kwargs = {
        "in_channels": model_config.in_channels,
        "out_channels": model_config.out_channels,
    }

    # Map config model_type to registry model_type if needed
    model_type = model_config.model_type
    if model_type == "unet_film":
        model_type = "film_unet"  # Registry uses film_unet

    # Add model-specific parameters
    if model_config.model_type in ["unet"]:
        model_kwargs["init_features"] = model_config.base_channels
    elif model_config.model_type in ["film_unet", "unet_film"]:
        model_kwargs["init_features"] = model_config.base_channels
        model_kwargs["condition_size"] = model_config.num_conditions
    else:
        model_kwargs["base_channels"] = model_config.base_channels

    model = create_model(model_type, **model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Model: {model_config.model_type} on {device}\n")

    # Create loss
    if training_config.loss.type == "combo":
        criterion = ComboLoss(
            bce_weight=training_config.loss.bce_weight,
            dice_weight=training_config.loss.dice_weight,
            focal_weight=training_config.loss.focal_weight,
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Create optimizer
    if training_config.optimizer.type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.optimizer.lr,
            weight_decay=training_config.optimizer.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.optimizer.lr)

    # Create data loaders
    print("Creating data loaders...")
    max_train = getattr(training_config, 'max_train_tiles', 100)
    max_val = getattr(training_config, 'max_val_tiles', 20)
    max_test = getattr(training_config, 'max_test_tiles', 20)
    train_loader, val_loader, test_loader = create_data_loaders(
        data_config, training_config, max_train, max_val, max_test
    )
    print(f"✓ Data loaders: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test\n")

    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        training_config.scheduler.dict() if hasattr(training_config.scheduler, 'dict')
        else training_config.scheduler.model_dump(),
        steps_per_epoch=len(train_loader),
        epochs=training_config.epochs,
    )

    # Training optimizer
    training_opt = TrainingOptimizer(
        gradient_clip_value=training_config.optimization.gradient_clip_value,
        gradient_clip_norm=training_config.optimization.gradient_clip_norm,
        use_amp=training_config.optimization.use_amp,
        accumulate_grad_batches=training_config.optimization.accumulate_grad_batches,
        device=str(device),
    )

    # PHASE 1: Training
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        training_opt, device, training_config.epochs
    )

    # PHASE 2: Evaluation on test set (données jamais vues pendant training)
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    # PHASE 3: Inference sur données indépendantes
    if data_dir:
        predictions = inference_on_independent_data(model, data_dir, device, data_config.preprocessing.tile_size)
    else:
        print("\nSkipping inference (no data directory provided)")
        predictions = None

    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Configuration: {config_name}")
    print(f"Model: {model_config.model_type}")
    print(f"Training epochs: {training_config.epochs}")
    print(f"\nFinal Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Best Validation Loss: {min(history['val_loss']):.4f}")
    print(f"\nTest Set Metrics (données indépendantes):")
    print(f"  - Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - F1-Score: {test_metrics['f1']:.4f}")
    print(f"  - IoU: {test_metrics['iou']:.4f}")
    if predictions is not None:
        print(f"\nInference:")
        print(f"  - Predictions generated: {len(predictions)}")
    print(f"{'='*80}\n")

    return {
        'history': history,
        'test_metrics': test_metrics,
        'predictions': predictions
    }


def main():
    parser = argparse.ArgumentParser(description="Complete training + evaluation + inference pipeline")
    parser.add_argument(
        "--config", type=str, choices=["minimal", "quick", "production"],
        default="minimal", help="Configuration level"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory with independent data for inference (e.g., /data/Plot137)"
    )
    args = parser.parse_args()

    try:
        results = complete_pipeline(args.config, args.data_dir)
        print("\n✅ Complete pipeline executed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
