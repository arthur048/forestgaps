"""
Benchmark stable et reproductible pour ForestGaps.

Features:
- Fixed random seeds pour reproductibilité
- Validation statistique (mean ± std sur N runs)
- Sauvegarde résultats JSON + CSV
- Comparaison multi-modèles robuste
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forestgaps.config import load_training_config
from forestgaps.data.loaders import create_data_loaders
from forestgaps.models import create_model
from forestgaps.training import Trainer
from forestgaps.training.losses import ComboLoss


def set_seed(seed: int = 42):
    """Fix all random seeds for reproductibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_benchmark(
    model_name: str,
    config_path: str,
    data_dir: str,
    seed: int,
    epochs: int = 5
) -> Dict:
    """Run single benchmark with fixed seed."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Seed: {seed} | Epochs: {epochs}")
    print(f"{'='*60}")

    # Set seed
    set_seed(seed)

    # Load config
    config = load_training_config(config_path)
    config.epochs = epochs

    # Create data loaders
    data_loaders = create_data_loaders(config, data_dir=data_dir)

    # Create model
    model_type = model_name
    if model_type == "unet_film":
        model_type = "film_unet"

    model_kwargs = {"in_channels": 1, "out_channels": 1}

    if "film" in model_type:
        model_kwargs["init_features"] = config.model.base_channels
        model_kwargs["condition_size"] = config.model.num_conditions
    elif model_type == "deeplabv3_plus":
        model_kwargs["encoder_name"] = "resnet18"
        model_kwargs["encoder_weights"] = None
    else:
        model_kwargs["init_features"] = config.model.base_channels

    model = create_model(model_type, **model_kwargs)

    # Create trainer
    criterion = ComboLoss(bce_weight=1.0, dice_weight=1.0, focal_weight=1.0)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=torch.optim.Adam(model.parameters(), lr=config.learning_rate),
        scheduler=None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_amp=False,  # Disable for stability
        gradient_clip_val=1.0
    )

    # Train
    history = trainer.train(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        epochs=epochs,
        save_best=False  # Don't save during benchmarking
    )

    # Extract metrics
    results = {
        "model": model_name,
        "seed": seed,
        "epochs": epochs,
        "final_train_loss": float(history['train_loss'][-1]),
        "final_val_loss": float(history['val_loss'][-1]),
        "best_val_loss": float(min(history['val_loss'])),
        "best_epoch": int(np.argmin(history['val_loss']) + 1),
        "train_loss_history": [float(x) for x in history['train_loss']],
        "val_loss_history": [float(x) for x in history['val_loss']]
    }

    print(f"\n✓ Results:")
    print(f"  - Best val loss: {results['best_val_loss']:.4f} @ epoch {results['best_epoch']}")
    print(f"  - Final train loss: {results['final_train_loss']:.4f}")
    print(f"  - Final val loss: {results['final_val_loss']:.4f}")

    return results


def run_multi_seed_benchmark(
    models: List[str],
    config_path: str,
    data_dir: str,
    n_seeds: int = 3,
    base_seed: int = 42,
    epochs: int = 5,
    output_dir: str = "./benchmark_results"
) -> pd.DataFrame:
    """Run benchmark across multiple models and seeds."""
    all_results = []

    for model_name in models:
        for i in range(n_seeds):
            seed = base_seed + i

            try:
                result = run_single_benchmark(
                    model_name=model_name,
                    config_path=config_path,
                    data_dir=data_dir,
                    seed=seed,
                    epochs=epochs
                )
                all_results.append(result)

            except Exception as e:
                print(f"\n✗ Failed: {model_name} (seed {seed}): {e}")
                import traceback
                traceback.print_exc()

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON (full history)
    json_path = output_path / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved results to {json_path}")

    # Save CSV (summary)
    csv_path = output_path / "benchmark_summary.csv"
    summary_df = df[['model', 'seed', 'best_val_loss', 'best_epoch', 'final_train_loss', 'final_val_loss']]
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary to {csv_path}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY (mean ± std across seeds)")
    print(f"{'='*80}")

    for model_name in models:
        model_results = df[df['model'] == model_name]

        if len(model_results) > 0:
            mean_best = model_results['best_val_loss'].mean()
            std_best = model_results['best_val_loss'].std()

            print(f"\n{model_name}:")
            print(f"  Best val loss: {mean_best:.4f} ± {std_best:.4f}")
            print(f"  N runs: {len(model_results)}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Stable benchmark script")
    parser.add_argument("--models", nargs="+", default=["unet", "film_unet", "deeplabv3_plus"],
                      help="Models to benchmark")
    parser.add_argument("--config", default="configs/test/quick.yaml",
                      help="Config file path")
    parser.add_argument("--data-dir", required=True,
                      help="Data directory")
    parser.add_argument("--n-seeds", type=int, default=3,
                      help="Number of random seeds")
    parser.add_argument("--base-seed", type=int, default=42,
                      help="Base random seed")
    parser.add_argument("--epochs", type=int, default=5,
                      help="Number of epochs")
    parser.add_argument("--output-dir", default="./benchmark_results",
                      help="Output directory")

    args = parser.parse_args()

    print("="*80)
    print("STABLE BENCHMARK - ForestGaps")
    print("="*80)
    print(f"Models: {args.models}")
    print(f"Config: {args.config}")
    print(f"Data dir: {args.data_dir}")
    print(f"Seeds: {args.n_seeds} (base={args.base_seed})")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {args.output_dir}")

    # Run benchmark
    results_df = run_multi_seed_benchmark(
        models=args.models,
        config_path=args.config,
        data_dir=args.data_dir,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        epochs=args.epochs,
        output_dir=args.output_dir
    )

    print(f"\n{'='*80}")
    print("✅ BENCHMARK COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
