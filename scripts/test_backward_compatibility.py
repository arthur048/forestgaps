#!/usr/bin/env python3
"""
Script de test de compatibilité backward pour ForestGaps.

Teste que toutes les fonctionnalités existantes continuent de fonctionner
après les modifications apportées au code (AttentionUNet fix, etc.).
"""

import sys
import torch
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def test_model_creation():
    """Test que tous les modèles peuvent être créés."""
    print_section("TEST 1: Model Creation")

    from forestgaps.models import create_model

    models_to_test = {
        "unet": {"in_channels": 1, "out_channels": 1, "init_features": 32},
        "film_unet": {"in_channels": 1, "out_channels": 1, "init_features": 32, "condition_size": 1},
        "attention_unet": {"in_channels": 1, "out_channels": 1, "init_features": 32, "depth": 4},
        "deeplabv3_plus": {"in_channels": 1, "out_channels": 1, "encoder_name": "resnet18", "encoder_weights": None},
        "res_unet": {"in_channels": 1, "out_channels": 1, "init_features": 32},
        "regression_unet": {"in_channels": 1, "out_channels": 1, "init_features": 32},
    }

    results = {}
    for model_name, kwargs in models_to_test.items():
        try:
            model = create_model(model_name, **kwargs)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✓ {model_name:20s}: {param_count:,} parameters")
            results[model_name] = "PASS"
        except Exception as e:
            print(f"✗ {model_name:20s}: {str(e)}")
            results[model_name] = f"FAIL: {str(e)}"

    return results

def test_model_forward():
    """Test que tous les modèles peuvent faire un forward pass."""
    print_section("TEST 2: Model Forward Pass")

    from forestgaps.models import create_model

    models_to_test = {
        "unet": {"in_channels": 1, "out_channels": 1, "init_features": 16},
        "film_unet": {"in_channels": 1, "out_channels": 1, "init_features": 16, "condition_size": 1},
        "attention_unet": {"in_channels": 1, "out_channels": 1, "init_features": 16, "depth": 4},
        "deeplabv3_plus": {"in_channels": 1, "out_channels": 1, "encoder_name": "resnet18", "encoder_weights": None},
        "res_unet": {"in_channels": 1, "out_channels": 1, "init_features": 16},
        "regression_unet": {"in_channels": 1, "out_channels": 1, "init_features": 16},
    }

    results = {}
    batch_size = 2
    height, width = 256, 256

    for model_name, kwargs in models_to_test.items():
        try:
            model = create_model(model_name, **kwargs)
            model.eval()

            # Create input
            inputs = torch.randn(batch_size, 1, height, width)

            # FiLM models need conditions
            if "film" in model_name:
                conditions = torch.randn(batch_size, 1)
                with torch.no_grad():
                    outputs = model(inputs, conditions)
            else:
                with torch.no_grad():
                    outputs = model(inputs)

            # Check output shape
            expected_shape = (batch_size, 1, height, width)
            if outputs.shape == expected_shape:
                print(f"✓ {model_name:20s}: {inputs.shape} → {outputs.shape}")
                results[model_name] = "PASS"
            else:
                print(f"✗ {model_name:20s}: Shape mismatch! Expected {expected_shape}, got {outputs.shape}")
                results[model_name] = f"FAIL: Shape mismatch"

        except Exception as e:
            print(f"✗ {model_name:20s}: {str(e)}")
            traceback.print_exc()
            results[model_name] = f"FAIL: {str(e)}"

    return results

def test_training_step():
    """Test qu'un training step fonctionne."""
    print_section("TEST 3: Training Step")

    from forestgaps.models import create_model
    from forestgaps.training.losses import ComboLoss

    results = {}

    try:
        # Create model
        model = create_model("unet", in_channels=1, out_channels=1, init_features=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = ComboLoss(bce_weight=1.0, dice_weight=1.0, focal_weight=1.0)

        # Create dummy data
        inputs = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss, breakdown = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  BCE:   {breakdown['bce_loss']:.4f}")
        print(f"  Dice:  {breakdown['dice_loss']:.4f}")
        print(f"  Focal: {breakdown['focal_loss']:.4f}")
        results["training_step"] = "PASS"

    except Exception as e:
        print(f"✗ Training step failed: {str(e)}")
        traceback.print_exc()
        results["training_step"] = f"FAIL: {str(e)}"

    return results

def test_config_loading():
    """Test que les configs peuvent être chargées."""
    print_section("TEST 4: Configuration Loading")

    from forestgaps.config import load_training_config

    configs_to_test = [
        "configs/test/minimal.yaml",
        "configs/test/quick.yaml",
    ]

    results = {}
    for config_path in configs_to_test:
        try:
            config = load_training_config(config_path)
            print(f"✓ {config_path:30s}: epochs={config.epochs}, batch_size={config.batch_size}")
            results[config_path] = "PASS"
        except Exception as e:
            print(f"✗ {config_path:30s}: {str(e)}")
            results[config_path] = f"FAIL: {str(e)}"

    return results

def test_attention_unet_specifically():
    """Test spécifique pour AttentionUNet (le modèle réparé)."""
    print_section("TEST 5: AttentionUNet Specific Tests")

    from forestgaps.models import create_model

    results = {}

    # Test different depths
    depths_to_test = [3, 4, 5]

    for depth in depths_to_test:
        try:
            model = create_model("attention_unet", in_channels=1, out_channels=1, init_features=16, depth=depth)

            # Test forward pass
            inputs = torch.randn(1, 1, 128, 128)
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)

            # Check shape
            if outputs.shape == inputs.shape:
                print(f"✓ AttentionUNet depth={depth}: {inputs.shape} → {outputs.shape}")
                results[f"depth_{depth}"] = "PASS"
            else:
                print(f"✗ AttentionUNet depth={depth}: Shape mismatch! Expected {inputs.shape}, got {outputs.shape}")
                results[f"depth_{depth}"] = f"FAIL: Shape mismatch"

        except Exception as e:
            print(f"✗ AttentionUNet depth={depth}: {str(e)}")
            traceback.print_exc()
            results[f"depth_{depth}"] = f"FAIL: {str(e)}"

    # Test gradient flow
    try:
        model = create_model("attention_unet", in_channels=1, out_channels=1, init_features=16, depth=4)
        model.train()

        inputs = torch.randn(1, 1, 128, 128)
        targets = torch.randn(1, 1, 128, 128)

        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()

        # Check that gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())

        if has_grads:
            print(f"✓ AttentionUNet gradient flow: OK")
            results["gradient_flow"] = "PASS"
        else:
            print(f"✗ AttentionUNet gradient flow: No gradients!")
            results["gradient_flow"] = "FAIL: No gradients"

    except Exception as e:
        print(f"✗ AttentionUNet gradient flow: {str(e)}")
        traceback.print_exc()
        results["gradient_flow"] = f"FAIL: {str(e)}"

    return results

def main():
    """Run all backward compatibility tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "FORESTGAPS BACKWARD COMPATIBILITY TESTS" + " " * 19 + "║")
    print("╚" + "═" * 78 + "╝")

    all_results = {}

    # Run all tests
    all_results["model_creation"] = test_model_creation()
    all_results["model_forward"] = test_model_forward()
    all_results["training_step"] = test_training_step()
    all_results["config_loading"] = test_config_loading()
    all_results["attention_unet"] = test_attention_unet_specifically()

    # Print summary
    print_section("SUMMARY")

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for category, results in all_results.items():
        print(f"\n{category.upper()}:")
        for test_name, result in results.items():
            total_tests += 1
            if result == "PASS":
                passed_tests += 1
                print(f"  ✓ {test_name}")
            else:
                failed_tests += 1
                print(f"  ✗ {test_name}: {result}")

    print("\n" + "=" * 80)
    print(f"TOTAL: {total_tests} tests")
    print(f"✓ PASSED: {passed_tests}")
    print(f"✗ FAILED: {failed_tests}")
    print("=" * 80)

    # Return exit code
    return 0 if failed_tests == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
