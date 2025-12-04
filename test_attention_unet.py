"""
Quick test to verify AttentionUNet fix.
"""

import torch
from forestgaps.models import create_model

def test_attention_unet():
    """Test AttentionUNet creation and forward pass."""
    print("=" * 60)
    print("Testing AttentionUNet Fix")
    print("=" * 60)

    # Create model
    print("\n1. Creating AttentionUNet model...")
    try:
        model = create_model(
            "attention_unet",
            in_channels=1,
            out_channels=1,
            init_features=32,  # Smaller for faster test
            depth=4
        )
        print(f"✓ Model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

    # Test forward pass
    print("\n2. Testing forward pass...")
    try:
        model.eval()
        inputs = torch.randn(2, 1, 256, 256)  # Batch of 2

        with torch.no_grad():
            outputs = model(inputs)

        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {inputs.shape}")
        print(f"  - Output shape: {outputs.shape}")
        print(f"  - Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

        # Verify shape
        assert outputs.shape == inputs.shape, "Output shape mismatch!"
        print(f"✓ Output shape matches input shape")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test training mode
    print("\n3. Testing training step...")
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        inputs = torch.randn(2, 1, 256, 256)
        targets = torch.randint(0, 2, (2, 1, 256, 256)).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"✓ Training step successful")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Gradients computed: {all(p.grad is not None for p in model.parameters() if p.requires_grad)}")

    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed! AttentionUNet is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_attention_unet()
    exit(0 if success else 1)
