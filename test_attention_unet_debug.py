"""
Debug test for AttentionUNet with detailed shape logging.
"""

import torch
from forestgaps.models import create_model

def test_attention_unet_debug():
    """Test AttentionUNet with detailed logging."""
    print("=" * 60)
    print("AttentionUNet Debug Test")
    print("=" * 60)

    # Create model
    model = create_model(
        "attention_unet",
        in_channels=1,
        out_channels=1,
        init_features=32,
        depth=4
    )
    model.eval()

    # Patch forward to add logging
    original_forward = model.forward

    def logged_forward(x):
        print(f"\nInput: {x.shape}")

        # Encoder
        encoder_features = []
        for i, encoder_block in enumerate(model.encoder_blocks):
            if i > 0:
                x = model.downsample_blocks[i-1](x)
                print(f"After downsample[{i-1}]: {x.shape}")
            x = encoder_block(x)
            encoder_features.append(x)
            print(f"Encoder[{i}]: {x.shape}")

        # Bottleneck
        x = model.bottleneck(x)
        print(f"\nBottleneck: {x.shape}")

        # Decoder
        for i in range(len(model.decoder_blocks)):
            skip = encoder_features[-(i+1)]
            print(f"\nDecoder iteration {i}:")
            print(f"  x before upsample: {x.shape}")
            print(f"  skip: {skip.shape}")

            # Upsample
            x = model.upsample_blocks[i](x)
            print(f"  x after upsample: {x.shape}")

            # Attention
            skip_attention = model.attention_gates[i](g=x, x=skip)
            print(f"  skip_attention: {skip_attention.shape}")

            # Concat
            print(f"  Trying to concat: {x.shape} + {skip_attention.shape}")
            x = torch.cat([x, skip_attention], dim=1)
            print(f"  After concat: {x.shape}")

            # Conv
            x = model.decoder_blocks[i](x)
            print(f"  After decoder block: {x.shape}")

        # Final
        x = model.final_conv(x)
        print(f"\nOutput: {x.shape}")
        return x

    model.forward = logged_forward

    # Test
    inputs = torch.randn(2, 1, 256, 256)
    try:
        with torch.no_grad():
            outputs = model(inputs)
        print("\n✓ SUCCESS!")
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_attention_unet_debug()
