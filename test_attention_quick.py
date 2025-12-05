"""Quick test AttentionUNet refactoré"""
import torch
import sys
sys.path.insert(0, 'g:/Mon Drive/forestgaps-dl')

from forestgaps.models import create_model

print("Testing AttentionUNet...")

# Create model
model = create_model(
    "attention_unet",
    in_channels=1,
    out_channels=1,
    init_features=32,
    depth=4
)
print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} params")

# Test forward
model.eval()
inputs = torch.randn(2, 1, 256, 256)
with torch.no_grad():
    outputs = model(inputs)

print(f"✓ Forward pass: {inputs.shape} → {outputs.shape}")

if outputs.shape == inputs.shape:
    print("✅ SUCCESS! AttentionUNet is fixed!")
else:
    print(f"❌ Shape mismatch: expected {inputs.shape}, got {outputs.shape}")

# Test training step
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

print(f"✓ Training step: loss={loss.item():.4f}")
print("✅ ALL TESTS PASSED!")
