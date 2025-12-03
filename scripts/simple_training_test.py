#!/usr/bin/env python
"""
Script de test d'entraînement minimal pour ForestGaps.
Simple, efficace, qui marche.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import rasterio
import numpy as np
from tqdm import tqdm

# Simple U-Net minimal
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d = self.up(e2)
        d = torch.cat([d, e1], dim=1)
        return self.dec(d)

# Simple Dataset
class SimpleForestDataset(Dataset):
    def __init__(self, tiles_dir):
        self.tiles_dir = Path(tiles_dir)
        self.dsm_files = sorted(self.tiles_dir.glob("*_dsm.tif"))
        self.mask_files = sorted(self.tiles_dir.glob("*_mask.tif"))
        print(f"Found {len(self.dsm_files)} DSM tiles and {len(self.mask_files)} masks")

    def __len__(self):
        return len(self.dsm_files)

    def __getitem__(self, idx):
        with rasterio.open(self.dsm_files[idx]) as src:
            dsm = src.read(1).astype(np.float32)
        with rasterio.open(self.mask_files[idx]) as src:
            mask = src.read(1).astype(np.float32)

        # Normaliser
        dsm = (dsm - np.nanmean(dsm)) / (np.nanstd(dsm) + 1e-6)
        mask = mask / 255.0 if mask.max() > 1 else mask

        # Gérer les NaN
        dsm = np.nan_to_num(dsm, 0)
        mask = np.nan_to_num(mask, 0)

        return torch.FloatTensor(dsm).unsqueeze(0), torch.FloatTensor(mask).unsqueeze(0)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for dsm, mask in tqdm(loader, desc="Training"):
        dsm, mask = dsm.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(dsm)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for dsm, mask in loader:
            dsm, mask = dsm.to(device), mask.to(device)
            output = model(dsm)
            loss = criterion(output, mask)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    print("=" * 60)
    print("SIMPLE TRAINING TEST - ForestGaps")
    print("=" * 60)

    # Config
    tiles_dir = Path("/app/forestgaps/data/processed/tiles/train")
    epochs = 3
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Dataset
    print("\nLoading dataset...")
    dataset = SimpleForestDataset(tiles_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

    print(f"Train: {len(train_ds)} tiles, Val: {len(val_ds)} tiles")

    # Model
    print("\nCreating model...")
    model = SimpleUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("\nTraining...")
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = val_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/tmp/outputs/best_model.pt')
            print("✓ Model saved!")

    print("\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETED!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: /tmp/outputs/best_model.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()
