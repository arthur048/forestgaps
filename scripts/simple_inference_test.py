#!/usr/bin/env python
"""
Script d'inférence minimal pour ForestGaps.
Simple, efficace, qui marche.
"""

import torch
import rasterio
import numpy as np
from pathlib import Path
from forestgaps.models.registry import model_registry

def simple_inference(model_path, dsm_path, output_path):
    """
    Fait de l'inférence simple sur un DSM.

    Args:
        model_path: Chemin vers le modèle .pt
        dsm_path: Chemin vers le DSM d'entrée
        output_path: Chemin pour sauvegarder la prédiction
    """
    print(f"=== INFERENCE SIMPLE ===")
    print(f"Model: {model_path}")
    print(f"DSM: {dsm_path}")
    print(f"Output: {output_path}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Charger le DSM
    print("\n1. Chargement DSM...")
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        profile = src.profile
        print(f"   DSM shape: {dsm.shape}")

    # Normaliser
    print("\n2. Normalisation...")
    dsm_norm = (dsm - np.nanmean(dsm)) / (np.nanstd(dsm) + 1e-6)
    dsm_norm = np.nan_to_num(dsm_norm, 0)

    # Charger le modèle
    print("\n3. Chargement modèle...")
    checkpoint = torch.load(model_path, map_location=device)
    print(f"   Checkpoint keys: {list(checkpoint.keys())}")

    # Pour simple_training_test.py, le checkpoint contient directement les poids
    # Créer un SimpleUNet
    from scripts.simple_training_test import SimpleUNet
    model = SimpleUNet().to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"   Model loaded: SimpleUNet")

    # Inference
    print("\n4. Inférence...")
    with torch.no_grad():
        dsm_tensor = torch.from_numpy(dsm_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        pred = model(dsm_tensor)
        pred = pred.squeeze().cpu().numpy()

    print(f"   Prediction shape: {pred.shape}")
    print(f"   Prediction range: [{pred.min():.3f}, {pred.max():.3f}]")

    # Sauvegarder
    print("\n5. Sauvegarde...")
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pred, 1)

    print(f"✅ Sauvegardé: {output_path}")
    return pred

if __name__ == "__main__":
    # Paths
    model_path = Path("/tmp/outputs/best_model.pt")
    dsm_dir = Path("/app/forestgaps/data/processed/tiles/train")
    dsm_files = list(dsm_dir.glob("*_dsm.tif"))

    if not dsm_files:
        print("❌ Aucun DSM trouvé")
        exit(1)

    dsm_path = dsm_files[0]  # Premier DSM
    output_path = Path("/tmp/outputs/inference_test.tif")

    # Run inference
    simple_inference(model_path, dsm_path, output_path)

    print("\n" + "="*60)
    print("✅ INFERENCE TEST COMPLETED!")
    print("="*60)
