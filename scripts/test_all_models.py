#!/usr/bin/env python
"""
Script de test complet de tous les modèles ForestGaps.

Test tous les modèles du registry avec forward pass et complexité.
À exécuter régulièrement pour validation.

Usage:
    python scripts/test_all_models.py
"""

import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forestgaps.models import model_registry


def test_all_models():
    """Teste l'instantiation et forward pass de tous les modèles."""

    print("="*70)
    print("TEST COMPLET DE TOUS LES MODÈLES")
    print("="*70)

    all_models = model_registry.list_models()
    print(f"\nNombre de modèles à tester: {len(all_models)}")
    print(f"Modèles: {', '.join(all_models)}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    results = {}
    dummy_input = torch.randn(2, 1, 256, 256).to(device)

    for i, model_name in enumerate(all_models, 1):
        print(f"[{i}/{len(all_models)}] Test: {model_name:30s} ... ", end='', flush=True)

        try:
            # Paramètres selon type
            if 'regression' in model_name:
                params = {'in_channels': 1, 'out_channels': 1}
            elif 'threshold' in model_name:
                params = {'in_channels': 1, 'out_channels': 1, 'threshold_value': 5.0}
            else:
                params = {'in_channels': 1, 'out_channels': 1}

            # Créer modèle
            model = model_registry.create(model_name, **params)
            model = model.to(device)
            model.eval()

            # Test forward pass
            with torch.no_grad():
                output = model(dummy_input)

            # Vérifications
            assert output.shape == (2, 1, 256, 256), f"Wrong output shape: {output.shape}"
            assert not torch.isnan(output).any(), "NaN in output"
            assert not torch.isinf(output).any(), "Inf in output"

            # Complexité
            n_params = model.get_num_parameters()

            results[model_name] = {
                'status': 'OK',
                'params': n_params,
                'output_shape': str(output.shape),
                'output_range': f'[{output.min():.3f}, {output.max():.3f}]'
            }

            print(f"✅ OK ({n_params:>10,} params)")

        except Exception as e:
            results[model_name] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"❌ FAILED: {str(e)[:50]}")

    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)

    success = sum(1 for r in results.values() if r['status'] == 'OK')
    total = len(results)

    print(f"\nRésultat: {success}/{total} modèles OK ({100*success/total:.1f}%)")

    if success < total:
        print("\n❌ MODÈLES EN ÉCHEC:")
        for name, result in results.items():
            if result['status'] != 'OK':
                print(f"  - {name}: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    else:
        print("\n✅ TOUS LES MODÈLES FONCTIONNENT!")

        print("\nDétails:")
        print(f"{'Modèle':<35} {'Paramètres':>15} {'Output Range':<20}")
        print("-"*70)
        for name, result in results.items():
            if result['status'] == 'OK':
                print(f"{name:<35} {result['params']:>15,} {result['output_range']:<20}")

    return success == total


if __name__ == "__main__":
    success = test_all_models()
    sys.exit(0 if success else 1)
