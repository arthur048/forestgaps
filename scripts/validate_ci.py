#!/usr/bin/env python
"""
Script de validation CI pour ForestGaps.

Valide que tous les composants critiques fonctionnent avant push.
À exécuter avant chaque commit important.

Usage:
    python scripts/validate_ci.py
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Execute une commande et retourne le résultat."""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ SUCCESS")
        if result.stdout:
            print(result.stdout)
        return True
    else:
        print(f"❌ FAILED (exit code {result.returncode})")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)
        return False


def validate_all():
    """Valide tous les composants."""

    print("\n" + "="*70)
    print("VALIDATION CI COMPLÈTE - ForestGaps")
    print("="*70)

    tests = []

    # Test 1: Import package
    tests.append(run_command(
        'python -c "import forestgaps; print(f\'Version: {forestgaps.__version__}\')"',
        "Import package principal"
    ))

    # Test 2: Model registry
    tests.append(run_command(
        'python -c "from forestgaps.models import model_registry; models = model_registry.list_models(); print(f\'{len(models)} modèles disponibles\'); assert len(models) == 9"',
        "Model registry (9 modèles)"
    ))

    # Test 3: Inference module
    tests.append(run_command(
        'python -c "from forestgaps.inference import InferenceManager; print(\'✅ Inference OK\')"',
        "Module inference"
    ))

    # Test 4: Evaluation module
    tests.append(run_command(
        'python -c "from forestgaps.evaluation import evaluate_model; print(\'✅ Evaluation OK\')"',
        "Module evaluation"
    ))

    # Test 5: Training module
    tests.append(run_command(
        'python -c "from forestgaps.training import Trainer; print(\'✅ Training OK\')"',
        "Module training"
    ))

    # Test 6: Benchmarking module
    tests.append(run_command(
        'python -c "from forestgaps.benchmarking import ModelComparison; print(\'✅ Benchmarking OK\')"',
        "Module benchmarking"
    ))

    # Test 7: Test tous les modèles
    if Path("scripts/test_all_models.py").exists():
        tests.append(run_command(
            'python scripts/test_all_models.py',
            "Test tous les modèles"
        ))

    # Test 8: Pytest suite (si disponible)
    if Path("tests/test_complete_workflow.py").exists():
        tests.append(run_command(
            'pytest tests/test_complete_workflow.py -v --tb=short',
            "Suite de tests pytest"
        ))

    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ VALIDATION CI")
    print("="*70)

    success = sum(tests)
    total = len(tests)

    print(f"\nRésultat: {success}/{total} tests passés ({100*success/total:.1f}%)")

    if success == total:
        print("\n✅ TOUS LES TESTS PASSENT - PRÊT POUR COMMIT/PUSH")
        return True
    else:
        print("\n❌ CERTAINS TESTS ONT ÉCHOUÉ - NE PAS PUSH")
        print(f"\nTests en échec: {total - success}/{total}")
        return False


if __name__ == "__main__":
    success = validate_all()
    sys.exit(0 if success else 1)
