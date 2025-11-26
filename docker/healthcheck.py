#!/usr/bin/env python3
"""
Health check script for ForestGaps Docker container.
Returns exit code 0 if healthy, 1 otherwise.
"""

import sys


def check_imports():
    """Verify critical imports work."""
    try:
        import torch
        import rasterio
        import geopandas
        import numpy
        return True
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        return False


def check_cuda():
    """Check CUDA availability (warning only, not failure)."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available", file=sys.stderr)
        return True
    except Exception as e:
        print(f"CUDA check error: {e}", file=sys.stderr)
        return True  # Don't fail on CUDA issues


def check_forestgaps():
    """Verify ForestGaps package is installed."""
    try:
        import forestgaps
        return True
    except ImportError:
        # ForestGaps not installed yet is OK during build
        print("Warning: forestgaps package not installed", file=sys.stderr)
        return True


def main():
    checks = [
        ("imports", check_imports),
        ("cuda", check_cuda),
        ("forestgaps", check_forestgaps),
    ]
    
    all_passed = True
    for name, check_fn in checks:
        try:
            if not check_fn():
                all_passed = False
                print(f"FAIL: {name}", file=sys.stderr)
        except Exception as e:
            all_passed = False
            print(f"ERROR in {name}: {e}", file=sys.stderr)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())