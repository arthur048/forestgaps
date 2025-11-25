#!/usr/bin/env python3
"""
ForestGaps Docker Health Check Script

Verifies that the Docker container is properly configured:
- All critical Python packages import correctly
- GPU is accessible (if available on host)
- Environment detection works
- GDAL/rasterio compatibility
"""

import sys
import os


def check_imports():
    """Verify that all critical packages can be imported."""
    print("🔍 Checking critical imports...")

    try:
        import torch
        print(f"  ✅ torch {torch.__version__}")

        import torchvision
        print(f"  ✅ torchvision {torchvision.__version__}")

        from osgeo import gdal
        print(f"  ✅ GDAL {gdal.__version__}")

        import rasterio
        print(f"  ✅ rasterio {rasterio.__version__}")

        import geopandas
        print(f"  ✅ geopandas {geopandas.__version__}")

        import forestgaps
        print(f"  ✅ forestgaps {forestgaps.__version__}")

        return True

    except ImportError as e:
        print(f"  ❌ Import error: {e}", file=sys.stderr)
        return False


def check_gpu():
    """Verify GPU availability and accessibility."""
    print("\n🔍 Checking GPU availability...")

    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda

            print(f"  ✅ GPU available: {device_name}")
            print(f"  ✅ Device count: {device_count}")
            print(f"  ✅ CUDA version: {cuda_version}")
            print(f"  ✅ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  ✅ cuDNN enabled: {torch.backends.cudnn.enabled}")

            # Test a simple GPU operation
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = x * 2
            assert y.is_cuda, "Tensor not on CUDA device"
            print(f"  ✅ GPU computation test passed")

            return True
        else:
            print("  ⚠️  GPU not available (running in CPU mode)", file=sys.stderr)
            # Don't fail health check - CPU mode is valid
            return True

    except Exception as e:
        print(f"  ❌ GPU check error: {e}", file=sys.stderr)
        return False


def check_gdal_rasterio():
    """Verify GDAL and rasterio compatibility."""
    print("\n🔍 Checking GDAL/rasterio compatibility...")

    try:
        from osgeo import gdal
        import rasterio

        gdal_version = gdal.__version__
        rasterio_version = rasterio.__version__

        print(f"  ✅ GDAL version: {gdal_version}")
        print(f"  ✅ Rasterio version: {rasterio_version}")

        # Check GDAL_DATA environment variable
        gdal_data = os.environ.get('GDAL_DATA')
        if gdal_data and os.path.exists(gdal_data):
            print(f"  ✅ GDAL_DATA: {gdal_data}")
        else:
            print(f"  ⚠️  GDAL_DATA not set or invalid: {gdal_data}", file=sys.stderr)

        # Check PROJ_LIB environment variable
        proj_lib = os.environ.get('PROJ_LIB')
        if proj_lib and os.path.exists(proj_lib):
            print(f"  ✅ PROJ_LIB: {proj_lib}")
        else:
            print(f"  ⚠️  PROJ_LIB not set or invalid: {proj_lib}", file=sys.stderr)

        return True

    except Exception as e:
        print(f"  ❌ GDAL/rasterio check error: {e}", file=sys.stderr)
        return False


def check_environment():
    """Verify environment detection works correctly."""
    print("\n🔍 Checking environment detection...")

    try:
        from forestgaps.environment import Environment

        env = Environment.detect()
        info = env.get_environment_info()

        env_type = info.get('environment_type', 'Unknown')
        print(f"  ✅ Detected environment: {env_type}")

        # In Docker, should detect DockerEnvironment
        if '/.dockerenv' in str(os.path.exists('/.dockerenv')):
            if 'Docker' in env_type:
                print(f"  ✅ Docker environment correctly detected")
            else:
                print(f"  ⚠️  Expected DockerEnvironment but got: {env_type}", file=sys.stderr)

        # Print environment info
        for key, value in info.items():
            if key != 'environment_type':
                print(f"  ℹ️  {key}: {value}")

        return True

    except Exception as e:
        print(f"  ❌ Environment check error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def check_directories():
    """Verify expected directories exist."""
    print("\n🔍 Checking directory structure...")

    expected_dirs = [
        '/app',
        '/app/forestgaps',
        '/app/data',
        '/app/models',
        '/app/outputs',
        '/app/logs'
    ]

    all_exist = True
    for directory in expected_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}")
        else:
            print(f"  ❌ Missing: {directory}", file=sys.stderr)
            all_exist = False

    return all_exist


def main():
    """Run all health checks."""
    print("=" * 60)
    print("ForestGaps Docker Health Check")
    print("=" * 60)

    checks = [
        ("Imports", check_imports),
        ("GPU", check_gpu),
        ("GDAL/Rasterio", check_gdal_rasterio),
        ("Environment", check_environment),
        ("Directories", check_directories)
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed with exception: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Health Check Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All health checks passed!")
        return 0
    else:
        print("\n⚠️  Some health checks failed. See details above.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
