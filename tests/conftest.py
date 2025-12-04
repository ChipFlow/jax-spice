"""Pytest configuration for JAX-SPICE tests

Handles platform-specific JAX configuration:
- macOS: Forces CPU backend since Metal doesn't support triangular_solve
- Linux with CUDA: Preloads CUDA libraries to help JAX discover them

Uses pytest_configure hook to ensure CUDA setup happens before any test imports.
"""

import os
import sys

def log(msg):
    """Log to stderr with flush to ensure output is captured"""
    sys.stderr.write(f"[conftest.py] {msg}\n")
    sys.stderr.flush()


def _setup_cuda_libraries():
    """Preload CUDA libraries before JAX import for proper GPU detection."""
    import ctypes

    cuda_libs = [
        "libcuda.so.1",
        "libcudart.so.12",
        "libnvrtc.so.12",
        "libnvJitLink.so.12",
        "libcusparse.so.12",
        "libcublas.so.12",
        "libcusolver.so.11",
    ]

    for lib in cuda_libs:
        try:
            handle = ctypes.CDLL(lib)
            log(f"Loaded {lib} -> {handle}")
        except OSError as e:
            log(f"FAILED to load {lib}: {e}")


def pytest_configure(config):
    """
    Pytest hook that runs before test collection.

    This ensures CUDA libraries are preloaded and JAX is configured
    BEFORE any test modules are imported.
    """
    log(f"pytest_configure called - platform={sys.platform}")
    log(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', 'NOT SET')}")
    log(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:200]}...")

    # Platform-specific configuration BEFORE importing JAX
    if sys.platform == 'darwin':
        # macOS: Force CPU backend - Metal doesn't support triangular_solve
        log("macOS detected - setting JAX_PLATFORMS=cpu")
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif sys.platform == 'linux' and os.environ.get('JAX_PLATFORMS', '').startswith('cuda'):
        # Linux with CUDA: Preload CUDA libraries before JAX import
        log("Linux+CUDA detected - preloading CUDA libraries")
        _setup_cuda_libraries()
    else:
        log(f"No special handling - platform={sys.platform}, JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', 'NOT SET')}")

    # Now import JAX and configure it
    log("About to import JAX...")
    import jax
    log(f"JAX imported - default_backend={jax.default_backend()}")
    log(f"JAX devices: {jax.devices()}")

    # Enable float64 for numerical precision in tests
    jax.config.update('jax_enable_x64', True)
    log("JAX configuration complete")
