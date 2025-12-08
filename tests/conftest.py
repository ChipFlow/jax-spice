"""Pytest configuration for JAX-SPICE tests

Handles platform-specific JAX configuration:
- macOS: Forces CPU backend since Metal doesn't support triangular_solve
- Linux with CUDA: Preloads CUDA libraries to help JAX discover them
- Linux with TPU: Uses TPU backend when JAX_PLATFORMS=tpu is set

Uses pytest_configure hook to ensure backend setup happens before any test imports.
"""

import os
import sys


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
            ctypes.CDLL(lib)
        except OSError:
            pass  # Some libraries may not be available


def pytest_configure(config):
    """
    Pytest hook that runs before test collection.

    This ensures backend libraries are preloaded and JAX is configured
    BEFORE any test modules are imported.
    """
    jax_platforms = os.environ.get('JAX_PLATFORMS', '')

    # Platform-specific configuration BEFORE importing JAX
    if sys.platform == 'darwin':
        # macOS: Force CPU backend - Metal doesn't support triangular_solve
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif sys.platform == 'linux' and jax_platforms.startswith('cuda'):
        # Linux with CUDA: Preload CUDA libraries before JAX import
        _setup_cuda_libraries()
    elif sys.platform == 'linux' and jax_platforms == 'tpu':
        # Linux with TPU: JAX handles TPU initialization via libtpu
        # No special preloading needed - libtpu is installed with jax[tpu]
        pass

    # Import JAX and configure it
    import jax

    # Enable float64 for numerical precision in tests
    jax.config.update('jax_enable_x64', True)
