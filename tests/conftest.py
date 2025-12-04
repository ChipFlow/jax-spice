"""Pytest configuration for JAX-SPICE tests

Handles platform-specific JAX configuration:
- macOS: Forces CPU backend since Metal doesn't support triangular_solve
- Linux with CUDA: Preloads CUDA libraries to help JAX discover them
"""

import os
import sys

print(f"[conftest.py] Loading conftest.py - platform={sys.platform}")
print(f"[conftest.py] JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', 'NOT SET')}")
print(f"[conftest.py] LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', 'NOT SET')[:200]}...")

# Platform-specific configuration BEFORE importing JAX
if sys.platform == 'darwin':
    # macOS: Force CPU backend - Metal doesn't support triangular_solve
    print("[conftest.py] macOS detected - setting JAX_PLATFORMS=cpu")
    os.environ['JAX_PLATFORMS'] = 'cpu'
elif sys.platform == 'linux' and os.environ.get('JAX_PLATFORMS', '').startswith('cuda'):
    # Linux with CUDA: Preload CUDA libraries before JAX import
    # JAX's library discovery needs libraries to be loaded first
    print("[conftest.py] Linux+CUDA detected - preloading CUDA libraries")
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
            print(f"[conftest.py] Loaded {lib} -> {handle}")
        except OSError as e:
            print(f"[conftest.py] FAILED to load {lib}: {e}")
else:
    print(f"[conftest.py] No special handling - platform={sys.platform}, JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', 'NOT SET')}")

print("[conftest.py] About to import JAX...")
import jax
print(f"[conftest.py] JAX imported - default_backend={jax.default_backend()}")
print(f"[conftest.py] JAX devices: {jax.devices()}")

# Enable float64 for numerical precision in tests
jax.config.update('jax_enable_x64', True)
