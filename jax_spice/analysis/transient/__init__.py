"""Transient analysis for JAX-SPICE.

This module provides transient analysis capabilities using OpenVAF-compiled devices:

Strategy classes for transient simulation:

- PythonLoopStrategy: Traditional Python for-loop with JIT-compiled NR solver
  - Full convergence tracking per timestep
  - Easy to debug and profile
  - Moderate performance (~0.5ms/step)

- ScanStrategy: Fully JIT-compiled using lax.scan
  - Best performance (~0.1ms/step on CPU)
  - 5x+ speedup over Python loop
  - Limited per-step debugging

Strategy Usage:
    from jax_spice.analysis.transient import PythonLoopStrategy, ScanStrategy

    # Using Python loop (default, more debugging info)
    strategy = PythonLoopStrategy(runner, use_sparse=False)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)

    # Using lax.scan (faster, but requires matching warmup steps)
    strategy = ScanStrategy(runner, use_sparse=False)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)
"""

# Strategy classes for OpenVAF-based transient
from .base import TransientSetup, TransientStrategy
from .python_loop import PythonLoopStrategy
from .scan import ScanStrategy

__all__ = [
    # Strategy classes for OpenVAF
    'TransientStrategy',
    'TransientSetup',
    'PythonLoopStrategy',
    'ScanStrategy',
]
