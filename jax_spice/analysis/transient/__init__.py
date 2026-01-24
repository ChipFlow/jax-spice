"""Transient analysis for JAX-SPICE.

This module provides transient analysis capabilities using OpenVAF-compiled devices:

Strategy classes for transient simulation:

- PythonLoopStrategy: Traditional Python for-loop with JIT-compiled NR solver
  - Full convergence tracking per timestep
  - Easy to debug and profile
  - Moderate performance (~0.5ms/step)

- ScanStrategy: Fully JIT-compiled using lax.scan
  - Best performance (~0.1ms/step on CPU) for fixed timestep
  - 5x+ speedup over Python loop
  - Limited per-step debugging

- AdaptiveStrategy: Adaptive timestep with Python loop (RECOMMENDED for adaptive)
  - Automatically adjusts timestep based on LTE (Local Truncation Error)
  - Smaller steps during fast transients, larger steps during slow evolution
  - Uses predictor-corrector scheme with polynomial extrapolation
  - Compatible with VACASK for validation
  - Python loop with JIT-compiled NR solver (~1ms/step)
  - Most reliable across all circuit sizes - no JIT compilation issues

- AdaptiveWhileLoopStrategy: Adaptive timestep using lax.while_loop
  - Same algorithm as AdaptiveStrategy, fully JIT-compiled
  - Early termination when t >= t_stop
  - CAUTION: Can have very long JIT compilation times for complex circuits
  - Best for small circuits with simple device models

- AdaptiveScanStrategy: Adaptive timestep using lax.scan (NOT recommended)
  - Full JIT compilation with lax.scan
  - Much slower (~10-18ms/step) - runs ALL max_steps iterations
  - Kept for completeness but not recommended for production use

- FullMNAStrategy: Full Modified Nodal Analysis with explicit branch currents
  - True MNA formulation (not high-G approximation)
  - More accurate current extraction
  - Smoother dI/dt matching VACASK reference

Performance Notes:
  For adaptive timestep, AdaptiveStrategy (Python loop) is recommended:
  - Reliable across all circuit sizes and complexities
  - NR solver is JIT-compiled (heavy computation is fast)
  - Python loop handles dynamic control flow naturally
  - ~1ms/step overhead is negligible compared to solver work

  lax.while_loop/scan approaches have JIT compilation issues for complex circuits

Strategy Usage:
    from jax_spice.analysis.transient import PythonLoopStrategy, ScanStrategy

    # Using Python loop (default, more debugging info)
    strategy = PythonLoopStrategy(runner, use_sparse=False)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)

    # Using lax.scan (faster, but requires matching warmup steps)
    strategy = ScanStrategy(runner, use_sparse=False)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)

    # Using adaptive timestep (best accuracy, automatic step sizing)
    from jax_spice.analysis.transient import AdaptiveStrategy, AdaptiveConfig
    config = AdaptiveConfig(lte_ratio=3.5, redo_factor=2.5)
    strategy = AdaptiveStrategy(runner, config=config)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)

    # Using full MNA for accurate current extraction
    from jax_spice.analysis.transient import FullMNAStrategy
    strategy = FullMNAStrategy(runner, use_sparse=False)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)
    I_VDD = stats['currents']['vdd']  # Direct branch current from solution
"""

# Strategy classes for OpenVAF-based transient
from .adaptive import (
    AdaptiveConfig,
    AdaptiveScanStrategy,
    AdaptiveStrategy,
    AdaptiveWhileLoopStrategy,
)
from .base import TransientSetup, TransientStrategy
from .full_mna import FullMNAStrategy
from .python_loop import PythonLoopStrategy
from .scan import ScanStrategy

__all__ = [
    # Strategy classes for OpenVAF
    "TransientStrategy",
    "TransientSetup",
    "PythonLoopStrategy",
    "ScanStrategy",
    "AdaptiveStrategy",
    "AdaptiveScanStrategy",
    "AdaptiveWhileLoopStrategy",
    "AdaptiveConfig",
    "FullMNAStrategy",
]
