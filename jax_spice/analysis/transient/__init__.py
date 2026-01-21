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

- AdaptiveStrategy: LTE-based adaptive timestep control (RECOMMENDED)
  - Automatically adjusts timestep based on solution accuracy
  - Smaller steps during fast transients, larger steps during slow evolution
  - Uses predictor-corrector scheme with polynomial extrapolation
  - Compatible with VACASK for validation
  - Python loop with JIT-compiled NR solver (~1.7ms/step)

- AdaptiveScanStrategy: Adaptive timestep using lax.scan (SLOWER - for reference)
  - Full JIT compilation with lax.scan
  - Much slower than Python loop (~170ms/step) due to nested lax.cond overhead
  - Kept for completeness but NOT recommended for production use

- AdaptiveWhileLoopStrategy: Adaptive timestep using lax.while_loop (SLOWER - for reference)
  - Full JIT compilation with lax.while_loop
  - Much slower than Python loop (~250ms/step) due to jnp.where tracing overhead
  - Kept for completeness but NOT recommended for production use

- FullMNAStrategy: Full Modified Nodal Analysis with explicit branch currents
  - True MNA formulation (not high-G approximation)
  - More accurate current extraction
  - Smoother dI/dt matching VACASK reference

Performance Notes:
  For adaptive timestep, the Python loop approach is fastest because:
  - The heavy computation (Newton-Raphson) is JIT-compiled
  - Python handles dynamic control flow naturally
  - lax.scan/while_loop incur massive tracing overhead for complex bodies

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
