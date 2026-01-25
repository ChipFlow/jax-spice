"""Transient analysis for JAX-SPICE.

This module provides transient analysis using full Modified Nodal Analysis (MNA)
with explicit branch currents for voltage sources. This provides:

- Better numerical conditioning (no G=1e12 high-G approximation)
- More accurate current extraction (branch currents are primary unknowns)
- Smoother dI/dt transitions matching VACASK reference

Usage:
    from jax_spice.analysis.transient import FullMNAStrategy, AdaptiveConfig

    # Default: adaptive timestep (recommended)
    config = AdaptiveConfig(lte_ratio=3.5, redo_factor=2.5)
    strategy = FullMNAStrategy(runner, config=config)
    times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)
    I_VDD = stats['currents']['vdd']  # Direct branch current from solution
"""

from .adaptive import (
    AdaptiveConfig,
    AdaptiveWhileLoopStrategy,
    # Shared LTE functions
    compute_lte_timestep_jax,
    predict_voltage_jax,
)
from .base import TransientSetup, TransientStrategy
from .full_mna import FullMNAStrategy

__all__ = [
    "TransientStrategy",
    "TransientSetup",
    "FullMNAStrategy",
    "AdaptiveConfig",
    # Shared LTE functions
    "compute_lte_timestep_jax",
    "predict_voltage_jax",
]
