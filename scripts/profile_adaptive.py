#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["jax", "matplotlib", "numpy", "jaxtyping"]
# ///
"""Profile adaptive timestep strategy to identify bottlenecks."""

import sys
sys.path.insert(0, '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice')

import cProfile
import pstats
import time
import io

import jax
jax.config.update('jax_platforms', 'cpu')

from pathlib import Path
from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.transient import AdaptiveStrategy, AdaptiveConfig

# Find ring benchmark
VACASK_PATH = Path('/Users/roberttaylor/Code/ChipFlow/reference/jax-spice/vendor/VACASK')
ring_sim = VACASK_PATH / 'benchmark' / 'ring' / 'vacask' / 'runme.sim'

if not ring_sim.exists():
    print(f"Ring benchmark not found at {ring_sim}")
    sys.exit(1)

print("Loading ring oscillator benchmark...")
engine = CircuitEngine(str(ring_sim))
engine.parse()

# Simulation parameters - shorter for profiling
t_stop = 5e-9  # 5ns
dt_init = 1e-12  # 1ps

# Configure adaptive
config = AdaptiveConfig(
    min_dt=1e-14,
    max_dt=1e-10,
    reltol=1e-3,
    abstol=1e-6,
    grow_factor=2.0,
)

strategy = AdaptiveStrategy(engine, use_sparse=False, config=config)

print("=== Warmup run ===")
# Warmup - JIT compile
t0 = time.perf_counter()
times, voltages, currents, stats = strategy.run(t_stop=t_stop, dt=dt_init)
warmup_time = time.perf_counter() - t0
print(f"Warmup: {warmup_time:.2f}s, {stats['total_timesteps']} steps")

print("\n=== Profiled run ===")
# Profile the actual run
profiler = cProfile.Profile()
profiler.enable()

times, voltages, currents, stats = strategy.run(t_stop=t_stop, dt=dt_init)

profiler.disable()

print(f"Steps: {stats['total_timesteps']}, Wall time: {stats['wall_time']:.3f}s")
print(f"Time per step: {stats['time_per_step_ms']:.2f}ms")

# Print profile stats
print("\n=== Profile Results (top 30 by cumulative time) ===")
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())

print("\n=== Profile Results (top 20 by total time in function) ===")
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
ps.print_stats(20)
print(s.getvalue())

print("\n=== Profile Results (calls to key functions) ===")
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s)
# Look for specific functions
ps.print_stats('compute_predictor_coeffs', 'scale_coefficients', 'source_fn', '_build_source_arrays', 'nr_solve')
print(s.getvalue())
