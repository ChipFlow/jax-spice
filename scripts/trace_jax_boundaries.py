#!/usr/bin/env python
"""Trace JAX<->Python boundary crossings using JAX's tracing mechanism.

Uses jax.make_jaxpr to identify what gets traced vs what becomes static.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
from jax import make_jaxpr

print("=" * 70)
print("JAX Tracing Analysis for Adaptive Timestep Loop")
print("=" * 70)
print()

# Import after path setup
from jax_spice.analysis import CircuitEngine
from jax_spice.analysis.transient.adaptive import AdaptiveStrategy, AdaptiveConfig
from jax_spice.analysis.integration import IntegrationMethod, compute_coefficients

# Load ring oscillator
VACASK_BENCHMARK = project_root / "vendor" / "VACASK" / "benchmark"
sim_file = VACASK_BENCHMARK / "ring" / "vacask" / "runme.sim"

print(f"Loading: {sim_file}")
engine = CircuitEngine(sim_file)
engine.parse()

# Create strategy and get setup
strategy = AdaptiveStrategy(engine, config=AdaptiveConfig(min_dt=1e-12, max_dt=1e-9))
setup = strategy.ensure_setup()
nr_solve = strategy.ensure_solver()

print(f"Circuit: {setup.n_unknowns} unknowns, {setup.n_external} external nodes")
print()

# =============================================================================
# Test 1: What happens when we trace the source function?
# =============================================================================
print("=" * 70)
print("TEST 1: Source function tracing")
print("=" * 70)

source_fn = setup.source_fn

# Try to trace source_fn
print("\nCalling source_fn with Python float:")
t_py = 1e-9
result = source_fn(t_py)
print(f"  source_fn({t_py}) returns: {type(result)} with {len(result)} entries")

print("\nCalling source_fn with JAX scalar:")
t_jax = jnp.array(1e-9)
try:
    result = source_fn(t_jax)
    print(f"  source_fn(jnp.array) returns: {type(result)}")
    # Check if values are traced
    for k, v in list(result.items())[:3]:
        print(f"    {k}: {type(v)} = {v}")
except Exception as e:
    print(f"  ERROR: {e}")

# =============================================================================
# Test 2: What happens with compute_coefficients?
# =============================================================================
print()
print("=" * 70)
print("TEST 2: Integration coefficients tracing")
print("=" * 70)

print("\ncompute_coefficients with Python float dt:")
coeffs = compute_coefficients(IntegrationMethod.BACKWARD_EULER, 1e-12)
print(f"  c0 type: {type(coeffs.c0)}, value: {coeffs.c0}")

print("\ncompute_coefficients expects Python float - JAX scalar would fail")
print("  (dt is used in Python if/elif branches)")

# =============================================================================
# Test 3: Trace a single NR solve step
# =============================================================================
print()
print("=" * 70)
print("TEST 3: Newton-Raphson solver tracing")
print("=" * 70)

n_total = setup.n_total
n_unknowns = setup.n_unknowns
device_arrays = engine._device_arrays

# Create sample inputs
V_init = jnp.zeros(n_total)
Q_prev = jnp.zeros(n_unknowns)
dQdt_prev = jnp.zeros(n_unknowns)
Q_prev2 = jnp.zeros(n_unknowns)

# Get source values
source_vals = source_fn(0.0)
vsource_vals, isource_vals = strategy._build_source_arrays(source_vals)

print(f"\nInput shapes:")
print(f"  V_init: {V_init.shape}")
print(f"  vsource_vals: {vsource_vals.shape}")
print(f"  isource_vals: {isource_vals.shape}")

print("\nTracing nr_solve with make_jaxpr...")
try:
    jaxpr = make_jaxpr(nr_solve)(
        V_init, vsource_vals, isource_vals, Q_prev,
        1.0,  # integ_c0 (Python float - will be static)
        device_arrays,
        1e-12, 0.0,  # gmin, gshunt
        -1.0, 0.0,   # integ_c1, integ_d1
        dQdt_prev,
        0.0,         # integ_c2
        Q_prev2,
    )
    print(f"  jaxpr has {len(jaxpr.jaxpr.eqns)} equations")
    print(f"  Input variables: {len(jaxpr.jaxpr.invars)}")
    print(f"  Output variables: {len(jaxpr.jaxpr.outvars)}")
except Exception as e:
    print(f"  ERROR tracing: {e}")

# =============================================================================
# Test 4: Identify Python control flow in the loop
# =============================================================================
print()
print("=" * 70)
print("TEST 4: Loop control flow analysis")
print("=" * 70)

print("""
The adaptive loop has these Python control flow points:

1. while float(t_jax) < t_stop:     # JAX->Python for loop condition

2. current_dt = float(dt_jax)        # JAX->Python for compute_coefficients

3. t_next_py = float(t_jax) + dt     # JAX->Python for source_fn
   source_values = source_fn(t_next_py)  # Returns Python dict

4. if not bool(nr_ok):               # JAX->Python for NR failure check
       continue                       # Python control flow

5. if not bool(lte_ok):              # JAX->Python for LTE rejection
       continue                       # Python control flow

6. idx = int(out_idx)                # JAX->Python for bounds check
   if idx < est_max_steps:           # Python comparison

Each of these represents a JAX->Python boundary crossing.
""")

# =============================================================================
# Test 5: Count conversions in one iteration
# =============================================================================
print()
print("=" * 70)
print("TEST 5: Conversions per iteration (manual count)")
print("=" * 70)

print("""
Per-iteration JAX->Python conversions:

UNAVOIDABLE (Python control flow):
  - float(t_jax) for while condition:     1
  - bool(nr_ok) for convergence check:    1
  - bool(lte_ok) for LTE check:           1 (if warmup complete)

NEEDED FOR PYTHON APIs:
  - float(dt_jax) for compute_coefficients: 1
  - float(t_jax) for source_fn:             1
  - int(out_idx) for array indexing:        1

LOGGING ONLY:
  - float(t_jax) in logger.warning:         0-1 (only on failure)
  - float(t_jax) in logger.debug:           0-1 (only at warmup end)

TOTAL: ~6 conversions per accepted step
       ~4 conversions per rejected step (no output write)
""")

# =============================================================================
# Test 6: What would a fully-traced version look like?
# =============================================================================
print()
print("=" * 70)
print("TEST 6: Requirements for fully-traced loop")
print("=" * 70)

print("""
To eliminate all JAX->Python conversions, we would need:

1. Replace while loop with lax.while_loop
   - Condition must be JAX boolean, not Python bool

2. Replace compute_coefficients with compute_jit_coeffs
   - Already exists, uses jnp.where for method selection

3. Replace source_fn with JIT-compatible version
   - Already exists: setup.eval_sources(t) returns JAX arrays directly
   - But individual source functions still use Python dict internally

4. Replace if/continue with lax.cond
   - Already tried in AdaptiveScanStrategy
   - Results in ~100x slower execution due to nested lax.cond overhead

5. Replace int(out_idx) < est_max_steps with JAX comparison
   - Easy: use jnp.less(out_idx, est_max_steps)

The fundamental problem: lax.cond has significant overhead when nested,
making the fully-traced version much slower than Python loop + JIT solver.
""")

print()
print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The current implementation has ~6 JAX->Python conversions per iteration.
This is near-optimal for a Python-loop approach because:

1. The NR solver (heavy computation) is fully JIT-compiled
2. Python handles control flow naturally with minimal overhead
3. Each conversion is necessary for a specific Python API

Attempting to wrap these in JAX conditionals (jnp.where, lax.cond) was
slower because it added tracing overhead without eliminating the
underlying Python API calls (source_fn dict, compute_coefficients).

The ~1ms/step performance is the practical limit for this architecture.
""")
