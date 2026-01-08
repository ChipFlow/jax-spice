#!/usr/bin/env python3
"""Trace DC solver to see if NR is being called."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("Tracing DC Solver")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Monkey-patch the NR solver to trace calls
    original_nr = None
    call_count = [0]

    def traced_nr_solve(V_init, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays, *args, **kwargs):
        call_count[0] += 1
        print(f"\n[NR CALL {call_count[0]}] inv_dt={float(inv_dt):.2e}")
        print(f"  V_init[1] = {float(V_init[1]):.6f}V")
        result = original_nr(V_init, vsource_vals, isource_vals, Q_prev, inv_dt, device_arrays, *args, **kwargs)
        V_final, iters, converged, max_f, Q = result
        print(f"  V_final[1] = {float(V_final[1]):.6f}V")
        print(f"  converged = {bool(converged)}, iters = {int(iters)}, max_f = {float(max_f):.2e}")
        return result

    # Run minimal transient to trigger DC
    print("\nRunning transient (will trigger DC computation)...\n")

    # Get to the point where nr_solve is created
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    print(f"\n{'='*80}")
    print(f"DC SOLVER SUMMARY")
    print(f"{'='*80}")
    print(f"  NR solver called: {call_count[0]} times")
    print(f"  Final DC: V(1) = {float(result.voltages['1'][0]):.6f}V")
    print(f"  Expected: V(1) = 0.660597V (VACASK)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
