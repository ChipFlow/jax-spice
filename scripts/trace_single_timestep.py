#!/usr/bin/env python3
"""Trace a single timestep during the pulse to see if isource affects the solution."""

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
    print("Tracing Single Timestep During Pulse")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run a minimal transient to get setup
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    setup = engine._transient_setup_cache
    device_arrays = engine._device_arrays
    n_external = engine.num_nodes
    n_unknowns = n_external - 1

    # Get DC voltage as initial condition
    V_dc = jnp.zeros(n_external, dtype=jnp.float64)
    for name, voltages in result.voltages.items():
        if name in engine.node_names:
            idx = engine.node_names[name]
            V_dc = V_dc.at[idx].set(float(voltages[0]))

    print(f"\nInitial DC voltage:")
    print(f"  V(1) = {float(V_dc[1]):.6f}V")

    # Get build_system
    build_system_fn, _ = engine._make_gpu_resident_build_system_fn(
        source_device_data=setup['source_device_data'],
        vmapped_fns=setup.get('vmapped_fns', {}),
        static_inputs_cache=setup.get('static_inputs_cache', {}),
        n_unknowns=n_unknowns,
        use_dense=True,
    )

    # Get NR solver
    from jax_spice.analysis.solver_factories import make_dense_solver
    n_nodes = n_external  # Total nodes including ground
    nr_solve = make_dense_solver(
        build_system_fn,
        n_nodes=n_nodes,
        max_iterations=100,
    )

    vsource_vals = jnp.array([1.2])
    Q_prev = jnp.zeros(n_unknowns)
    dt = 0.1e-9
    inv_dt = 1.0 / dt

    # Test timestep DURING pulse (t=2.0ns, should have 10µA)
    print("\n" + "="*80)
    print("Timestep at t=2.0ns (pulse active, 10µA)")
    print("="*80)

    isource_10ua = jnp.array([10e-6])
    V_new_10ua, iters_10ua, conv_10ua, max_f_10ua, Q_10ua = nr_solve(
        V_dc, vsource_vals, isource_10ua, Q_prev, inv_dt, device_arrays
    )

    print(f"NR solver result:")
    print(f"  Converged: {bool(conv_10ua)}")
    print(f"  Iterations: {int(iters_10ua)}")
    print(f"  Max |f|: {float(max_f_10ua):.6e} A")
    print(f"  V(1) = {float(V_new_10ua[1]):.6f}V")

    # Test timestep WITHOUT pulse (t=0.0ns, should have 0µA)
    print("\n" + "="*80)
    print("Timestep at t=0.0ns (no pulse, 0µA)")
    print("="*80)

    isource_0ua = jnp.array([0.0])
    V_new_0ua, iters_0ua, conv_0ua, max_f_0ua, Q_0ua = nr_solve(
        V_dc, vsource_vals, isource_0ua, Q_prev, inv_dt, device_arrays
    )

    print(f"NR solver result:")
    print(f"  Converged: {bool(conv_0ua)}")
    print(f"  Iterations: {int(iters_0ua)}")
    print(f"  Max |f|: {float(max_f_0ua):.6e} A")
    print(f"  V(1) = {float(V_new_0ua[1]):.6f}V")

    # Compare results
    print("\n" + "="*80)
    print("Comparison")
    print("="*80)

    delta_V = float(V_new_10ua[1] - V_new_0ua[1])
    print(f"  ΔV(1) = {delta_V:.6f}V ({delta_V*1e3:.3f}mV)")

    expected_dV = 10e-6 / 1.003e-6  # I / G from Jacobian analysis
    print(f"  Expected ΔV ≈ {expected_dV:.6f}V (from Jacobian)")

    if abs(delta_V) < 0.001:
        print(f"\n⚠️  WARNING: Voltage barely changes despite 10µA difference!")
        print(f"   The solver is NOT responding to isource!")
    else:
        print(f"\n✓ Solver responds to isource")

    # Directly check residual change
    J_0, f_0, Q_0 = build_system_fn(V_dc, vsource_vals, isource_0ua, Q_prev, 0.0, device_arrays)
    J_10, f_10, Q_10 = build_system_fn(V_dc, vsource_vals, isource_10ua, Q_prev, 0.0, device_arrays)

    print("\n" + "="*80)
    print("Direct Residual Check (at same voltage)")
    print("="*80)
    print(f"  f[0] with 0µA: {float(f_0[0]):.6e} A")
    print(f"  f[0] with 10µA: {float(f_10[0]):.6e} A")
    print(f"  Δf[0]: {float(f_10[0] - f_0[0]):.6e} A (should be +10µA)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
