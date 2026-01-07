#!/usr/bin/env python3
"""Trace Q (charge) values during transient simulation."""

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
    print("Tracing Q (Charge) Values During Transient")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run actual transient simulation
    print("\nRunning transient simulation (10ns)...")
    result = engine.run_transient(t_stop=10e-9, dt=0.1e-9)

    # Check if the simulation actually worked
    times_ns = result.times * 1e9
    v1 = result.voltages['1']

    print(f"\nSimulation results:")
    print(f"  Timesteps: {len(result.times)}")
    print(f"  V(1) statistics:")
    print(f"    Initial: {float(v1[0]):.6f}V")
    print(f"    Min: {float(jnp.min(v1)):.6f}V")
    print(f"    Max: {float(jnp.max(v1)):.6f}V")
    print(f"    Final: {float(v1[-1]):.6f}V")
    print(f"    Swing: {float(jnp.max(v1) - jnp.min(v1)):.6f}V")

    # Check stats
    if 'stats' in result.__dict__:
        stats = result.stats
        print(f"\n  Solver statistics:")
        print(f"    Total NR iterations: {stats.get('total_nr_iterations', 'N/A')}")
        avg_iters = stats.get('avg_nr_iterations', None)
        if avg_iters is not None:
            print(f"    Avg NR iterations/step: {avg_iters:.2f}")
        print(f"    Non-converged steps: {stats.get('non_converged_count', 'N/A')}")
        conv_rate = stats.get('convergence_rate', None)
        if conv_rate is not None:
            print(f"    Convergence rate: {conv_rate*100:.1f}%")

    # Now trace Q values by manually stepping through
    print("\n" + "="*80)
    print("Manual Transient Stepping to Trace Q Values")
    print("="*80)

    setup = engine._transient_setup_cache
    device_arrays = engine._device_arrays
    n_external = engine.num_nodes
    n_unknowns = n_external - 1

    # Use the engine's transient strategy to get the NR solver
    from jax_spice.analysis.transient.scan import ScanStrategy
    strategy = ScanStrategy(engine, use_dense=True)
    setup_obj = strategy.ensure_setup()
    nr_solve = strategy.ensure_solver()

    # Also get build_system for manual checks
    build_system_fn, _ = engine._make_gpu_resident_build_system_fn(
        source_device_data=setup['source_device_data'],
        vmapped_fns=setup.get('vmapped_fns', {}),
        static_inputs_cache=setup.get('static_inputs_cache', {}),
        n_unknowns=n_unknowns,
        use_dense=True,
    )

    # Start from DC operating point
    V = jnp.zeros(n_external, dtype=jnp.float64)
    for name, voltages in result.voltages.items():
        if name in engine.node_names:
            idx = engine.node_names[name]
            V = V.at[idx].set(float(voltages[0]))

    # Get initial Q by evaluating at DC
    vsource_vals = jnp.array([1.2])
    isource_vals = jnp.array([0.0])
    J_dc, f_dc, Q_dc = build_system_fn(V, vsource_vals, isource_vals, jnp.zeros(n_unknowns), 0.0, device_arrays)

    print(f"\nInitial conditions (DC):")
    print(f"  V(1) = {float(V[1]):.6f}V")
    print(f"  Q statistics:")
    print(f"    Min Q: {float(jnp.min(Q_dc)):.6e} C")
    print(f"    Max Q: {float(jnp.max(Q_dc)):.6e} C")
    print(f"    Mean |Q|: {float(jnp.mean(jnp.abs(Q_dc))):.6e} C")
    print(f"    Non-zero Q: {int(jnp.sum(Q_dc != 0.0))}/{len(Q_dc)}")

    # Trace a few timesteps
    dt = 0.1e-9
    inv_dt = 1.0 / dt
    Q_prev = Q_dc

    test_times = [0.0, 1.0, 2.0, 3.0, 5.0]  # ns
    print(f"\n" + "="*80)
    print("Stepping through timesteps:")
    print("="*80)

    for t_ns in test_times:
        t = t_ns * 1e-9

        # Compute isource value at this time
        pulse_dev = next((d for d in engine.devices if d['model'] == 'isource' and d['params'].get('type') == 'pulse'), None)
        if pulse_dev:
            src_fn = engine._get_source_fn_for_device(pulse_dev)
            isource_val = float(src_fn(t))
        else:
            isource_val = 0.0

        isource_vals_t = jnp.array([isource_val])

        print(f"\nt={t_ns:.1f}ns (isource={isource_val*1e6:.1f}µA):")
        print(f"  Q_prev stats BEFORE nr_solve: min={float(jnp.min(Q_prev)):.2e}, max={float(jnp.max(Q_prev)):.2e}, mean={float(jnp.mean(jnp.abs(Q_prev))):.2e}C")

        # Run NR solver
        V_new, iters, converged, max_f, Q_new = nr_solve(
            V, vsource_vals, isource_vals_t, Q_prev, inv_dt, device_arrays
        )

        # Manually recompute Q to verify
        J_check, f_check, Q_check = build_system_fn(V_new, vsource_vals, isource_vals_t, Q_prev, inv_dt, device_arrays)

        print(f"  NR: {int(iters)} iters, converged={bool(converged)}, max|f|={float(max_f):.2e}A")
        print(f"  V(1): {float(V_new[1]):.6f}V")
        print(f"  Q returned from nr_solve: min={float(jnp.min(Q_new)):.2e}, max={float(jnp.max(Q_new)):.2e}C")
        print(f"  Q recomputed manually:    min={float(jnp.min(Q_check)):.2e}, max={float(jnp.max(Q_check)):.2e}C")

        if jnp.max(jnp.abs(Q_new - Q_check)) > 1e-15:
            print(f"  ⚠️  Q from nr_solve differs from manual recomputation by {float(jnp.max(jnp.abs(Q_new - Q_check))):.2e} C!")

        if not converged:
            print(f"  ⚠️  WARNING: NR solver did not converge!")
            # Print first few residuals
            J_debug, f_debug, Q_debug = build_system_fn(V_new, vsource_vals, isource_vals_t, Q_prev, 0.0, device_arrays)
            print(f"  Residuals: f[0]={float(f_debug[0]):.2e}, f[1]={float(f_debug[1]):.2e}, f[2]={float(f_debug[2]):.2e}")

        # Update for next timestep
        V = V_new
        Q_prev = Q_new

    return 0


if __name__ == "__main__":
    sys.exit(main())
