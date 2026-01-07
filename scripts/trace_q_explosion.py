#!/usr/bin/env python3
"""Trace exactly where Q explodes."""

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
    print("Tracing Q Explosion")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run minimal transient to get setup
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    setup = engine._transient_setup_cache
    device_arrays = engine._device_arrays
    n_external = engine.num_nodes
    n_unknowns = n_external - 1

    # Get DC voltage
    V_dc = jnp.zeros(n_external, dtype=jnp.float64)
    for name, voltages in result.voltages.items():
        if name in engine.node_names:
            idx = engine.node_names[name]
            V_dc = V_dc.at[idx].set(float(voltages[0]))

    print(f"\nDC operating point: V(1) = {float(V_dc[1]):.6f}V")

    # Get build_system
    build_system_fn, _ = engine._make_gpu_resident_build_system_fn(
        source_device_data=setup['source_device_data'],
        vmapped_fns=setup.get('vmapped_fns', {}),
        static_inputs_cache=setup.get('static_inputs_cache', {}),
        n_unknowns=n_unknowns,
        use_dense=True,
    )

    vsource_vals = jnp.array([1.2])
    isource_vals = jnp.array([0.0])
    dt = 0.1e-9
    inv_dt = 1.0 / dt

    # Get DC charges
    J_dc, f_dc, Q_dc = build_system_fn(V_dc, vsource_vals, isource_vals, jnp.zeros(n_unknowns), 0.0, device_arrays)

    print(f"\nDC charges (inv_dt=0):")
    print(f"  Min Q: {float(jnp.min(Q_dc)):.6e} C")
    print(f"  Max Q: {float(jnp.max(Q_dc)):.6e} C")
    print(f"  Mean |Q|: {float(jnp.mean(jnp.abs(Q_dc))):.6e} C")

    # Now call build_system with SAME voltage but with inv_dt > 0 and Q_prev = Q_dc
    print(f"\n" + "="*80)
    print("Same voltage, but with inv_dt=1e10 and Q_prev=Q_dc")
    print("="*80)

    J_t, f_t, Q_t = build_system_fn(V_dc, vsource_vals, isource_vals, Q_dc, inv_dt, device_arrays)

    print(f"\nCharges with inv_dt=1e10:")
    print(f"  Min Q: {float(jnp.min(Q_t)):.6e} C")
    print(f"  Max Q: {float(jnp.max(Q_t)):.6e} C")
    print(f"  Mean |Q|: {float(jnp.mean(jnp.abs(Q_t))):.6e} C")

    if jnp.max(jnp.abs(Q_t)) > 1e-10:
        print(f"\n⚠️  BUG: Q exploded to {float(jnp.max(Q_t)):.2e} C!")
    else:
        print(f"\n✓ Q is still reasonable")

    # Check the residual
    print(f"\nResidual statistics:")
    print(f"  DC residual max: {float(jnp.max(jnp.abs(f_dc))):.6e} A")
    print(f"  Transient residual max: {float(jnp.max(jnp.abs(f_t))):.6e} A")

    # Decode the residual contribution
    # f = f_resist + integ_c0 * Q + integ_c1 * Q_prev
    # For backward Euler: integ_c0 = inv_dt, integ_c1 = -inv_dt
    # So: f = f_resist + inv_dt * (Q - Q_prev)

    print(f"\nCharge contribution to residual:")
    print(f"  integ_c0 * Q = {inv_dt:.2e} * {float(jnp.mean(jnp.abs(Q_t))):.2e} = {float(inv_dt * jnp.mean(jnp.abs(Q_t))):.2e} A")
    print(f"  integ_c1 * Q_prev = {-inv_dt:.2e} * {float(jnp.mean(jnp.abs(Q_dc))):.2e} = {float(-inv_dt * jnp.mean(jnp.abs(Q_dc))):.2e} A")

    # The key question: does Q_t differ from Q_dc?
    delta_Q = Q_t - Q_dc
    print(f"\nQ difference (Q_t - Q_dc):")
    print(f"  Min: {float(jnp.min(delta_Q)):.6e} C")
    print(f"  Max: {float(jnp.max(delta_Q)):.6e} C")
    print(f"  Mean |ΔQ|: {float(jnp.mean(jnp.abs(delta_Q))):.6e} C")

    if jnp.max(jnp.abs(delta_Q)) < 1e-15:
        print(f"\n✓ Q is identical for DC and transient calls")
        print(f"   The explosion must happen during NR iteration, not in build_system")
    else:
        print(f"\n⚠️  Q changes between DC and transient!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
