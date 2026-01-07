#!/usr/bin/env python3
"""Simple Q debugging - just call build_system and check Q values."""

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
    print("Simple Q Value Debugging")
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

    # Test 1: DC analysis (inv_dt = 0)
    print("\n" + "="*80)
    print("Test 1: DC Analysis (inv_dt=0, Q_prev=zeros)")
    print("="*80)

    Q_prev_zeros = jnp.zeros(n_unknowns)
    J_dc, f_dc, Q_dc = build_system_fn(V_dc, vsource_vals, isource_vals, Q_prev_zeros, 0.0, device_arrays)

    print(f"Q statistics:")
    print(f"  Min: {float(jnp.min(Q_dc)):.6e} C")
    print(f"  Max: {float(jnp.max(Q_dc)):.6e} C")
    print(f"  Mean |Q|: {float(jnp.mean(jnp.abs(Q_dc))):.6e} C")
    print(f"  Range expected: ~1e-13 to 1e-12 C (femto/pico-Coulombs)")

    # Test 2: Transient with small inv_dt and Q_prev=zeros
    print("\n" + "="*80)
    print("Test 2: Transient with inv_dt=1e10 (dt=0.1ns), Q_prev=zeros")
    print("="*80)

    inv_dt = 1e10  # dt = 0.1ns
    J_t, f_t, Q_t = build_system_fn(V_dc, vsource_vals, isource_vals, Q_prev_zeros, inv_dt, device_arrays)

    print(f"Q statistics:")
    print(f"  Min: {float(jnp.min(Q_t)):.6e} C")
    print(f"  Max: {float(jnp.max(Q_t)):.6e} C")
    print(f"  Mean |Q|: {float(jnp.mean(jnp.abs(Q_t))):.6e} C")

    # Test 3: Transient with Q_prev=Q_dc (the DC charges)
    print("\n" + "="*80)
    print("Test 3: Transient with inv_dt=1e10 (dt=0.1ns), Q_prev=Q_dc")
    print("="*80)

    J_t2, f_t2, Q_t2 = build_system_fn(V_dc, vsource_vals, isource_vals, Q_dc, inv_dt, device_arrays)

    print(f"Q statistics:")
    print(f"  Min: {float(jnp.min(Q_t2)):.6e} C")
    print(f"  Max: {float(jnp.max(Q_t2)):.6e} C")
    print(f"  Mean |Q|: {float(jnp.mean(jnp.abs(Q_t2))):.6e} C")

    # Compare with DC
    delta_Q = Q_t2 - Q_dc
    print(f"\nΔQ (transient - DC):")
    print(f"  Min: {float(jnp.min(delta_Q)):.6e} C")
    print(f"  Max: {float(jnp.max(delta_Q)):.6e} C")
    print(f"  Mean |ΔQ|: {float(jnp.mean(jnp.abs(delta_Q))):.6e} C")

    if jnp.max(jnp.abs(delta_Q)) > 1e-10:
        print(f"\n⚠️  WARNING: Q changed by > 1e-10 C just from inv_dt change!")
        print(f"   This suggests analysis_type or inv_dt affects device charge computation.")
    else:
        print(f"\n✓ Q values are consistent between DC and transient calls.")

    # Check residual contribution
    print(f"\nResidual contribution from charges:")
    print(f"  inv_dt * Q_t = {inv_dt} * {float(jnp.mean(jnp.abs(Q_t))):.2e} = {float(inv_dt * jnp.mean(jnp.abs(Q_t))):.2e} A")
    print(f"  This should be small compared to ~1e-12 A tolerance")

    return 0


if __name__ == "__main__":
    sys.exit(main())
