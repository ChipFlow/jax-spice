#!/usr/bin/env python3
"""Check if the circuit is too stiff (high Jacobian diagonal) to respond to 10µA."""

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

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run a minimal transient
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

    # Get build_system
    build_system_fn, _ = engine._make_gpu_resident_build_system_fn(
        source_device_data=setup['source_device_data'],
        vmapped_fns=setup.get('vmapped_fns', {}),
        static_inputs_cache=setup.get('static_inputs_cache', {}),
        n_unknowns=n_unknowns,
        use_dense=True,
    )

    # Build system at DC with no isource
    vsource_vals = jnp.array([1.2])
    isource_vals = jnp.zeros(1)
    Q_prev = jnp.zeros(n_unknowns)

    J, f, Q = build_system_fn(V_dc, vsource_vals, isource_vals, Q_prev, 0.0, device_arrays)

    print("="*80)
    print("Jacobian Analysis at DC Operating Point")
    print("="*80)

    # Check Jacobian diagonal
    J_diag = jnp.diag(J)

    print(f"\nJacobian diagonal (conductance to ground):")
    for i in range(min(10, len(J_diag))):
        node_name = next((n for n, idx in engine.node_names.items() if idx == i+1), f"node_{i+1}")
        G = float(J_diag[i])
        R_eq = 1/G if G > 0 else float('inf')
        print(f"  J[{i},{i}] ({node_name}): {G:.6e} S  (Req = {R_eq:.3e} Ω)")

    # Estimate voltage change from 10µA
    print(f"\nEstimated voltage change from 10µA current injection:")
    for i in range(min(10, len(J_diag))):
        node_name = next((n for n, idx in engine.node_names.items() if idx == i+1), f"node_{i+1}")
        G = float(J_diag[i])
        if G > 0:
            dV = 10e-6 / G
            print(f"  {node_name}: ΔV ≈ {dV*1e3:.6f} mV  (G={G:.3e} S)")

    # Check off-diagonal coupling
    print(f"\n Jacobian off-diagonal (first few rows):")
    for i in range(min(3, n_unknowns)):
        node_name = next((n for n, idx in engine.node_names.items() if idx == i+1), f"node_{i+1}")
        row = J[i, :]
        max_off_diag = float(jnp.max(jnp.abs(row) * (jnp.arange(len(row)) != i)))
        print(f"  Row {i} ({node_name}): diag={float(J[i,i]):.3e}, max_off_diag={max_off_diag:.3e}")

    # Compare with VACASK
    print(f"\n" + "="*80)
    print("Comparison with VACASK")
    print("="*80)
    print(f"VACASK Jacobian diagonal at V=0 (MOSFETs off): ~1.17e-06 S")
    print(f"JAX-SPICE Jacobian diagonal at node 1: {float(J_diag[0]):.3e} S")

    ratio = float(J_diag[0]) / 1.17e-6
    print(f"Ratio: {ratio:.2f}x")

    if ratio > 100:
        print(f"\n⚠️  WARNING: Jacobian is {ratio:.0f}x stiffer than VACASK!")
        print("   This could prevent the circuit from responding to the pulse.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
