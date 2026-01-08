#!/usr/bin/env python3
"""Check if 0.6V and 0.66V are stable DC equilibria by examining Jacobian eigenvalues."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine


def check_stability(V, label, build_system_fn, vsource_vals, isource_vals, device_arrays, n_unknowns):
    """Check if a voltage is a stable equilibrium."""
    print(f"\n{'='*80}")
    print(f"Checking {label}: V(1) = {float(V[1]):.6f}V")
    print(f"{'='*80}")

    Q_prev = jnp.zeros(n_unknowns)
    J, f, Q = build_system_fn(V, vsource_vals, isource_vals, Q_prev, 0.0, device_arrays)

    # Check residual
    max_residual = float(jnp.max(jnp.abs(f)))
    print(f"  Max |f| = {max_residual:.6e} A")

    if max_residual < 1e-9:
        print(f"  ✓ Near-valid DC solution (residual < 1e-9)")
    elif max_residual < 1e-6:
        print(f"  ~ Approximate DC solution (residual < 1e-6)")
    else:
        print(f"  ✗ NOT a DC solution (residual > 1e-6)")
        return

    # Compute eigenvalues to check stability
    # For DC stability, we want eigenvalues of -J (since dV/dt = -J^-1 * f)
    # Stable if all eigenvalues of J have positive real parts
    print(f"\n  Computing eigenvalues of {n_unknowns}x{n_unknowns} Jacobian...")
    try:
        eigenvalues = np.linalg.eigvals(np.array(J))

        # Separate real and complex eigenvalues
        real_eigs = eigenvalues[np.abs(eigenvalues.imag) < 1e-10].real
        complex_eigs = eigenvalues[np.abs(eigenvalues.imag) >= 1e-10]

        print(f"  Real eigenvalues: {len(real_eigs)}")
        print(f"  Complex eigenvalues: {len(complex_eigs)}")

        # Check for stability (all eigenvalues should have positive real parts for stable DC)
        min_real = float(np.min(eigenvalues.real))
        max_real = float(np.max(eigenvalues.real))

        print(f"\n  Eigenvalue real parts:")
        print(f"    Min: {min_real:.6e}")
        print(f"    Max: {max_real:.6e}")

        # For DC stability, we need Re(λ) > 0 for all eigenvalues of G (conductance)
        # If any λ has Re(λ) ≤ 0, the circuit can drift away from this point
        unstable_count = np.sum(eigenvalues.real <= 0)

        if unstable_count > 0:
            print(f"\n  ⚠️  UNSTABLE: {unstable_count} eigenvalues with Re(λ) ≤ 0")
            print(f"    This is an unstable equilibrium - circuit will drift away")

            # Print the unstable eigenvalues
            unstable_eigs = eigenvalues[eigenvalues.real <= 0]
            print(f"\n    Unstable eigenvalues:")
            for i, eig in enumerate(unstable_eigs[:5]):  # Show first 5
                print(f"      λ_{i} = {eig.real:.6e} + {eig.imag:.6e}j")
        else:
            print(f"\n  ✓ STABLE: All eigenvalues have Re(λ) > 0")
            print(f"    This is a stable DC equilibrium")

    except Exception as e:
        print(f"  Error computing eigenvalues: {e}")


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("DC Stability Analysis: 0.60V vs 0.66V")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run minimal transient to get setup
    result = engine.run_transient(t_stop=0.1e-9, dt=0.1e-9)

    setup = engine._transient_setup_cache
    device_arrays = engine._device_arrays
    n_external = engine.num_nodes
    n_unknowns = n_external - 1

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

    # Test 1: Our DC point (0.600V)
    V_jax = jnp.zeros(n_external, dtype=jnp.float64)
    for name, voltages in result.voltages.items():
        if name in engine.node_names:
            idx = engine.node_names[name]
            V_jax = V_jax.at[idx].set(float(voltages[0]))

    check_stability(V_jax, "JAX-SPICE DC (0.600V)", build_system_fn,
                   vsource_vals, isource_vals, device_arrays, n_unknowns)

    # Test 2: VACASK's DC point (0.661V)
    V_vacask = jnp.full(n_external, 0.660597, dtype=jnp.float64)
    V_vacask = V_vacask.at[0].set(0.0)  # Ground
    V_vacask = V_vacask.at[engine.node_names['vdd']].set(1.2)  # VDD

    check_stability(V_vacask, "VACASK DC (0.661V)", build_system_fn,
                   vsource_vals, isource_vals, device_arrays, n_unknowns)

    return 0


if __name__ == "__main__":
    sys.exit(main())
