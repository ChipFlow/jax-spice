#!/usr/bin/env python3
"""Test Full MNA with ring oscillator for longer simulation.

This runs the full MNA strategy for enough time to see oscillations
and compares with the AdaptiveStrategy results.
"""

import os
import sys

# Set JAX platform before importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax.numpy as jnp
import numpy as np

from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.transient import FullMNAStrategy, AdaptiveStrategy


def main():
    # Path to ring oscillator benchmark
    ring_sim = 'vendor/VACASK/benchmark/ring/vacask/runme.sim'

    if not os.path.exists(ring_sim):
        print(f"Error: {ring_sim} not found")
        return 1

    print("=" * 70)
    print("Full MNA Ring Oscillator Comparison")
    print("=" * 70)

    # Simulation parameters
    t_stop = 20e-9  # 20ns - should see several oscillations
    dt = 1e-12  # 1ps timestep

    # =====================================================================
    # Run AdaptiveStrategy (baseline)
    # =====================================================================
    print("\n1. Running AdaptiveStrategy (KCL current extraction)...")
    runner1 = CircuitEngine(ring_sim)
    runner1.parse()

    adaptive = AdaptiveStrategy(runner1, use_sparse=False)
    times_a, voltages_a, currents_a, stats_a = adaptive.run(t_stop=t_stop, dt=dt)

    print(f"   Completed: {stats_a['total_timesteps']} steps, {stats_a['wall_time']:.2f}s")

    # Get VDD current
    I_vdd_a = None
    for name, I in currents_a.items():
        if 'vdd' in name.lower():
            I_vdd_a = np.asarray(I)
            print(f"   {name}: min={I_vdd_a.min()*1e3:.3f} mA, "
                  f"max={I_vdd_a.max()*1e3:.3f} mA, "
                  f"mean={I_vdd_a.mean()*1e3:.3f} mA")

    # =====================================================================
    # Run FullMNAStrategy
    # =====================================================================
    print("\n2. Running FullMNAStrategy (explicit branch currents)...")
    runner2 = CircuitEngine(ring_sim)
    runner2.parse()

    full_mna = FullMNAStrategy(runner2, use_sparse=False)
    times_m, voltages_m, stats_m = full_mna.run(t_stop=t_stop, dt=dt)

    print(f"   Completed: {stats_m['total_timesteps']} steps, {stats_m['wall_time']:.2f}s")
    print(f"   NR iterations: {stats_m['total_nr_iterations']}, "
          f"non-converged: {stats_m['non_converged_count']}")

    # Get VDD current from stats
    I_vdd_m = None
    if 'currents' in stats_m:
        currents_m = stats_m['currents']
        for name, I in currents_m.items():
            I_arr = np.asarray(I)
            I_vdd_m = I_arr
            print(f"   {name}: min={I_arr.min()*1e3:.3f} mA, "
                  f"max={I_arr.max()*1e3:.3f} mA, "
                  f"mean={I_arr.mean()*1e3:.3f} mA")

    # =====================================================================
    # Compare results
    # =====================================================================
    print("\n3. Comparing results...")

    # Check if voltage waveforms match
    t_a = np.asarray(times_a)
    t_m = np.asarray(times_m)

    # Find common node
    common_node = None
    for name in voltages_a.keys():
        if name in voltages_m:
            common_node = name
            break

    if common_node:
        v_a = np.asarray(voltages_a[common_node])
        v_m = np.asarray(voltages_m[common_node])

        # Interpolate to common time base
        n_points = min(len(t_a), len(t_m))
        t_common = np.linspace(0, float(t_stop), n_points)

        v_a_interp = np.interp(t_common, t_a[:len(v_a)], v_a)
        v_m_interp = np.interp(t_common, t_m[:len(v_m)], v_m)

        rms_diff = np.sqrt(np.mean((v_a_interp - v_m_interp)**2))
        print(f"   V({common_node}) RMS difference: {rms_diff*1e3:.3f} mV")

        # Check voltage range
        print(f"   Adaptive V range: [{v_a.min():.3f}, {v_a.max():.3f}] V")
        print(f"   Full MNA V range: [{v_m.min():.3f}, {v_m.max():.3f}] V")

    # Compare currents
    if I_vdd_a is not None and I_vdd_m is not None:
        print("\n   Current comparison:")
        print(f"   Adaptive  - mean: {I_vdd_a.mean()*1e3:.3f} mA, "
              f"peak-to-peak: {(I_vdd_a.max()-I_vdd_a.min())*1e3:.3f} mA")
        print(f"   Full MNA  - mean: {I_vdd_m.mean()*1e3:.3f} mA, "
              f"peak-to-peak: {(I_vdd_m.max()-I_vdd_m.min())*1e3:.3f} mA")

        # Compute dI/dt
        if len(t_a) > 1 and len(t_m) > 1:
            dt_a = np.diff(t_a[:len(I_vdd_a)])
            dI_a = np.diff(I_vdd_a)
            dIdt_a = dI_a / dt_a

            dt_m = np.diff(t_m[:len(I_vdd_m)])
            dI_m = np.diff(I_vdd_m)
            dIdt_m = dI_m / dt_m

            print(f"\n   dI/dt comparison:")
            print(f"   Adaptive  - max|dI/dt|: {np.max(np.abs(dIdt_a))*1e-6:.2f} mA/ns")
            print(f"   Full MNA  - max|dI/dt|: {np.max(np.abs(dIdt_m))*1e-6:.2f} mA/ns")

    print("\n" + "=" * 70)
    print("Test complete")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
