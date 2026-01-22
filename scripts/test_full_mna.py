#!/usr/bin/env python3
"""Test script for Full MNA implementation.

This script tests the FullMNAStrategy with the ring oscillator benchmark
and compares the results with the standard KCL-based current extraction.
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
        print("Run from jax-spice root directory")
        return 1

    print("=" * 60)
    print("Full MNA Implementation Test")
    print("=" * 60)

    # Parse circuit
    print("\n1. Parsing circuit...")
    runner = CircuitEngine(ring_sim)
    runner.parse()
    print(f"   Nodes: {len(runner.node_names)}")
    print(f"   Devices: {len(runner.devices)}")

    # Count voltage sources
    n_vsources = sum(1 for d in runner.devices if d.get('model') == 'vsource')
    print(f"   Voltage sources: {n_vsources}")

    # Run simulation parameters
    t_stop = 10e-9  # 10ns - enough for a few oscillations
    dt = 0.5e-12  # 0.5ps timestep

    # Test 1: Run with standard AdaptiveStrategy (KCL current extraction)
    print("\n2. Running AdaptiveStrategy (KCL current extraction)...")
    try:
        adaptive_strategy = AdaptiveStrategy(runner, use_sparse=False)
        times_adaptive, voltages_adaptive, currents_adaptive, stats_adaptive = adaptive_strategy.run(
            t_stop=t_stop, dt=dt
        )
        print(f"   Completed: {stats_adaptive['total_timesteps']} steps, "
              f"{stats_adaptive['wall_time']:.2f}s")

        # Get current from currents dict
        if currents_adaptive:
            for name, I in currents_adaptive.items():
                I_arr = np.asarray(I)
                print(f"   {name}: min={I_arr.min()*1e3:.3f} mA, "
                      f"max={I_arr.max()*1e3:.3f} mA, "
                      f"mean={I_arr.mean()*1e3:.3f} mA")
        else:
            print("   (No currents in adaptive result)")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 2: Run with FullMNAStrategy (explicit branch currents)
    print("\n3. Running FullMNAStrategy (explicit branch currents)...")
    try:
        # Need fresh runner to avoid cached solver conflicts
        runner2 = CircuitEngine(ring_sim)
        runner2.parse()

        full_mna_strategy = FullMNAStrategy(runner2, use_sparse=False)
        times_mna, voltages_mna, stats_mna = full_mna_strategy.run(
            t_stop=t_stop, dt=dt, max_steps=100  # Limit steps for debugging
        )
        print(f"   Completed: {stats_mna['total_timesteps']} steps, "
              f"{stats_mna['wall_time']:.2f}s")
        print(f"   NR iterations: {stats_mna['total_nr_iterations']}, "
              f"non-converged: {stats_mna['non_converged_count']}")

        # Get current from stats
        if 'currents' in stats_mna:
            currents = stats_mna['currents']
            print(f"   Branch currents available: {list(currents.keys())}")
            for name, I in currents.items():
                I_arr = np.asarray(I)
                print(f"   {name}: min={I_arr.min()*1e3:.3f} mA, "
                      f"max={I_arr.max()*1e3:.3f} mA, "
                      f"mean={I_arr.mean()*1e3:.3f} mA")
        else:
            print("   (No currents in stats)")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Compare voltages
    print("\n4. Comparing results...")
    try:
        # Get a common node for comparison
        common_node = 'inv1'  # First inverter output
        if common_node in voltages_adaptive and common_node in voltages_mna:
            v_adaptive = np.asarray(voltages_adaptive[common_node])
            v_mna = np.asarray(voltages_mna[common_node])

            # Interpolate to common time base
            t_common = np.linspace(0, float(t_stop), min(len(v_adaptive), len(v_mna)))
            t_adaptive = np.asarray(times_adaptive)
            t_mna = np.asarray(times_mna)

            v_adaptive_interp = np.interp(t_common, t_adaptive[:len(v_adaptive)], v_adaptive)
            v_mna_interp = np.interp(t_common, t_mna[:len(v_mna)], v_mna)

            # Compute RMS difference
            rms_diff = np.sqrt(np.mean((v_adaptive_interp - v_mna_interp)**2))
            print(f"   V({common_node}) RMS difference: {rms_diff*1e3:.3f} mV")

            # Check if they're reasonably close
            if rms_diff < 0.01:  # 10mV tolerance
                print("   PASS: Voltages match within tolerance")
            else:
                print("   WARNING: Voltages differ significantly")
        else:
            print(f"   Node '{common_node}' not found in both results")
    except Exception as e:
        print(f"   ERROR comparing: {e}")

    print("\n" + "=" * 60)
    print("Full MNA Test Complete")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
