#!/usr/bin/env python3
"""Run ring oscillator for longer time to see if it eventually starts oscillating."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("Ring Oscillator - Long Transient (100ns)")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Run longer transient to see if oscillation eventually starts
    print("\nRunning transient: 100ns, dt=0.05ns...")
    result = engine.run_transient(t_stop=100e-9, dt=0.05e-9)

    times_ns = result.times * 1e9
    v1 = result.voltages['1']

    print(f"\nResults:")
    print(f"  Time range: {float(times_ns[0]):.2f}ns to {float(times_ns[-1]):.2f}ns")
    print(f"  Timesteps: {len(times_ns)}")
    print(f"  V(1) range: {float(jnp.min(v1)):.6f}V to {float(jnp.max(v1)):.6f}V")
    print(f"  V(1) swing: {float(jnp.max(v1) - jnp.min(v1))*1000:.3f}mV")
    print(f"  DC value: {float(v1[0]):.6f}V")

    # Check if any oscillation happened
    swing_threshold = 0.01  # 10mV
    if float(jnp.max(v1) - jnp.min(v1)) > swing_threshold:
        print(f"\n✓ Oscillation detected (swing > {swing_threshold*1000}mV)!")

        # Find oscillation period
        # Simple peak detection
        mean_v = float(jnp.mean(v1))
        crossings = jnp.diff(jnp.sign(v1 - mean_v)) != 0
        crossing_times = times_ns[1:][crossings]

        if len(crossing_times) > 2:
            periods = jnp.diff(crossing_times)[::2]  # Every other crossing for full period
            if len(periods) > 0:
                avg_period = float(jnp.mean(periods))
                freq_mhz = 1000 / avg_period  # Period in ns -> freq in MHz
                print(f"  Estimated period: {avg_period:.2f}ns")
                print(f"  Estimated frequency: {freq_mhz:.1f}MHz")
    else:
        print(f"\n✗ No oscillation (swing < {swing_threshold*1000}mV)")
        print(f"  Circuit stuck at DC point: {float(jnp.mean(v1)):.6f}V")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(times_ns, v1, '-', linewidth=0.5)
    plt.xlabel('Time (ns)')
    plt.ylabel('V(1) (V)')
    plt.title('Ring Oscillator V(1) - 100ns simulation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(__file__).parent / 'ring_long_transient.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
