#!/usr/bin/env python3
"""Check if pulse repeats during simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
from jax_spice.analysis.engine import CircuitEngine


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Get pulse device
    pulse_dev = next((d for d in engine.devices if d['model'] == 'isource' and d['params'].get('type') == 'pulse'), None)

    print("Pulse parameters:")
    for k, v in pulse_dev['params'].items():
        print(f"  {k}: {v}")

    source_fn = engine._get_source_fn_for_device(pulse_dev)

    # Check current values over 10ns
    times_ns = jnp.linspace(0, 10, 1001)
    currents_ua = jnp.array([source_fn(t*1e-9) * 1e6 for t in times_ns])

    print(f"\nCurrent statistics (0-10ns):")
    print(f"  Min: {float(jnp.min(currents_ua)):.6f} µA")
    print(f"  Max: {float(jnp.max(currents_ua)):.6f} µA")
    print(f"  Mean: {float(jnp.mean(currents_ua)):.6f} µA")

    # Count how much time is spent at each level
    at_zero = jnp.sum(currents_ua < 0.1)
    at_full = jnp.sum(currents_ua > 9.9)
    ramping = len(currents_ua) - at_zero - at_full

    print(f"\nTime distribution:")
    print(f"  At 0µA: {at_zero/len(currents_ua)*100:.1f}%")
    print(f"  Ramping: {ramping/len(currents_ua)*100:.1f}%")
    print(f"  At 10µA: {at_full/len(currents_ua)*100:.1f}%")

    # Find when pulse goes back to zero
    zero_after_pulse = times_ns[jnp.where((times_ns > 2) & (currents_ua < 0.1))[0]]
    if len(zero_after_pulse) > 0:
        first_zero = float(zero_after_pulse[0])
        print(f"\nPulse returns to 0µA at: {first_zero:.3f}ns")
    else:
        print(f"\n⚠️  Pulse NEVER returns to 0µA in first 10ns!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
