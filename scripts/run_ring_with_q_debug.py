#!/usr/bin/env python3
"""Run ring oscillator with q_debug=1 to see charge values."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine, DebugOptions


def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("Ring Oscillator with q_debug=1")
    print("="*80)

    engine = CircuitEngine(sim_path)
    engine.parse()

    # Enable q_debug
    engine.debug_options = DebugOptions(q_debug=1)

    print(f"\nRunning transient (10ns) with q_debug=1...")
    result = engine.run_transient(t_stop=10e-9, dt=0.1e-9)

    print(f"\nSimulation completed:")
    print(f"  Timesteps: {len(result.times)}")
    print(f"  V(1) swing: {float(result.voltages['1'].max() - result.voltages['1'].min())*1000:.3f}mV")

    return 0


if __name__ == "__main__":
    sys.exit(main())
