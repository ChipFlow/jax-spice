#!/usr/bin/env python3
"""Run ring benchmark and plot V/I for all nodes.

Usage:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python scripts/plot_ring_vi.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from jax_spice.analysis import CircuitEngine

def main():
    # Run simulation
    base = Path(__file__).parent.parent
    sim_path = base / "vendor/VACASK/benchmark/ring/vacask/runme.sim"

    print(f"Loading circuit from {sim_path}")
    engine = CircuitEngine(str(sim_path))
    engine.parse()

    print("Running transient simulation...")
    result = engine.run_transient(t_stop=1e-8, dt=1e-12)

    # Extract data
    time = np.array(result.times)
    voltages = {name: np.array(v) for name, v in result.voltages.items()}
    currents = {name: np.array(i) for name, i in result.currents.items()}

    print(f"Simulation: {len(time)} time points, {time[0]:.2e}s to {time[-1]:.2e}s")
    print(f"Voltage nodes: {list(voltages.keys())}")
    print(f"Current probes: {list(currents.keys())}")

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot voltages - separate ring nodes from internal nodes
    ax1 = axes[0]
    ring_nodes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    for name in ring_nodes:
        if name in voltages:
            ax1.plot(time * 1e9, voltages[name], label=f'Node {name}', linewidth=1.0)

    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Ring Oscillator Output Node Voltages')
    ax1.legend(loc='upper right', ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.3)

    # Plot currents
    ax2 = axes[1]
    for name, i in sorted(currents.items()):
        ax2.plot(time * 1e9, i * 1e3, label=name, linewidth=1.0)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Current (mA)')
    ax2.set_title('Source Currents')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = base / 'ring_vi_plot.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")

    # Print summary statistics
    print("\n=== Ring Node Voltage Summary ===")
    for name in ring_nodes:
        if name in voltages:
            v = voltages[name]
            print(f"  Node {name}: min={v.min():.4f}V, max={v.max():.4f}V, swing={v.max()-v.min():.4f}V")

    print("\n=== Current Summary ===")
    for name, i in sorted(currents.items()):
        print(f"  {name:10s}: min={i.min()*1e3:.4f}mA, max={i.max()*1e3:.4f}mA")

    # Check if oscillating
    swings = [voltages[n].max() - voltages[n].min() for n in ring_nodes if n in voltages]
    if max(swings) < 0.1:
        print("\n⚠️  WARNING: Ring oscillator is NOT oscillating (voltage swing < 100mV)")
    else:
        print(f"\n✓ Ring oscillator is oscillating (max swing: {max(swings):.3f}V)")

if __name__ == '__main__':
    main()
