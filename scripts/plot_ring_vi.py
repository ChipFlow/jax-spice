#!/usr/bin/env python3
"""Run ring benchmark and plot V/I for all nodes.

Usage:
    # Basic run with defaults
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run python scripts/plot_ring_vi.py

    # Custom simulation parameters
    uv run python scripts/plot_ring_vi.py --t-stop 1e-7 --dt 1e-11

    # Use sparse solver
    uv run python scripts/plot_ring_vi.py --sparse

    # Disable node collapse (for debugging)
    uv run python scripts/plot_ring_vi.py --no-collapse

    # Use lax.scan for faster simulation
    uv run python scripts/plot_ring_vi.py --use-scan
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from jax_spice.analysis import CircuitEngine
from jax_spice._logging import enable_performance_logging


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ring benchmark and plot V/I for all nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Simulation parameters
    parser.add_argument(
        "--t-stop",
        type=float,
        default=1e-8,
        help="Stop time in seconds (default: 1e-8 = 10ns)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1e-12,
        help="Time step in seconds (default: 1e-12 = 1ps)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000000,
        help="Maximum number of timesteps (default: 1000000)",
    )

    # Solver options
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse solver (default: dense)",
    )
    parser.add_argument(
        "--use-scan",
        action="store_true",
        help="Use lax.scan for faster simulation",
    )
    parser.add_argument(
        "--use-while-loop",
        action="store_true",
        help="Use lax.while_loop (computes sources on-the-fly)",
    )

    # Node collapse control
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="Disable node collapse (keeps all internal nodes)",
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output plot file path (default: ring_vi_plot.png in repo root)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plot (just print summary)",
    )

    # Debugging options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose performance logging",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.15,
        help="Simulation temperature in Kelvin (default: 300.15K = 27C)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        enable_performance_logging()

    # Run simulation
    base = Path(__file__).parent.parent
    sim_path = base / "vendor/VACASK/benchmark/ring/vacask/runme.sim"

    print(f"Loading circuit from {sim_path}")
    engine = CircuitEngine(str(sim_path))

    # Apply no-collapse option BEFORE parse (affects _compute_early_collapse_decisions)
    if args.no_collapse:
        print("Node collapse DISABLED")
        # Pre-initialize collapse decisions dict - _compute_early_collapse_decisions
        # will populate it, but we can override it with a custom handler
        original_compute = engine._compute_early_collapse_decisions
        def no_collapse_handler():
            """Override collapse computation to return empty collapse pairs."""
            engine._device_collapse_decisions = {}
            # Still need to mark it as computed
        engine._compute_early_collapse_decisions = no_collapse_handler

    engine.parse()

    print(f"Circuit: {engine.num_nodes} external nodes")
    if args.no_collapse:
        print("  (node collapse disabled - all internal nodes preserved)")

    print(f"\nRunning transient simulation...")
    print(f"  t_stop={args.t_stop:.2e}s, dt={args.dt:.2e}s, max_steps={args.max_steps}")
    print(f"  sparse={args.sparse}, use_scan={args.use_scan}, temperature={args.temperature}K")

    result = engine.run_transient(
        t_stop=args.t_stop,
        dt=args.dt,
        max_steps=args.max_steps,
        use_sparse=args.sparse,
        use_scan=args.use_scan,
        use_while_loop=args.use_while_loop,
        temperature=args.temperature,
    )

    # Extract data
    time = np.array(result.times)
    voltages = {name: np.array(v) for name, v in result.voltages.items()}
    currents = {name: np.array(i) for name, i in result.currents.items()}

    print(f"\nSimulation complete: {len(time)} time points, {time[0]:.2e}s to {time[-1]:.2e}s")
    print(f"Voltage nodes: {list(voltages.keys())}")
    print(f"Current probes: {list(currents.keys())}")

    if not args.no_plot:
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

        output_path = Path(args.output) if args.output else base / 'ring_vi_plot.png'
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved plot to {output_path}")

    # Print summary statistics
    ring_nodes = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
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
        print("\nWARNING: Ring oscillator is NOT oscillating (voltage swing < 100mV)")
        return 1
    else:
        print(f"\nRing oscillator is oscillating (max swing: {max(swings):.3f}V)")
        return 0


if __name__ == '__main__':
    sys.exit(main())
