#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["jax", "matplotlib", "numpy"]
# ///
"""Compare ring oscillator with fixed vs adaptive timestep strategies."""

import sys
sys.path.insert(0, '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice')

import matplotlib.pyplot as plt
import numpy as np
import jax

# Force CPU for consistency
jax.config.update('jax_platforms', 'cpu')

from pathlib import Path
from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.transient import AdaptiveStrategy, AdaptiveConfig, FullMNAStrategy

# Find ring benchmark
VACASK_PATH = Path('/Users/roberttaylor/Code/ChipFlow/reference/jax-spice/vendor/VACASK')
ring_sim = VACASK_PATH / 'benchmark' / 'ring' / 'vacask' / 'runme.sim'

if not ring_sim.exists():
    print(f"Ring benchmark not found at {ring_sim}")
    sys.exit(1)

print("Loading ring oscillator benchmark...")
engine = CircuitEngine(str(ring_sim))
engine.parse()

# Simulation parameters
t_stop = 20e-9  # 20ns - enough for a few oscillations
dt_fixed = 1e-12  # 1ps fixed timestep

print(f"\n=== Fixed Timestep (FullMNAStrategy) ===")
print(f"t_stop={t_stop*1e9:.1f}ns, dt={dt_fixed*1e12:.1f}ps")

# Run with fixed timestep using FullMNAStrategy (returns currents)
fixed_strategy = FullMNAStrategy(engine, use_sparse=False)
times_fixed, voltages_fixed, stats_fixed = fixed_strategy.run(t_stop=t_stop, dt=dt_fixed)
currents_fixed = stats_fixed.get('currents', {})

print(f"Steps: {stats_fixed.get('total_timesteps', len(times_fixed))}")
print(f"NR iterations: {stats_fixed.get('total_nr_iterations', 'N/A')}")
print(f"Time: {stats_fixed.get('wall_time', 0):.2f}s")
if 'convergence_rate' in stats_fixed:
    print(f"Convergence: {stats_fixed['convergence_rate']*100:.1f}%")

print(f"\n=== Adaptive Timestep (AdaptiveStrategy) ===")
# Configure adaptive with reasonable settings for ring oscillator
# Using defaults for lte_ratio=7.0 and warmup_steps=4
adaptive_config = AdaptiveConfig(
    min_dt=1e-14,   # 10fs minimum
    max_dt=1e-10,   # 100ps maximum
    reltol=1e-3,
    abstol=1e-9,
    grow_factor=2.0,
)

adaptive_strategy = AdaptiveStrategy(engine, use_sparse=False, config=adaptive_config)
times_adaptive, voltages_adaptive, currents_adaptive, stats_adaptive = adaptive_strategy.run(t_stop=t_stop, dt=dt_fixed)

print(f"Steps: {stats_adaptive.get('total_timesteps', len(times_adaptive))}")
print(f"NR iterations: {stats_adaptive.get('total_nr_iterations', 'N/A')}")
print(f"Time: {stats_adaptive.get('wall_time', 0):.2f}s")
if 'convergence_rate' in stats_adaptive:
    print(f"Convergence: {stats_adaptive['convergence_rate']*100:.1f}%")
print(f"Rejected steps: {stats_adaptive.get('rejected_steps', 0)}")
print(f"dt range: {stats_adaptive.get('min_dt', 0)*1e12:.3f}ps - {stats_adaptive.get('max_dt', 0)*1e12:.3f}ps")

# Find an output node to plot
output_nodes = [name for name in voltages_fixed.keys() if 'out' in name.lower() or 'y' in name.lower()]
if not output_nodes:
    # Fall back to first non-vdd node
    output_nodes = [name for name in voltages_fixed.keys() if 'vdd' not in name.lower()]

plot_node = output_nodes[0] if output_nodes else list(voltages_fixed.keys())[0]
print(f"\nPlotting node: {plot_node}")

# Find VDD source for current
vdd_sources = [name for name in currents_fixed.keys() if 'vdd' in name.lower()]
if not vdd_sources:
    vdd_sources = list(currents_fixed.keys())
current_source = vdd_sources[0] if vdd_sources else None
print(f"Current source: {current_source}")
print(f"Available current sources (fixed): {list(currents_fixed.keys())}")
print(f"Available current sources (adaptive): {list(currents_adaptive.keys())}")

# Create figure with 5 subplots
fig, axes = plt.subplots(5, 1, figsize=(12, 16))

# Plot 1: Voltage comparison
ax1 = axes[0]
ax1.plot(np.array(times_fixed) * 1e9, np.array(voltages_fixed[plot_node]),
         'b-', label=f'Fixed dt={dt_fixed*1e12:.0f}ps', alpha=0.7)
ax1.plot(np.array(times_adaptive) * 1e9, np.array(voltages_adaptive[plot_node]),
         'r-', label='Adaptive', alpha=0.7)
ax1.set_xlabel('Time (ns)')
ax1.set_ylabel('Voltage (V)')
ax1.set_title(f'Ring Oscillator Output ({plot_node})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Timestep comparison
ax2 = axes[1]
dt_fixed_arr = np.diff(np.array(times_fixed))
dt_adaptive_arr = np.diff(np.array(times_adaptive))
ax2.semilogy(np.array(times_fixed[:-1]) * 1e9, dt_fixed_arr * 1e12,
             'b-', label='Fixed', alpha=0.7)
ax2.semilogy(np.array(times_adaptive[:-1]) * 1e9, dt_adaptive_arr * 1e12,
             'r-', label='Adaptive', alpha=0.7)
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Timestep (ps)')
ax2.set_title('Timestep Size')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Supply current comparison
ax3 = axes[2]
if current_source and current_source in currents_fixed and current_source in currents_adaptive:
    i_fixed = np.array(currents_fixed[current_source])
    i_adaptive = np.array(currents_adaptive[current_source])
    ax3.plot(np.array(times_fixed) * 1e9, i_fixed * 1e3,
             'b-', label='Fixed', alpha=0.7)
    ax3.plot(np.array(times_adaptive) * 1e9, i_adaptive * 1e3,
             'r-', label='Adaptive', alpha=0.7)
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Current (mA)')
    ax3.set_title(f'Supply Current ({current_source})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'No current data available', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Supply Current (no data)')

# Plot 4: dI/dt comparison
ax4 = axes[3]
if current_source and current_source in currents_fixed and current_source in currents_adaptive:
    i_fixed = np.array(currents_fixed[current_source])
    i_adaptive = np.array(currents_adaptive[current_source])

    # Compute dI/dt
    didt_fixed = np.diff(i_fixed) / np.diff(np.array(times_fixed))
    didt_adaptive = np.diff(i_adaptive) / np.diff(np.array(times_adaptive))

    # Convert to mA/ns
    ax4.plot(np.array(times_fixed[:-1]) * 1e9, didt_fixed * 1e-6,
             'b-', label='Fixed', alpha=0.7)
    ax4.plot(np.array(times_adaptive[:-1]) * 1e9, didt_adaptive * 1e-6,
             'r-', label='Adaptive', alpha=0.7)
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('dI/dt (mA/ns)')
    ax4.set_title(f'Current Derivative ({current_source})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Print dI/dt statistics
    print(f"\ndI/dt statistics:")
    print(f"  Fixed - max: {np.max(np.abs(didt_fixed))*1e-6:.2f} mA/ns")
    print(f"  Adaptive - max: {np.max(np.abs(didt_adaptive))*1e-6:.2f} mA/ns")
else:
    ax4.text(0.5, 0.5, 'No current data available', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Current Derivative (no data)')

# Plot 5: Zoomed view during transition
ax5 = axes[4]
# Find a transition region (where voltage is changing quickly)
v_adaptive = np.array(voltages_adaptive[plot_node])
dv = np.abs(np.diff(v_adaptive))
transition_idx = np.argmax(dv)
t_transition = float(times_adaptive[transition_idx])

# Zoom window around transition
zoom_start = max(0, t_transition - 0.5e-9)
zoom_end = min(t_stop, t_transition + 0.5e-9)

# Filter data for zoom
mask_fixed = (np.array(times_fixed) >= zoom_start) & (np.array(times_fixed) <= zoom_end)
mask_adaptive = (np.array(times_adaptive) >= zoom_start) & (np.array(times_adaptive) <= zoom_end)

ax5.plot(np.array(times_fixed)[mask_fixed] * 1e9, np.array(voltages_fixed[plot_node])[mask_fixed],
         'b.-', label='Fixed', markersize=2, alpha=0.7)
ax5.plot(np.array(times_adaptive)[mask_adaptive] * 1e9, np.array(voltages_adaptive[plot_node])[mask_adaptive],
         'r.-', label='Adaptive', markersize=3, alpha=0.7)
ax5.set_xlabel('Time (ns)')
ax5.set_ylabel('Voltage (V)')
ax5.set_title(f'Zoomed View During Transition ({zoom_start*1e9:.2f}-{zoom_end*1e9:.2f}ns)')
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.tight_layout()

# Add summary text
summary = (
    f"Fixed: {stats_fixed.get('total_timesteps', len(times_fixed))} steps, "
    f"{stats_fixed.get('total_nr_iterations', 'N/A')} NR iters, "
    f"{stats_fixed.get('wall_time', 0):.2f}s\n"
    f"Adaptive: {stats_adaptive.get('total_timesteps', len(times_adaptive))} steps, "
    f"{stats_adaptive.get('total_nr_iterations', 'N/A')} NR iters, "
    f"{stats_adaptive.get('wall_time', 0):.2f}s"
)
fig.text(0.5, 0.01, summary, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.subplots_adjust(bottom=0.05)

# Save figure
output_path = '/Users/roberttaylor/Code/ChipFlow/reference/jax-spice/ring_fixed_vs_adaptive.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved plot to: {output_path}")

plt.show()
