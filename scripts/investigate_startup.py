#!/usr/bin/env python3
"""Investigate the startup current spike in VACASK vs Full MNA."""

import os
import sys
import struct

os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt

from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.transient import FullMNAStrategy


def read_spice_raw(filename):
    """Read a SPICE raw file."""
    with open(filename, 'rb') as f:
        content = f.read()

    binary_marker = b'Binary:\n'
    binary_pos = content.find(binary_marker)
    header = content[:binary_pos].decode('utf-8')
    lines = header.strip().split('\n')

    n_vars = n_points = None
    variables = []
    in_variables = False

    for line in lines:
        if line.startswith('No. Variables:'):
            n_vars = int(line.split(':')[1].strip())
        elif line.startswith('No. Points:'):
            n_points = int(line.split(':')[1].strip())
        elif line.startswith('Variables:'):
            in_variables = True
        elif in_variables and line.strip():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                variables.append(parts[1])

    binary_data = content[binary_pos + len(binary_marker):]
    point_size = n_vars * 8
    n_points = min(n_points, len(binary_data) // point_size)

    data = np.zeros((n_points, n_vars), dtype=np.float64)
    for i in range(n_points):
        offset = i * point_size
        for j in range(n_vars):
            val_bytes = binary_data[offset + j*8 : offset + (j+1)*8]
            if len(val_bytes) == 8:
                data[i, j] = struct.unpack('d', val_bytes)[0]

    return {name: data[:, i] for i, name in enumerate(variables)}


def main():
    ring_sim = 'vendor/VACASK/benchmark/ring/vacask/runme.sim'
    vacask_raw = 'vendor/VACASK/tran1.raw'

    print("Loading VACASK data...")
    vacask_data = read_spice_raw(vacask_raw)
    t_vacask = vacask_data['time']
    V1_vacask = vacask_data['1']
    I_vacask = vacask_data.get('vdd:flow(br)')

    print("Running Full MNA (10ns to capture startup)...")
    runner = CircuitEngine(ring_sim)
    runner.parse()
    full_mna = FullMNAStrategy(runner, use_sparse=False)
    times_mna, voltages_mna, stats_mna = full_mna.run(t_stop=10e-9, dt=1e-12)

    t_mna = np.asarray(times_mna)
    V1_mna = np.asarray(voltages_mna['1'])
    I_mna = np.asarray(stats_mna['currents']['vdd'])

    # Focus on startup region (0-5ns)
    print("\n" + "=" * 60)
    print("Analyzing startup region (0-5ns):")
    print("=" * 60)

    mask_startup_v = t_vacask < 5e-9
    mask_startup_m = t_mna < 5e-9

    print(f"\nVACASK current during 0-5ns:")
    print(f"  Min: {I_vacask[mask_startup_v].min()*1e6:.1f} µA")
    print(f"  Max: {I_vacask[mask_startup_v].max()*1e6:.1f} µA")
    print(f"  Peak-to-peak: {(I_vacask[mask_startup_v].max() - I_vacask[mask_startup_v].min())*1e6:.1f} µA")

    print(f"\nFull MNA current during 0-5ns:")
    print(f"  Min: {I_mna[mask_startup_m].min()*1e6:.1f} µA")
    print(f"  Max: {I_mna[mask_startup_m].max()*1e6:.1f} µA")
    print(f"  Peak-to-peak: {(I_mna[mask_startup_m].max() - I_mna[mask_startup_m].min())*1e6:.1f} µA")

    # Find the peak in VACASK
    peak_idx = np.argmax(np.abs(I_vacask[mask_startup_v]))
    peak_time = t_vacask[mask_startup_v][peak_idx]
    peak_current = I_vacask[mask_startup_v][peak_idx]
    print(f"\nVACASK peak at t={peak_time*1e9:.3f}ns: {peak_current*1e6:.1f} µA")

    # Check what Full MNA shows at that time
    mna_at_peak = np.interp(peak_time, t_mna, I_mna)
    print(f"Full MNA at same time: {mna_at_peak*1e6:.1f} µA")

    # Check pulse source behavior - what's the input?
    print("\n" + "=" * 60)
    print("Checking circuit startup conditions:")
    print("=" * 60)

    # Check V(1) at startup
    print(f"\nV(1) at t=0:")
    print(f"  VACASK: {V1_vacask[0]:.3f} V")
    print(f"  Full MNA: {V1_mna[0]:.3f} V")

    # Check if there's a pulse triggering the oscillation
    print(f"\nV(1) at t=1ns (after pulse):")
    idx_v_1ns = np.searchsorted(t_vacask, 1e-9)
    idx_m_1ns = np.searchsorted(t_mna, 1e-9)
    print(f"  VACASK: {V1_vacask[idx_v_1ns]:.3f} V")
    print(f"  Full MNA: {V1_mna[idx_m_1ns]:.3f} V")

    # Create diagnostic plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Startup window
    t_start, t_end = 0, 6e-9

    # Panel 1: Voltage
    ax1 = axes[0]
    mask_v = (t_vacask >= t_start) & (t_vacask <= t_end)
    mask_m = (t_mna >= t_start) & (t_mna <= t_end)
    ax1.plot(t_vacask[mask_v] * 1e9, V1_vacask[mask_v], 'b-', lw=1.5, label='VACASK', alpha=0.8)
    ax1.plot(t_mna[mask_m] * 1e9, V1_mna[mask_m], 'r--', lw=1.5, label='Full MNA', alpha=0.8)
    ax1.set_ylabel('V(1) [V]')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Startup Transient Analysis (0-6ns)', fontsize=12, fontweight='bold')

    # Panel 2: Current
    ax2 = axes[1]
    ax2.plot(t_vacask[mask_v] * 1e9, I_vacask[mask_v] * 1e6, 'b-', lw=1.5, label='VACASK', alpha=0.8)
    ax2.plot(t_mna[mask_m] * 1e9, I_mna[mask_m] * 1e6, 'r--', lw=1.5, label='Full MNA', alpha=0.8)
    ax2.set_ylabel('I(VDD) [µA]')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=1.0, color='g', linestyle=':', alpha=0.5, label='Pulse edge')

    # Panel 3: dI/dt
    ax3 = axes[2]
    dt_v = np.diff(t_vacask)
    dt_m = np.diff(t_mna)
    dIdt_v = np.diff(I_vacask) / dt_v
    dIdt_m = np.diff(I_mna) / dt_m
    t_didt_v = t_vacask[:-1] + dt_v/2
    t_didt_m = t_mna[:-1] + dt_m/2

    mask_dv = (t_didt_v >= t_start) & (t_didt_v <= t_end)
    mask_dm = (t_didt_m >= t_start) & (t_didt_m <= t_end)
    ax3.plot(t_didt_v[mask_dv] * 1e9, dIdt_v[mask_dv] * 1e-6, 'b-', lw=1, label='VACASK', alpha=0.8)
    ax3.plot(t_didt_m[mask_dm] * 1e9, dIdt_m[mask_dm] * 1e-6, 'r--', lw=1, label='Full MNA', alpha=0.8)
    ax3.set_ylabel('dI/dt [mA/ns]')
    ax3.set_xlabel('Time [ns]')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'ring_startup_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
