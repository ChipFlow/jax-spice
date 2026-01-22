#!/usr/bin/env python3
"""Plot Full MNA vs VACASK comparison for ring oscillator.

Compares voltage and current waveforms between JAX-SPICE Full MNA
and VACASK reference simulator.
"""

import os
import sys
import struct

# Set JAX platform before importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt

from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.transient import FullMNAStrategy


def read_spice_raw(filename):
    """Read a SPICE raw file (ASCII header + binary data)."""
    with open(filename, 'rb') as f:
        content = f.read()

    # Find the Binary: marker
    binary_marker = b'Binary:\n'
    binary_pos = content.find(binary_marker)
    if binary_pos == -1:
        raise ValueError("Binary marker not found in raw file")

    # Parse header
    header = content[:binary_pos].decode('utf-8')
    lines = header.strip().split('\n')

    n_vars = None
    n_points = None
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
                idx = int(parts[0])
                name = parts[1]
                variables.append(name)

    if n_vars is None or n_points is None:
        raise ValueError("Could not parse header")

    # Parse binary data
    binary_data = content[binary_pos + len(binary_marker):]

    # Each point has n_vars double values
    point_size = n_vars * 8
    expected_size = n_points * point_size

    if len(binary_data) < expected_size:
        print(f"Warning: binary data size {len(binary_data)} < expected {expected_size}")
        n_points = len(binary_data) // point_size

    # Read as doubles
    data = np.zeros((n_points, n_vars), dtype=np.float64)
    for i in range(n_points):
        offset = i * point_size
        for j in range(n_vars):
            val_bytes = binary_data[offset + j*8 : offset + (j+1)*8]
            if len(val_bytes) == 8:
                data[i, j] = struct.unpack('d', val_bytes)[0]

    # Build result dict
    result = {}
    for i, name in enumerate(variables):
        result[name] = data[:, i]

    return result


def main():
    ring_sim = 'vendor/VACASK/benchmark/ring/vacask/runme.sim'
    vacask_raw = 'vendor/VACASK/tran1.raw'

    if not os.path.exists(ring_sim):
        print(f"Error: {ring_sim} not found")
        return 1

    print("=" * 70)
    print("Full MNA vs VACASK Comparison Plot")
    print("=" * 70)

    # Load VACASK reference
    print("\n1. Loading VACASK reference data...")
    if os.path.exists(vacask_raw):
        vacask_data = read_spice_raw(vacask_raw)
        t_vacask = vacask_data['time']
        V1_vacask = vacask_data['1']

        # Try to find VDD current
        I_vacask = None
        for key in vacask_data.keys():
            if 'vdd' in key.lower() and ('flow' in key.lower() or 'br' in key.lower()):
                I_vacask = vacask_data[key]
                print(f"   Found current column: {key}")
                break

        print(f"   VACASK time range: [{t_vacask.min()*1e9:.2f}, {t_vacask.max()*1e9:.2f}] ns")
        print(f"   VACASK points: {len(t_vacask)}")
        if I_vacask is not None:
            print(f"   VACASK I_VDD range: [{I_vacask.min()*1e3:.3f}, {I_vacask.max()*1e3:.3f}] mA")
    else:
        print(f"   VACASK file not found: {vacask_raw}")
        return 1

    # Run Full MNA to match VACASK time range
    t_stop = min(t_vacask.max(), 100e-9)  # Cap at 100ns for reasonable runtime
    dt = 1e-12  # 1ps timestep

    print(f"\n2. Running FullMNAStrategy (t_stop={t_stop*1e9:.1f}ns)...")
    runner = CircuitEngine(ring_sim)
    runner.parse()

    full_mna = FullMNAStrategy(runner, use_sparse=False)
    times_mna, voltages_mna, stats_mna = full_mna.run(t_stop=t_stop, dt=dt)

    t_mna = np.asarray(times_mna)
    V1_mna = np.asarray(voltages_mna['1'])
    I_mna = np.asarray(stats_mna['currents']['vdd'])

    print(f"   Full MNA points: {len(t_mna)}")
    print(f"   Full MNA I_VDD range: [{I_mna.min()*1e3:.3f}, {I_mna.max()*1e3:.3f}] mA")

    # Create comparison plot
    print("\n3. Creating comparison plot...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Voltage comparison
    ax1 = axes[0]
    ax1.plot(t_vacask * 1e9, V1_vacask, 'b-', linewidth=1, label='VACASK', alpha=0.8)
    ax1.plot(t_mna * 1e9, V1_mna, 'r--', linewidth=1, label='Full MNA', alpha=0.8)
    ax1.set_ylabel('V(1) [V]')
    ax1.set_title('Ring Oscillator: Full MNA vs VACASK Comparison')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, t_stop * 1e9)

    # Current comparison
    ax2 = axes[1]
    if I_vacask is not None:
        ax2.plot(t_vacask * 1e9, I_vacask * 1e3, 'b-', linewidth=1, label='VACASK', alpha=0.8)
    ax2.plot(t_mna * 1e9, I_mna * 1e3, 'r--', linewidth=1, label='Full MNA', alpha=0.8)
    ax2.set_ylabel('I(VDD) [mA]')
    ax2.set_xlabel('Time [ns]')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_file = 'ring_full_mna_vacask_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_file}")

    # Compute comparison metrics (skip initial transient)
    print("\n4. Comparison metrics (after 2ns transient)...")
    t_start = 2e-9

    # Find common time range
    t_min = max(t_mna.min(), t_vacask.min(), t_start)
    t_max = min(t_mna.max(), t_vacask.max())

    # Interpolate to common time base
    n_points = 5000
    t_common = np.linspace(t_min, t_max, n_points)

    V1_mna_interp = np.interp(t_common, t_mna, V1_mna)
    V1_vacask_interp = np.interp(t_common, t_vacask, V1_vacask)

    V_rms_diff = np.sqrt(np.mean((V1_mna_interp - V1_vacask_interp)**2))
    print(f"   V(1) RMS difference: {V_rms_diff*1e3:.2f} mV")

    if I_vacask is not None:
        I_mna_interp = np.interp(t_common, t_mna, I_mna)
        I_vacask_interp = np.interp(t_common, t_vacask, I_vacask)

        I_mean_mna = I_mna_interp.mean()
        I_mean_vacask = I_vacask_interp.mean()
        I_mean_diff = (I_mean_mna - I_mean_vacask) / abs(I_mean_vacask) * 100

        I_rms_diff = np.sqrt(np.mean((I_mna_interp - I_vacask_interp)**2))
        I_rms_rel = I_rms_diff / abs(I_mean_vacask) * 100

        # dI/dt comparison
        dt_common = np.diff(t_common)
        dIdt_mna = np.diff(I_mna_interp) / dt_common
        dIdt_vacask = np.diff(I_vacask_interp) / dt_common

        max_dIdt_mna = np.max(np.abs(dIdt_mna))
        max_dIdt_vacask = np.max(np.abs(dIdt_vacask))
        dIdt_diff = (max_dIdt_mna - max_dIdt_vacask) / max_dIdt_vacask * 100

        print(f"   I(VDD) VACASK mean: {I_mean_vacask*1e3:.3f} mA")
        print(f"   I(VDD) Full MNA mean: {I_mean_mna*1e3:.3f} mA")
        print(f"   I(VDD) mean difference: {I_mean_diff:+.1f}%")
        print(f"   I(VDD) RMS difference: {I_rms_diff*1e3:.3f} mA ({I_rms_rel:.1f}%)")
        print(f"   Max |dI/dt| VACASK: {max_dIdt_vacask*1e-6:.2f} mA/ns")
        print(f"   Max |dI/dt| Full MNA: {max_dIdt_mna*1e-6:.2f} mA/ns")
        print(f"   Max |dI/dt| difference: {dIdt_diff:+.1f}%")

    print("\n" + "=" * 70)
    print("Plot saved to:", output_file)
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
