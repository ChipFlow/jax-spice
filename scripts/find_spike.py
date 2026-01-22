#!/usr/bin/env python3
"""Find where the current spike occurs."""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.transient import FullMNAStrategy

ring_sim = 'vendor/VACASK/benchmark/ring/vacask/runme.sim'

runner = CircuitEngine(ring_sim)
runner.parse()
full_mna = FullMNAStrategy(runner, use_sparse=False)
times, voltages, stats = full_mna.run(t_stop=5e-9, dt=1e-12)

t = np.asarray(times)
I = np.asarray(stats['currents']['vdd'])

# Find the spike
spike_idx = np.argmin(I)  # Most negative
spike_time = t[spike_idx]
spike_current = I[spike_idx]

print(f"Spike at t={spike_time*1e9:.3f}ns: I={spike_current*1e6:.1f} µA")
print(f"\nCurrent around spike:")
for i in range(max(0, spike_idx-3), min(len(I), spike_idx+4)):
    print(f"  t={t[i]*1e12:.0f}ps: I={I[i]*1e6:.1f} µA, V(1)={float(voltages['1'][i]):.4f}V")
