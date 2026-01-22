#!/usr/bin/env python3
"""Debug DC initial conditions."""

import os
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'

import logging
logging.basicConfig(level=logging.DEBUG)

from jax_spice.analysis.engine import CircuitEngine
from jax_spice.analysis.transient import FullMNAStrategy

ring_sim = 'vendor/VACASK/benchmark/ring/vacask/runme.sim'

print("Creating runner...")
runner = CircuitEngine(ring_sim)
runner.parse()

print(f"\nicmode = {runner.analysis_params.get('icmode', 'NOT SET')}")

print("\nCreating Full MNA strategy...")
full_mna = FullMNAStrategy(runner, use_sparse=False)

print("\nRunning for 1 timestep to check initial conditions...")
times, voltages, stats = full_mna.run(t_stop=1e-12, dt=1e-12)

import numpy as np
print(f"\nV(1) at t=0: {float(voltages['1'][0]):.6f} V")
print(f"V(2) at t=0: {float(voltages['2'][0]):.6f} V")
print(f"I(VDD) at t=0: {float(stats['currents']['vdd'][0])*1e6:.1f} µA")
