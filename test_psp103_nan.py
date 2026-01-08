#!/usr/bin/env python3
"""Test if PSP103 still produces NaN values with defaults fix."""

import sys
sys.path.insert(0, 'openvaf-py')

import openvaf_py
from pathlib import Path
import json
import numpy as np

# Compile PSP103
psp103_va = Path(__file__).parent / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
modules = openvaf_py.compile_va(str(psp103_va))
psp103 = modules[0]

# Load model parameters
params_file = Path(__file__).parent / "scripts" / "psp103_model_params.json"
with open(params_file) as f:
    all_params = json.load(f)
pmos_model_params = all_params['pmos']

# Build init params - only provide a FEW user params, let defaults handle the rest
init_params = {}
for name in psp103.init_param_names:
    if name == '$temperature':
        init_params[name] = 300.0
    elif name.upper() == 'TYPE':
        init_params[name] = -1.0  # PMOS
    elif name.upper() == 'W':
        init_params[name] = 20e-6
    elif name.upper() == 'L':
        init_params[name] = 1e-6
    elif name.lower() == 'mfactor':
        init_params[name] = 1.0
    # For model params from JSON, use lowercase key
    elif name.lower() in pmos_model_params:
        init_params[name] = pmos_model_params[name.lower()]

print(f"Providing {len(init_params)} of {len(psp103.init_param_names)} init params")

# Get cached values from init
cache_values = psp103.debug_init_cache(init_params)

# Check for NaN
nan_values = [(idx, val) for idx, val in cache_values if np.isnan(val)]
inf_values = [(idx, val) for idx, val in cache_values if np.isinf(val)]

print(f"\nCache analysis:")
print(f"  Total cache values: {len(cache_values)}")
print(f"  NaN values: {len(nan_values)}")
print(f"  Inf values: {len(inf_values)}")

if nan_values:
    print(f"\nNaN cache indices:")
    for idx, val in nan_values[:20]:
        print(f"  cache[{idx}] = {val}")

if inf_values:
    print(f"\nInf cache indices:")
    for idx, val in inf_values[:20]:
        print(f"  cache[{idx}] = {val}")

# Now test actual device evaluation
print("\n\nTesting device evaluation at Vgs=-1.2V:")
voltages = {
    'D': 1.2,  # Vds = 1.2V
    'G': 0.0,  # Vgs = -1.2V (referenced to S)
    'S': 1.2,
    'B': 1.2
}

device_params = {
    'TYPE': -1.0,  # PMOS
    'W': 20e-6,
    'L': 1e-6
}
# Add model params
for key in pmos_model_params:
    device_params[key.upper()] = pmos_model_params[key]

try:
    result = psp103.run_init_eval(voltages, device_params)
    print(f"Success! Ids = {result.get('i_D_S', 'N/A')}")
    print(f"Result keys: {list(result.keys())}")
except Exception as e:
    print(f"ERROR: {e}")
