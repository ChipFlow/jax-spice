#!/usr/bin/env python3
"""Check PSP103 parameter structure and defaults."""

import sys
sys.path.insert(0, 'openvaf-py')

import openvaf_py
from pathlib import Path
import json

# Compile PSP103
psp103_va = Path(__file__).parent / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
modules = openvaf_py.compile_va(str(psp103_va))
psp103 = modules[0]

print(f'Init params: {len(psp103.init_param_names)}')
print(f'Eval params: {psp103.func_num_params}')
defaults = psp103.get_param_defaults()
print(f'Param defaults: {len(defaults)}')
print()

# Load model parameters
params_file = Path(__file__).parent / "scripts" / "psp103_model_params.json"
with open(params_file) as f:
    all_params = json.load(f)
pmos_model_params = all_params['pmos']

print('First 30 init params (with defaults and model file values):')
for i, name in enumerate(psp103.init_param_names[:30]):
    default_val = defaults.get(name.lower(), None)
    model_val = pmos_model_params.get(name, None)
    print(f'  {i}: {name:20s} default={default_val!s:15s} model={model_val!s:15s}')

print('\nChecking for init params NOT in defaults:')
missing_defaults = []
for i, name in enumerate(psp103.init_param_names):
    if name.lower() not in defaults and name not in ['$temperature', 'mfactor']:
        missing_defaults.append((i, name))

print(f'Found {len(missing_defaults)} init params without defaults:')
for i, name in missing_defaults[:20]:
    print(f'  {i}: {name}')
