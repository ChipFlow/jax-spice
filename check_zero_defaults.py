#!/usr/bin/env python3
"""Check which PSP103 defaults are zero."""

import sys
sys.path.insert(0, 'openvaf-py')

import openvaf_py
from pathlib import Path

# Compile PSP103
psp103_va = Path(__file__).parent / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
modules = openvaf_py.compile_va(str(psp103_va))
psp103 = modules[0]

defaults = psp103.get_param_defaults()

# Find parameters with zero defaults
zero_defaults = {k: v for k, v in defaults.items() if v == 0.0}
nonzero_defaults = {k: v for k, v in defaults.items() if v != 0.0}

print(f"Total defaults: {len(defaults)}")
print(f"Zero defaults: {len(zero_defaults)}")
print(f"Nonzero defaults: {len(nonzero_defaults)}")
print()

# Check for suspicious zero defaults that might cause division by zero
suspicious_names = ['tox', 'nsubo', 'npck', 'wseg', 'lpck', 'wsegp', 'epsroxo', 'qmc']
print("Checking suspicious parameters:")
for name in suspicious_names:
    val = defaults.get(name, 'NOT FOUND')
    print(f"  {name:15s} = {val}")

print(f"\nSample nonzero defaults:")
for i, (k, v) in enumerate(list(nonzero_defaults.items())[:20]):
    print(f"  {k:15s} = {v}")

print(f"\nSample zero defaults:")
for i, (k, v) in enumerate(list(zero_defaults.items())[:30]):
    print(f"  {k:15s} = {v}")
