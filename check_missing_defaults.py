#!/usr/bin/env python3
"""Check which init parameters are missing defaults."""

import sys
sys.path.insert(0, 'openvaf-py')

import openvaf_py
from pathlib import Path

# Compile PSP103
psp103_va = Path(__file__).parent / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
modules = openvaf_py.compile_va(str(psp103_va))
psp103 = modules[0]

defaults = psp103.get_param_defaults()

# Check which init params don't have defaults
missing = []
have_defaults = []

for name in psp103.init_param_names:
    if name in ['$temperature', 'mfactor']:
        # These are special system parameters
        continue
    if name.lower() not in defaults:
        missing.append(name)
    else:
        have_defaults.append(name)

print(f"Total init params: {len(psp103.init_param_names)}")
print(f"Have defaults: {len(have_defaults)}")
print(f"Missing defaults: {len(missing)}")
print()

if missing:
    print("Init params WITHOUT defaults (will get 0.0):")
    for name in missing[:50]:
        print(f"  {name}")
else:
    print("All init params have defaults!")

# Also check what temperature and mfactor default to
print(f"\nSpecial params:")
print(f"  $temperature: {'YES' if '$temperature' in defaults else 'NO'}")
print(f"  mfactor: {'YES' if 'mfactor' in defaults else 'NO'}")
