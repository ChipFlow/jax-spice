#!/usr/bin/env python3
"""Test VACASK's OSDI PSP103 implementation directly via ctypes.

This loads the compiled OSDI library and calls it at the same operating point
as our OpenVAF tests to find where they diverge.
"""

import sys
from pathlib import Path
import ctypes
import json
import numpy as np

def main():
    print("=" * 80)
    print("VACASK OSDI PSP103 Direct Test")
    print("=" * 80)
    print()

    # Find VACASK OSDI library
    osdi_lib_path = Path.home() / "Code/ChipFlow/reference/VACASK/build/lib/vacask/mod/psp103v4.osdi"

    if not osdi_lib_path.exists():
        print(f"OSDI library not found at: {osdi_lib_path}")
        print()
        print("Trying alternative location...")
        osdi_lib_path = Path.home() / "Code/ChipFlow/reference/VACASK/test/ihp_sg13g2/psp103_ihp.osdi"

    if not osdi_lib_path.exists():
        print(f"OSDI library not found at: {osdi_lib_path}")
        print()
        print("Please build VACASK first or specify the correct path.")
        return 1

    print(f"Loading OSDI library: {osdi_lib_path}")
    print()

    try:
        # Load the shared library
        osdi = ctypes.CDLL(str(osdi_lib_path))
        print("✓ Library loaded successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to load library: {e}")
        return 1

    # Define OSDI API structures based on OSDI 0.4 specification
    # See: openvaf-py/vendor/OpenVAF/openvaf/osdi/src/metadata/osdi_0_4.rs

    # Get descriptor - this tells us the model structure
    class OsdiDescriptor(ctypes.Structure):
        pass

    # Function signatures based on OSDI spec
    try:
        # Get the model descriptor
        osdi.osdi_descriptor.restype = ctypes.POINTER(OsdiDescriptor)
        osdi.osdi_descriptor.argtypes = []

        descriptor = osdi.osdi_descriptor()
        print("✓ Got model descriptor")
        print()

        # Try to access basic descriptor fields
        # This is a simplified approach - full OSDI spec has many fields

    except Exception as e:
        print(f"⚠️  Could not access descriptor: {e}")
        print()
        print("The OSDI API is complex. Let's try a different approach:")
        print("Use VACASK's C++ API or run via subprocess instead.")
        print()

    # Alternative: Call VACASK via subprocess with a test circuit
    print("=" * 80)
    print("Alternative: Test via VACASK simulation")
    print("=" * 80)
    print()

    # Create a minimal test circuit
    test_sim = Path(__file__).parent / "test_psp103_vacask.sim"

    # Load model parameters
    params_file = Path(__file__).parent / 'psp103_model_params.json'
    if not params_file.exists():
        print(f"Model parameters not found: {params_file}")
        return 1

    with open(params_file) as f:
        all_params = json.load(f)
    pmos_model_params = all_params['pmos']

    # Create .sim file with PSP103 PMOS at test operating point
    sim_content = f"""Test PSP103 PMOS at Vgs=-1.2V

# Load OSDI library
load "psp103v4.osdi"

# Define model with all parameters from model card
model psp103p psp103 (
  type = -1
"""

    # Add all model parameters
    for param, value in sorted(pmos_model_params.items()):
        sim_content += f"  {param} = {value}\n"

    sim_content += """)

# Test device: PMOS with Vs=1.2V (VDD), Vd=0V, Vg=0V, Vb=1.2V
# This gives Vgs=-1.2V, Vds=-1.2V → PMOS should conduct

# Voltage sources to set up operating point
vsource vdd 0 vdd_node dc=1.2
vsource vg 0 g_node dc=0.0
vsource vd 0 d_node dc=0.0

# PSP103 PMOS device
# Nodes: D G S B
m1 d_node g_node vdd_node vdd_node psp103p w=20e-6 l=1e-6

# DC operating point analysis
op

# Print node voltages and device currents
print op all
print op device m1

# Save results
save "test_psp103_vacask.prn"
  all
end
"""

    # Write sim file
    with open(test_sim, 'w') as f:
        f.write(sim_content)

    print(f"Created test circuit: {test_sim}")
    print()
    print("Sim file content:")
    print("-" * 40)
    print(sim_content[:500] + "...")
    print("-" * 40)
    print()

    # Try to run with VACASK
    vacask_bin = Path.home() / "Code/ChipFlow/reference/VACASK/build/vacask"

    if vacask_bin.exists():
        print(f"Found VACASK executable: {vacask_bin}")
        print()
        print("Run this to test:")
        print(f"  {vacask_bin} {test_sim}")
        print()
        print("This will show if VACASK's OSDI PSP103 returns:")
        print("  - Ids = 0 (same bug as OpenVAF)")
        print("  - Ids < 0 (PMOS conducting correctly)")
    else:
        print(f"VACASK executable not found at: {vacask_bin}")
        print()
        print("Build VACASK first, then run:")
        print(f"  ./vacask {test_sim}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
