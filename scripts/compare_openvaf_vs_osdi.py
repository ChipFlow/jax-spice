#!/usr/bin/env python3
"""Compare OpenVAF-compiled PSP103 model vs VACASK OSDI PSP103.

Tests device evaluation at V=0 (initial condition) to find where models diverge.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from jax_spice.analysis.engine import CircuitEngine

def main():
    sim_path = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"Ring benchmark not found at {sim_path}")
        return 1

    print("="*80)
    print("OpenVAF vs OSDI Model Comparison")
    print("="*80)
    print("\nGoal: Find where PSP103 models diverge at V=0")
    print()

    # Parse circuit to get PSP103 parameters
    engine = CircuitEngine(sim_path)
    engine.parse()

    # Find a PSP103 device
    psp103_device = None
    for dev in engine.devices:
        if dev['model'] == 'psp103':
            psp103_device = dev
            break

    if not psp103_device:
        print("No PSP103 device found!")
        return 1

    print(f"Found PSP103 device: {psp103_device['name']}")
    print(f"  Nodes: {psp103_device['nodes']}")
    print(f"  Model card: {psp103_device.get('model_card', 'unknown')}")
    print()

    # Test case 1: All terminals at V=0 (initial condition)
    print("="*80)
    print("Test Case 1: V(d,g,s,b) = 0V (Initial Condition)")
    print("="*80)
    print()
    print("This is the state at DC solver start.")
    print("VACASK expects:")
    print("  - NMOS: Vgs=0, NMOS off, Ids≈0")
    print("  - PMOS: Vgs=0, PMOS on (Vth<0), Ids>0 (charging output)")
    print()

    # We need to:
    # 1. Load OpenVAF PSP103 model
    # 2. Call it with V=0
    # 3. Load VACASK OSDI PSP103 model
    # 4. Call it with V=0
    # 5. Compare

    print("TODO: Implement model comparison")
    print("  1. Load OpenVAF-compiled PSP103 from openvaf-py")
    print("  2. Load VACASK OSDI PSP103 (via ctypes or subprocess)")
    print("  3. Set up test inputs: Vd=0, Vg=0, Vs=0, Vb=0")
    print("  4. Call both models' eval functions")
    print("  5. Compare outputs:")
    print("     - Currents: Id, Ig, Is, Ib")
    print("     - Charges: Qd, Qg, Qs, Qb")
    print("     - Jacobian: dId/dVd, dId/dVg, etc.")
    print()

    # For now, let's at least get the OpenVAF model evaluation
    print("="*80)
    print("OpenVAF PSP103 Evaluation at V=0")
    print("="*80)

    # Get device parameters from model card
    # This requires parsing the model card from the netlist
    # For now, just document what we need

    print()
    print("Next steps:")
    print("  1. Extract PSP103 model parameters from .sim file")
    print("  2. Call openvaf-py PSP103 eval with V=0")
    print("  3. Set up VACASK OSDI harness (via shared library)")
    print("  4. Compare and identify divergence")
    print()
    print("This will tell us if the problem is:")
    print("  - Wrong OpenVAF compilation")
    print("  - Wrong parameter mapping")
    print("  - Different model behavior")

    return 0

if __name__ == "__main__":
    sys.exit(main())
