#!/usr/bin/env python
"""Test the Phi/Upsilon transformation on PSP102."""
import os
os.environ['JAX_ENABLE_X64'] = 'true'

from pathlib import Path

import openvaf_py
from openvaf_jax.mir.types import parse_mir_function
from openvaf_jax.mir.phi_upsilon import (
    transform_to_phi_upsilon,
    print_phi_upsilon_summary,
    step4_generate_jax_code,
)

PROJECT_ROOT = Path(__file__).parent.parent


def test_simple_model():
    """Test on a simpler model first - resistor."""
    print("=" * 70)
    print("Testing Phi/Upsilon on RESISTOR (simple model)")
    print("=" * 70)

    va_path = PROJECT_ROOT / "vendor" / "VACASK" / "devices" / "resistor.va"
    modules = openvaf_py.compile_va(str(va_path))
    module = modules[0]

    mir_data = module.get_mir_instructions()
    str_constants = module.get_str_constants()
    mir_func = parse_mir_function("eval", mir_data, str_constants)

    print(f"\nMIR stats: {len(mir_func.blocks)} blocks")

    ir = transform_to_phi_upsilon(mir_func)
    print_phi_upsilon_summary(ir)

    if ir.shadow_vars:
        print("\n" + "-" * 70)
        print("Generated JAX code snippet:")
        print("-" * 70)
        code = step4_generate_jax_code(ir)
        print(code[:2000] if len(code) > 2000 else code)


def test_diode():
    """Test on diode (has NMOS/PMOS-like branching)."""
    print("\n" + "=" * 70)
    print("Testing Phi/Upsilon on DIODE (has conditionals)")
    print("=" * 70)

    va_path = PROJECT_ROOT / "vendor" / "VACASK" / "devices" / "diode.va"
    modules = openvaf_py.compile_va(str(va_path))
    module = modules[0]

    mir_data = module.get_mir_instructions()
    str_constants = module.get_str_constants()
    mir_func = parse_mir_function("eval", mir_data, str_constants)

    print(f"\nMIR stats: {len(mir_func.blocks)} blocks")

    ir = transform_to_phi_upsilon(mir_func)
    print_phi_upsilon_summary(ir)

    if ir.shadow_vars:
        print("\n" + "-" * 70)
        print("Generated JAX code snippet:")
        print("-" * 70)
        code = step4_generate_jax_code(ir)
        print(code[:2000] if len(code) > 2000 else code)


def test_psp102():
    """Test on PSP102 (complex model with many PHIs)."""
    print("\n" + "=" * 70)
    print("Testing Phi/Upsilon on PSP102 (complex model)")
    print("=" * 70)

    va_path = PROJECT_ROOT / "vendor" / "OpenVAF" / "integration_tests" / "PSP102" / "psp102_nqs.va"
    modules = openvaf_py.compile_va(str(va_path))
    module = modules[0]

    mir_data = module.get_mir_instructions()
    str_constants = module.get_str_constants()
    mir_func = parse_mir_function("eval", mir_data, str_constants)

    print(f"\nMIR stats: {len(mir_func.blocks)} blocks")

    ir = transform_to_phi_upsilon(mir_func)
    print_phi_upsilon_summary(ir, max_items=20)

    # Show some specific PHIs
    print("\n" + "-" * 70)
    print("Sample Upsilons for first few PHIs:")
    print("-" * 70)

    seen_shadows = set()
    for ups in ir.upsilons:
        if ups.shadow_name not in seen_shadows and len(seen_shadows) < 5:
            seen_shadows.add(ups.shadow_name)
            # Find all upsilons for this shadow
            related = [u for u in ir.upsilons if u.shadow_name == ups.shadow_name]
            print(f"\n{ups.shadow_name}:")
            for r in related[:5]:
                cond_short = r.condition[:50] + "..." if len(r.condition) > 50 else r.condition
                print(f"  = {r.value} when {cond_short}")
            if len(related) > 5:
                print(f"  ... and {len(related) - 5} more updates")


if __name__ == "__main__":
    test_simple_model()
    test_diode()
    test_psp102()
