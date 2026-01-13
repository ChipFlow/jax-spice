#!/usr/bin/env python
"""Trace FALLBACK PHIs to understand what values they're returning."""
import os
os.environ['JAX_ENABLE_X64'] = 'true'

from pathlib import Path

import openvaf_py
from openvaf_jax.mir.types import parse_mir_function, ValueId, V_F_ZERO
from openvaf_jax.mir.cfg import CFGAnalyzer
from openvaf_jax.mir.ssa import SSAAnalyzer, PHIResolutionType

PROJECT_ROOT = Path(__file__).parent.parent
PSP102_VA = PROJECT_ROOT / "vendor" / "OpenVAF" / "integration_tests" / "PSP102" / "psp102_nqs.va"


def main():
    print("Compiling PSP102...")
    modules = openvaf_py.compile_va(str(PSP102_VA))
    module = modules[0]

    mir_data = module.get_mir_instructions()
    str_constants = module.get_str_constants()
    mir_func = parse_mir_function("eval", mir_data, str_constants)

    cfg = CFGAnalyzer(mir_func)
    ssa = SSAAnalyzer(mir_func, cfg)

    # Get all constants (including pre-allocated)
    all_constants = set(mir_func.constants.keys())
    all_constants.add('v3')  # Pre-allocated 0.0
    all_constants.add('v6')  # Pre-allocated 1.0
    all_constants.add('v7')  # Pre-allocated -1.0

    print("\n" + "=" * 70)
    print("Analyzing FALLBACK PHIs")
    print("=" * 70)

    fallback_phis = []
    for block_name, block in mir_func.blocks.items():
        for phi in block.phi_nodes:
            resolution = ssa.resolve_phi(phi)
            if resolution.type == PHIResolutionType.FALLBACK:
                fallback_phis.append({
                    'block': block_name,
                    'result': str(phi.result),
                    'single_value': str(resolution.single_value),
                    'operands': [(str(op.block), str(op.value)) for op in phi.phi_operands],
                    'is_constant': str(resolution.single_value) in all_constants,
                })

    print(f"\nTotal FALLBACK PHIs: {len(fallback_phis)}")

    # Group by whether fallback value is a constant
    constant_fallbacks = [p for p in fallback_phis if p['is_constant']]
    variable_fallbacks = [p for p in fallback_phis if not p['is_constant']]

    print(f"  Fallback to constant: {len(constant_fallbacks)}")
    print(f"  Fallback to variable: {len(variable_fallbacks)}")

    # Check if variable fallbacks reference defined values
    print("\n" + "-" * 70)
    print("Variable fallbacks (these might cause issues if not defined):")
    print("-" * 70)

    for phi in variable_fallbacks[:10]:
        val = phi['single_value']
        # Check if this value is defined elsewhere
        defining_inst = mir_func.get_instruction_by_result(ValueId(val))
        if defining_inst:
            print(f"\n  {phi['result']} = fallback to {val}")
            print(f"    Defined by: {defining_inst.opcode} in {defining_inst.block}")
        else:
            print(f"\n  {phi['result']} = fallback to {val}")
            print(f"    WARNING: {val} has no defining instruction!")

    # Check for PHIs that reference other PHIs with FALLBACK resolution
    print("\n" + "=" * 70)
    print("Checking for PHI chains (PHI referencing FALLBACK PHI)")
    print("=" * 70)

    fallback_results = {p['result'] for p in fallback_phis}

    phi_chains = []
    for block_name, block in mir_func.blocks.items():
        for phi in block.phi_nodes:
            for op in phi.phi_operands:
                if str(op.value) in fallback_results:
                    phi_chains.append({
                        'phi': str(phi.result),
                        'block': block_name,
                        'references_fallback': str(op.value),
                    })

    print(f"\nPHIs referencing FALLBACK results: {len(phi_chains)}")
    for chain in phi_chains[:5]:
        print(f"  {chain['phi']} in {chain['block']} references fallback {chain['references_fallback']}")


if __name__ == "__main__":
    main()
