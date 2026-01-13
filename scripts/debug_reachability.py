#!/usr/bin/env python
"""Debug reachability for PSP102 PHI resolution."""
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

    # Find the problematic PHI from previous debug
    block = mir_func.blocks['block3727']
    target_phi = None
    for phi in block.phi_nodes:
        if str(phi.result) == 'v692514':
            target_phi = phi
            break

    if not target_phi:
        print("Could not find target PHI")
        return

    print(f"\nTarget PHI: {target_phi.result}")
    print(f"Operands:")
    for op in target_phi.phi_operands:
        print(f"  from {op.block}: {op.value}")

    # Group by value
    val_to_preds = {}
    for op in target_phi.phi_operands:
        val = str(op.value)
        if val not in val_to_preds:
            val_to_preds[val] = []
        val_to_preds[val].append(op.block)

    print(f"\nGrouped by value:")
    for val, preds in val_to_preds.items():
        print(f"  {val}: {preds}")

    # This is a 3-operand PHI with 2 unique values
    preds_a = val_to_preds.get('v3', [])  # v3 = 0.0
    preds_b = val_to_preds.get('v643048', [])  # computed value

    print(f"\npreds_a (v3=0.0): {preds_a}")
    print(f"preds_b (computed): {preds_b}")

    # Get entry block branch info
    entry = mir_func.entry_block
    entry_block = mir_func.blocks.get(entry)
    term = entry_block.terminator

    print(f"\nEntry block: {entry}")
    print(f"Branch condition: {term.condition} (TYPE check)")
    print(f"True target: {term.true_block} (NMOS path)")
    print(f"False target: {term.false_block} (PMOS path)")

    nmos_block = term.true_block  # block3710
    pmos_block = term.false_block  # block3711

    # Check reachability using _any_reachable (no depth limit)
    print(f"\n" + "=" * 70)
    print("Reachability analysis using _any_reachable (no depth limit):")
    print("=" * 70)

    nmos_reaches_a = ssa._any_reachable(nmos_block, preds_a)
    nmos_reaches_b = ssa._any_reachable(nmos_block, preds_b)
    pmos_reaches_a = ssa._any_reachable(pmos_block, preds_a)
    pmos_reaches_b = ssa._any_reachable(pmos_block, preds_b)

    print(f"\nNMOS ({nmos_block}) reaches preds_a (v3 group): {nmos_reaches_a}")
    print(f"NMOS ({nmos_block}) reaches preds_b (computed group): {nmos_reaches_b}")
    print(f"PMOS ({pmos_block}) reaches preds_a (v3 group): {pmos_reaches_a}")
    print(f"PMOS ({pmos_block}) reaches preds_b (computed group): {pmos_reaches_b}")

    # Now let's trace what _find_condition_for_groups does
    print(f"\n" + "=" * 70)
    print("Tracing _find_condition_for_groups logic:")
    print("=" * 70)

    # This is what the function checks:
    # For each branch block (including entry), check:
    #   t0_reaches_a = t0 in preds_a or any_reachable(t0, preds_a)
    #   t1_reaches_b = t1 in preds_b or any_reachable(t1, preds_b)
    #
    # If t0_reaches_a and t1_reaches_b:
    #   return cond_var if is_t0_true else f"!{cond_var}"
    #   (and true_value = val_a, false_value = val_b)

    # Check for entry block
    t0, t1 = nmos_block, pmos_block
    t0_in_a = t0 in preds_a
    t0_reaches_a = t0_in_a or ssa._any_reachable(t0, preds_a)
    t1_in_b = t1 in preds_b
    t1_reaches_b = t1_in_b or ssa._any_reachable(t1, preds_b)

    print(f"\nFor entry block ({entry}):")
    print(f"  t0 ({t0}) in preds_a: {t0_in_a}")
    print(f"  t0 ({t0}) reaches preds_a: {t0_reaches_a}")
    print(f"  t1 ({t1}) in preds_b: {t1_in_b}")
    print(f"  t1 ({t1}) reaches preds_b: {t1_reaches_b}")

    # Check vice versa
    t0_in_b = t0 in preds_b
    t0_reaches_b = t0_in_b or ssa._any_reachable(t0, preds_b)
    t1_in_a = t1 in preds_a
    t1_reaches_a = t1_in_a or ssa._any_reachable(t1, preds_a)

    print(f"\n  t0 ({t0}) in preds_b: {t0_in_b}")
    print(f"  t0 ({t0}) reaches preds_b: {t0_reaches_b}")
    print(f"  t1 ({t1}) in preds_a: {t1_in_a}")
    print(f"  t1 ({t1}) reaches preds_a: {t1_reaches_a}")

    print(f"\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)

    if t0_reaches_a and t1_reaches_b:
        print(f"\nCurrent logic: t0→preds_a and t1→preds_b")
        print(f"  → condition = {term.condition} (because t0 is true branch)")
        print(f"  → true_value = val_a = v3 (0.0)")
        print(f"  → false_value = val_b = computed")
        print(f"\n  BUG: For NMOS (TYPE=1, condition TRUE), we get 0.0!")
        print(f"       Should get computed value.")

    if t0_reaches_b and t1_reaches_a:
        print(f"\nAlternative: t0→preds_b and t1→preds_a")
        print(f"  → condition = !{term.condition}")
        print(f"  → true_value = val_b = computed")
        print(f"  → false_value = val_a = v3 (0.0)")
        print(f"\n  CORRECT: For NMOS (TYPE=1, condition TRUE → negated FALSE), we get computed!")

    # Check if BOTH paths can reach BOTH groups (this would be a problem)
    print(f"\n" + "=" * 70)
    print("Checking if paths are exclusive:")
    print("=" * 70)

    print(f"\nNMOS path can reach:")
    print(f"  v3 group: {nmos_reaches_a}")
    print(f"  computed group: {nmos_reaches_b}")

    print(f"\nPMOS path can reach:")
    print(f"  v3 group: {pmos_reaches_a}")
    print(f"  computed group: {pmos_reaches_b}")

    if nmos_reaches_a and nmos_reaches_b:
        print(f"\n*** PROBLEM: NMOS path reaches BOTH groups!")
        print(f"    This makes the TYPE check useless as a discriminator.")
    if pmos_reaches_a and pmos_reaches_b:
        print(f"\n*** PROBLEM: PMOS path reaches BOTH groups!")
        print(f"    This makes the TYPE check useless as a discriminator.")


if __name__ == "__main__":
    main()
