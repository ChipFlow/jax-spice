#!/usr/bin/env python
"""Debug PHI operand order for PSP102 - checking if v3 assignment is correct."""
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

    print("\n" + "=" * 70)
    print("Analyzing PHIs where v3 is assigned to TRUE branch")
    print("=" * 70)

    # Find PHIs where v3 ends up as true_value
    v3_as_true = []

    for block_name, block in mir_func.blocks.items():
        for phi in block.phi_nodes:
            resolution = ssa.resolve_phi(phi)
            if resolution.type == PHIResolutionType.TWO_WAY:
                if str(resolution.true_value) == 'v3':
                    v3_as_true.append({
                        'block': block_name,
                        'result': str(phi.result),
                        'condition': str(resolution.condition),
                        'true_value': str(resolution.true_value),
                        'false_value': str(resolution.false_value),
                        'operands': [(str(op.block), str(op.value)) for op in phi.phi_operands],
                    })

    print(f"\nFound {len(v3_as_true)} PHIs where v3 is assigned to TRUE branch")

    # Group by condition
    by_condition = {}
    for phi in v3_as_true:
        cond = phi['condition']
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(phi)

    print(f"\nGrouped by condition:")
    for cond, phis in sorted(by_condition.items(), key=lambda x: -len(x[1])):
        print(f"  {cond}: {len(phis)} PHIs")

    # Show samples for v597553 (TYPE check)
    type_check_phis = by_condition.get('v597553', [])
    if type_check_phis:
        print(f"\n" + "-" * 70)
        print(f"Detailed analysis of v597553 (TYPE check) PHIs ({len(type_check_phis)} total):")
        print("-" * 70)

        for phi in type_check_phis[:5]:
            print(f"\n  Block: {phi['block']}, Result: {phi['result']}")
            print(f"  PHI operands (in MIR order):")
            for i, (pred, val) in enumerate(phi['operands']):
                marker = " <-- v3 (0.0)" if val == 'v3' else ""
                print(f"    [{i}] from {pred}: {val}{marker}")
            print(f"  Resolution:")
            print(f"    condition: {phi['condition']}")
            print(f"    true_value: {phi['true_value']}")
            print(f"    false_value: {phi['false_value']}")

    # Now let's trace WHY v3 ends up as true_value
    print(f"\n" + "=" * 70)
    print("Tracing resolution logic for first problematic PHI")
    print("=" * 70)

    if type_check_phis:
        phi_info = type_check_phis[0]
        block_name = phi_info['block']
        result = phi_info['result']

        # Find the actual PHI
        block = mir_func.blocks[block_name]
        target_phi = None
        for phi in block.phi_nodes:
            if str(phi.result) == result:
                target_phi = phi
                break

        if target_phi:
            ops = target_phi.phi_operands
            pred0, pred1 = ops[0].block, ops[1].block
            val0, val1 = ops[0].value, ops[1].value

            print(f"\nPHI: {result}")
            print(f"  pred0={pred0}, val0={val0}")
            print(f"  pred1={pred1}, val1={val1}")

            # Check what the resolution logic sees
            branch_conds = ssa.branch_conditions

            print(f"\n  Checking Strategy 1 (predecessor is branch block):")
            print(f"    pred0={pred0} in branch_conds: {pred0 in branch_conds}")
            print(f"    pred1={pred1} in branch_conds: {pred1 in branch_conds}")

            print(f"\n  Checking Strategy 2 (succ_pair_map):")
            pred_key = frozenset([pred0, pred1])
            candidates = ssa.succ_pair_map.get(pred_key, [])
            print(f"    succ_pair_map[{pred0}, {pred1}] = {candidates}")

            print(f"\n  Checking Strategy 3 (ancestor trace):")
            # Find entry block's branch (TYPE check)
            entry = mir_func.entry_block
            entry_block = mir_func.blocks.get(entry)
            if entry_block and entry_block.terminator:
                term = entry_block.terminator
                print(f"    Entry block: {entry}")
                print(f"    Branch condition: {term.condition}")
                print(f"    True target: {term.true_block}")
                print(f"    False target: {term.false_block}")

                # Check reachability
                true_target = term.true_block
                false_target = term.false_block

                pred0_from_true = ssa._is_reachable(true_target, pred0)
                pred0_from_false = ssa._is_reachable(false_target, pred0)
                pred1_from_true = ssa._is_reachable(true_target, pred1)
                pred1_from_false = ssa._is_reachable(false_target, pred1)

                print(f"\n    Reachability analysis for {pred0} and {pred1}:")
                print(f"      pred0 ({pred0}) from true ({true_target}): {pred0_from_true}")
                print(f"      pred0 ({pred0}) from false ({false_target}): {pred0_from_false}")
                print(f"      pred1 ({pred1}) from true ({true_target}): {pred1_from_true}")
                print(f"      pred1 ({pred1}) from false ({false_target}): {pred1_from_false}")

                # What we WANT for correct resolution:
                # If val0 is v3 (0.0) and v3 is the "unused" PMOS path value:
                #   - pred0 should be reachable ONLY from PMOS path (false)
                #   - pred1 should be reachable ONLY from NMOS path (true)
                #   - Then: true_value=val1 (computed), false_value=val0 (v3)
                #
                # But if val0 is v3 and the resolution puts it as true_value:
                #   - That means the logic thinks pred0 is reachable from NMOS (true)

                print(f"\n    Current resolution:")
                print(f"      val0={val0} is assigned to: {phi_info['true_value'] if str(val0) == phi_info['true_value'] else phi_info['false_value']}")
                print(f"      val1={val1} is assigned to: {phi_info['true_value'] if str(val1) == phi_info['true_value'] else phi_info['false_value']}")

                if str(val0) == 'v3' and phi_info['true_value'] == 'v3':
                    print(f"\n    BUG: v3 (0.0) is assigned to TRUE branch!")
                    print(f"      For NMOS (TYPE=1), condition is TRUE, so we get 0.0")
                    print(f"      But v3 should be the PMOS path's default value")


if __name__ == "__main__":
    main()
