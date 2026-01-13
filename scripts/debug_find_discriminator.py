#!/usr/bin/env python
"""Find the actual discriminating branch for a PHI node."""
import os
os.environ['JAX_ENABLE_X64'] = 'true'

from pathlib import Path
from collections import deque

import openvaf_py
from openvaf_jax.mir.types import parse_mir_function, ValueId
from openvaf_jax.mir.cfg import CFGAnalyzer
from openvaf_jax.mir.ssa import SSAAnalyzer

PROJECT_ROOT = Path(__file__).parent.parent
PSP102_VA = PROJECT_ROOT / "vendor" / "OpenVAF" / "integration_tests" / "PSP102" / "psp102_nqs.va"


def trace_predecessors_to_branches(mir_func, start_block: str, max_depth: int = 20):
    """Trace backwards from a block to find all branching predecessors."""
    branches_found = []
    visited = set()
    queue = deque([(start_block, 0)])

    while queue:
        block_name, depth = queue.popleft()
        if block_name in visited or depth > max_depth:
            continue
        visited.add(block_name)

        block = mir_func.blocks.get(block_name)
        if not block:
            continue

        for pred in block.predecessors:
            pred_block = mir_func.blocks.get(pred)
            if not pred_block:
                continue

            term = pred_block.terminator
            if term and term.is_branch:
                # This predecessor is a branching block
                is_true_target = term.true_block == block_name
                branches_found.append({
                    'branch_block': pred,
                    'condition': term.condition,
                    'true_target': term.true_block,
                    'false_target': term.false_block,
                    'reached_via': 'true' if is_true_target else 'false',
                    'depth': depth,
                })
            queue.append((pred, depth + 1))

    return branches_found


def main():
    print("Compiling PSP102...")
    modules = openvaf_py.compile_va(str(PSP102_VA))
    module = modules[0]

    mir_data = module.get_mir_instructions()
    str_constants = module.get_str_constants()
    mir_func = parse_mir_function("eval", mir_data, str_constants)

    cfg = CFGAnalyzer(mir_func)
    ssa = SSAAnalyzer(mir_func, cfg)

    # Target PHI predecessors
    preds_a = ['block3725', 'block3733']  # v3 = 0.0
    preds_b = ['block3742']  # computed value

    print("=" * 70)
    print("Tracing branches that lead to each PHI predecessor")
    print("=" * 70)

    # Trace branches leading to each predecessor
    for pred in preds_a + preds_b:
        print(f"\nPredecessor {pred}:")
        branches = trace_predecessors_to_branches(mir_func, pred)

        # Show first few branches
        for br in branches[:5]:
            marker = " (v3 group)" if pred in preds_a else " (computed group)"
            print(f"  depth={br['depth']}: {br['branch_block']} branches on {br['condition']}")
            print(f"         true→{br['true_target']}, false→{br['false_target']}")
            print(f"         reached via: {br['reached_via']} branch{marker}")

    # Now find a branch that discriminates between preds_a and preds_b
    print("\n" + "=" * 70)
    print("Finding discriminating branch")
    print("=" * 70)

    # Get all branches leading to each group
    branches_to_a = {}
    for pred in preds_a:
        for br in trace_predecessors_to_branches(mir_func, pred):
            key = br['branch_block']
            if key not in branches_to_a:
                branches_to_a[key] = {'via': set(), 'condition': br['condition']}
            branches_to_a[key]['via'].add(br['reached_via'])

    branches_to_b = {}
    for pred in preds_b:
        for br in trace_predecessors_to_branches(mir_func, pred):
            key = br['branch_block']
            if key not in branches_to_b:
                branches_to_b[key] = {'via': set(), 'condition': br['condition']}
            branches_to_b[key]['via'].add(br['reached_via'])

    # Find branches where preds_a comes from one branch and preds_b from the other
    print("\nLooking for exclusive discriminators:")

    for block_name in set(branches_to_a.keys()) | set(branches_to_b.keys()):
        a_via = branches_to_a.get(block_name, {}).get('via', set())
        b_via = branches_to_b.get(block_name, {}).get('via', set())
        cond = branches_to_a.get(block_name, branches_to_b.get(block_name, {})).get('condition', '?')

        # Check if exclusive
        if a_via and b_via and not a_via.intersection(b_via):
            print(f"\n  FOUND: {block_name} (condition={cond})")
            print(f"    preds_a (v3) reached via: {a_via}")
            print(f"    preds_b (computed) reached via: {b_via}")

            if a_via == {'false'} and b_via == {'true'}:
                print(f"    → For condition=TRUE: use computed value")
                print(f"    → For condition=FALSE: use v3 (0.0)")
            elif a_via == {'true'} and b_via == {'false'}:
                print(f"    → For condition=TRUE: use v3 (0.0)")
                print(f"    → For condition=FALSE: use computed value")

    # Check immediate predecessors of the PHI block (block3727)
    print("\n" + "=" * 70)
    print("Checking immediate predecessors of PHI block")
    print("=" * 70)

    phi_block = 'block3727'
    block = mir_func.blocks.get(phi_block)
    print(f"\n{phi_block} predecessors: {block.predecessors}")

    for pred in block.predecessors:
        pred_block = mir_func.blocks.get(pred)
        if pred_block:
            term = pred_block.terminator
            if term:
                if term.is_branch:
                    print(f"\n  {pred}: BRANCH on {term.condition}")
                    print(f"    true→{term.true_block}, false→{term.false_block}")
                elif term.is_jump:
                    print(f"\n  {pred}: JMP to {term.target_block}")
                    # Check what leads to this pred
                    for ppred in pred_block.predecessors:
                        ppred_block = mir_func.blocks.get(ppred)
                        if ppred_block and ppred_block.terminator and ppred_block.terminator.is_branch:
                            pterm = ppred_block.terminator
                            print(f"    (led by {ppred} branching on {pterm.condition})")


if __name__ == "__main__":
    main()
