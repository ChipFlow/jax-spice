#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["jax-spice"]
# ///
"""Debug PHI resolution for PSP102 model.

Analyzes how many PHI nodes are resolved with each strategy to understand
why JAX produces wrong results for PSP102.
"""
import os
os.environ['JAX_ENABLE_X64'] = 'true'

from pathlib import Path
from collections import Counter

import openvaf_py
import openvaf_jax
from openvaf_jax.mir.types import parse_mir_function, ValueId
from openvaf_jax.mir.cfg import CFGAnalyzer
from openvaf_jax.mir.ssa import SSAAnalyzer, PHIResolutionType

PROJECT_ROOT = Path(__file__).parent.parent
PSP102_VA = PROJECT_ROOT / "vendor" / "OpenVAF" / "integration_tests" / "PSP102" / "psp102_nqs.va"


def analyze_phi_resolution(va_path: Path, func_name: str = "eval"):
    """Analyze PHI resolution strategies for a model."""
    print(f"Compiling {va_path.name}...")
    modules = openvaf_py.compile_va(str(va_path))
    module = modules[0]

    print(f"Getting MIR for {func_name}...")
    if func_name == "eval":
        mir_data = module.get_mir_instructions()
    else:
        mir_data = module.get_init_mir_instructions()
    str_constants = module.get_str_constants()
    mir_func = parse_mir_function(func_name, mir_data, str_constants)

    print(f"Building CFG and SSA analyzer...")
    cfg = CFGAnalyzer(mir_func)
    ssa = SSAAnalyzer(mir_func, cfg)

    # Count resolution types
    resolution_counts = Counter()
    fallback_phis = []
    two_way_phis = []
    multi_way_phis = []

    print(f"\nAnalyzing {len(mir_func.blocks)} blocks...")

    total_phis = 0
    for block_name, block in mir_func.blocks.items():
        for phi in block.phi_nodes:
            total_phis += 1
            resolution = ssa.resolve_phi(phi)
            resolution_counts[resolution.type.name] += 1

            if resolution.type == PHIResolutionType.FALLBACK:
                fallback_phis.append({
                    'block': block_name,
                    'result': phi.result,
                    'operands': [(op.block, op.value) for op in phi.phi_operands],
                    'fallback_value': resolution.single_value,
                })
            elif resolution.type == PHIResolutionType.TWO_WAY:
                two_way_phis.append({
                    'block': block_name,
                    'result': phi.result,
                    'condition': resolution.condition,
                    'true_value': resolution.true_value,
                    'false_value': resolution.false_value,
                })
            elif resolution.type == PHIResolutionType.MULTI_WAY:
                multi_way_phis.append({
                    'block': block_name,
                    'result': phi.result,
                    'cases': resolution.cases,
                    'default': resolution.default,
                    'num_preds': len(phi.phi_operands),
                })

    print(f"\n{'='*60}")
    print(f"PHI Resolution Summary for {va_path.name} ({func_name})")
    print(f"{'='*60}")
    print(f"Total PHI nodes: {total_phis}")
    print(f"\nResolution type breakdown:")
    for res_type, count in sorted(resolution_counts.items()):
        pct = 100 * count / total_phis if total_phis > 0 else 0
        print(f"  {res_type:15s}: {count:4d} ({pct:5.1f}%)")

    if fallback_phis:
        print(f"\n{'='*60}")
        print(f"FALLBACK PHIs ({len(fallback_phis)}) - These may be causing issues:")
        print(f"{'='*60}")

        # Group by fallback value
        fallback_by_value = Counter()
        for phi in fallback_phis:
            fallback_by_value[str(phi['fallback_value'])] += 1

        print(f"\nFallback value distribution:")
        for val, count in fallback_by_value.most_common(10):
            print(f"  {val}: {count} PHIs")

        # Show first few FALLBACK PHIs with v3 (0.0) as fallback
        v3_fallbacks = [p for p in fallback_phis if str(p['fallback_value']) == 'v3']
        if v3_fallbacks:
            print(f"\n\nFirst 5 PHIs with v3 (0.0) as fallback value:")
            for phi in v3_fallbacks[:5]:
                print(f"\n  Block: {phi['block']}")
                print(f"  Result: {phi['result']}")
                print(f"  Operands:")
                for pred, val in phi['operands']:
                    marker = " <-- v3 (0.0)" if str(val) == 'v3' else ""
                    print(f"    from {pred}: {val}{marker}")

    # Check for PHIs where v3 appears but isn't the fallback (correctly resolved)
    v3_in_resolved = 0
    v3_as_true = 0
    v3_as_false = 0
    sample_v3_phis = []

    for phi in two_way_phis:
        has_v3_true = str(phi['true_value']) == 'v3'
        has_v3_false = str(phi['false_value']) == 'v3'
        if has_v3_true or has_v3_false:
            v3_in_resolved += 1
            if has_v3_true:
                v3_as_true += 1
            if has_v3_false:
                v3_as_false += 1
            if len(sample_v3_phis) < 5:
                sample_v3_phis.append(phi)

    print(f"\n{'='*60}")
    print(f"v3 (0.0) Analysis:")
    print(f"{'='*60}")
    print(f"PHIs with v3 resolved via TWO_WAY: {v3_in_resolved}")
    print(f"  - v3 as TRUE value: {v3_as_true}")
    print(f"  - v3 as FALSE value: {v3_as_false}")
    print(f"PHIs with v3 as FALLBACK value: {len([p for p in fallback_phis if str(p['fallback_value']) == 'v3'])}")

    if sample_v3_phis:
        print(f"\n\nSample TWO_WAY PHIs with v3:")
        for phi in sample_v3_phis:
            print(f"\n  Block: {phi['block']}, Result: {phi['result']}")
            print(f"  Condition: {phi['condition']}")
            print(f"  True value: {phi['true_value']}")
            print(f"  False value: {phi['false_value']}")

    # Count unique conditions used for v3 PHIs
    v3_conditions = Counter()
    for phi in two_way_phis:
        if str(phi['true_value']) == 'v3' or str(phi['false_value']) == 'v3':
            v3_conditions[str(phi['condition'])] += 1

    print(f"\n\nCondition variable distribution for v3 PHIs:")
    for cond, count in v3_conditions.most_common(10):
        print(f"  {cond}: {count} PHIs")

    # Analyze MULTI_WAY PHIs
    print(f"\n{'='*60}")
    print(f"MULTI_WAY PHI Analysis ({len(multi_way_phis)} total)")
    print(f"{'='*60}")

    # Count by number of cases
    cases_count = Counter()
    v3_in_cases = 0
    v3_as_default = 0
    sample_multi_phis = []

    for phi in multi_way_phis:
        num_cases = len(phi['cases']) if phi['cases'] else 0
        cases_count[num_cases] += 1

        # Check for v3 in cases or default
        if phi['cases']:
            for cond, val in phi['cases']:
                if str(val) == 'v3':
                    v3_in_cases += 1
                    break
        if str(phi['default']) == 'v3':
            v3_as_default += 1

        if len(sample_multi_phis) < 3:
            sample_multi_phis.append(phi)

    print(f"\nNumber of cases distribution:")
    for num_cases, count in sorted(cases_count.items()):
        print(f"  {num_cases} cases: {count} PHIs")

    print(f"\nv3 (0.0) in MULTI_WAY:")
    print(f"  PHIs with v3 in cases: {v3_in_cases}")
    print(f"  PHIs with v3 as default: {v3_as_default}")

    if sample_multi_phis:
        print(f"\n\nSample MULTI_WAY PHIs:")
        for phi in sample_multi_phis:
            print(f"\n  Block: {phi['block']}, Result: {phi['result']}")
            print(f"  Predecessors: {phi['num_preds']}")
            print(f"  Cases ({len(phi['cases']) if phi['cases'] else 0}):")
            if phi['cases']:
                for cond, val in phi['cases'][:5]:
                    print(f"    if {cond}: {val}")
            print(f"  Default: {phi['default']}")

    # Show sample MULTI_WAY PHIs with v3 as default
    v3_default_phis = [p for p in multi_way_phis if str(p['default']) == 'v3']
    print(f"\n\nSample MULTI_WAY PHIs with v3 as default ({len(v3_default_phis)} total):")
    for phi in v3_default_phis[:5]:
        print(f"\n  Block: {phi['block']}, Result: {phi['result']}")
        print(f"  Predecessors: {phi['num_preds']}")
        print(f"  Cases ({len(phi['cases']) if phi['cases'] else 0}):")
        if phi['cases']:
            for cond, val in phi['cases'][:10]:
                print(f"    if {cond}: {val}")
        print(f"  Default: {phi['default']} (0.0)")

    return resolution_counts, fallback_phis


def analyze_first_branch(va_path: Path):
    """Analyze the first branch (NMOS/PMOS selection) in PSP102."""
    print(f"\n{'='*60}")
    print("Analyzing entry block branch (TYPE check)")
    print(f"{'='*60}")

    modules = openvaf_py.compile_va(str(va_path))
    module = modules[0]
    mir_data = module.get_mir_instructions()
    str_constants = module.get_str_constants()
    mir_func = parse_mir_function("eval", mir_data, str_constants)

    # Find entry block
    entry = mir_func.entry_block
    print(f"Entry block: {entry}")

    entry_block = mir_func.blocks.get(entry)
    if entry_block:
        term = entry_block.terminator
        if term and term.is_branch:
            print(f"Branch condition: {term.condition}")
            print(f"True target: {term.true_block}")
            print(f"False target: {term.false_block}")

            # Trace how the condition is computed
            cond_id = term.condition
            print(f"\nTracing condition {cond_id}...")

            # Find instruction that produces this condition
            cond_inst = mir_func.get_instruction_by_result(ValueId(cond_id))
            if cond_inst:
                print(f"  Condition instruction: {cond_inst.opcode}")
                print(f"  Operands: {cond_inst.operands}")

                # If it's a comparison, trace the operands
                if cond_inst.operands:
                    for op in cond_inst.operands:
                        op_inst = mir_func.get_instruction_by_result(ValueId(op))
                        if op_inst:
                            print(f"    {op}: {op_inst.opcode} {op_inst.operands}")

            print(f"\nThis branch likely checks: TYPE == 1 (NMOS) vs TYPE == -1 (PMOS)")

    return mir_func


if __name__ == "__main__":
    if not PSP102_VA.exists():
        print(f"PSP102 model not found at {PSP102_VA}")
        exit(1)

    analyze_first_branch(PSP102_VA)
    resolution_counts, fallback_phis = analyze_phi_resolution(PSP102_VA, "eval")
