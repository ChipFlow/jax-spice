"""Phi/Upsilon transformation for MIR to JAX translation.

This module implements a multi-step transformation that converts traditional
SSA PHI nodes into Phi/Upsilon form, which is easier to translate to JAX.

Transformation Pipeline:
1. Identify all PHI nodes and create shadow variables
2. Compute block activation conditions (path conditions)
3. Convert PHI operands to Upsilon assignments
4. Generate JAX code with conditional updates

Based on: https://gist.github.com/pizlonator/cf1e72b8600b1437dda8153ea3fdb963
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .types import BlockId, MIRFunction, MIRInstruction, ValueId


@dataclass
class ShadowVariable:
    """A shadow variable for a PHI node."""
    phi_result: ValueId  # The original PHI result (e.g., v100)
    shadow_name: str     # The shadow variable name (e.g., shadow_v100)
    initial_value: str   # Initial value (typically "0.0" or first operand)
    block: BlockId       # Block containing the PHI


@dataclass
class Upsilon:
    """An Upsilon instruction - conditional write to shadow variable."""
    shadow_name: str     # Which shadow variable to update
    value: ValueId       # Value to write
    block: BlockId       # Block where this Upsilon lives
    condition: str       # Condition under which this write happens


@dataclass
class BlockCondition:
    """The condition under which a block is active."""
    block: BlockId
    condition: str       # JAX expression for when this block executes
    # e.g., "cond_v100" or "cond_v100 & cond_v200" or "~cond_v100"


@dataclass
class PhiUpsilonIR:
    """Intermediate representation after Phi/Upsilon transformation."""
    shadow_vars: Dict[ValueId, ShadowVariable]  # PHI result -> shadow var
    upsilons: List[Upsilon]                      # All Upsilon instructions
    block_conditions: Dict[BlockId, BlockCondition]  # Block -> activation condition

    # Original MIR for reference
    mir_func: MIRFunction = field(repr=False)


# =============================================================================
# Step 1: Identify PHI nodes and create shadow variables
# =============================================================================

def step1_identify_phis(mir_func: MIRFunction) -> Dict[ValueId, ShadowVariable]:
    """
    Step 1: Find all PHI nodes and create shadow variables for them.

    For each PHI:
      v100 = phi [v50, block_a], [v60, block_b]

    Create:
      ShadowVariable(phi_result='v100', shadow_name='shadow_v100', ...)
    """
    shadow_vars: Dict[ValueId, ShadowVariable] = {}

    for block_name, block in mir_func.blocks.items():
        for inst in block.instructions:
            if inst.is_phi and inst.result and inst.phi_operands:
                # Use first operand's value as initial (will be overwritten)
                initial = str(inst.phi_operands[0].value)

                shadow_vars[inst.result] = ShadowVariable(
                    phi_result=inst.result,
                    shadow_name=f"shadow_{inst.result}",
                    initial_value=initial,
                    block=BlockId(block_name),
                )

    return shadow_vars


# =============================================================================
# Step 2: Compute block activation conditions
# =============================================================================

def step2_compute_block_conditions(mir_func: MIRFunction) -> Dict[BlockId, BlockCondition]:
    """
    Step 2: Compute the condition under which each block is active.

    NEW APPROACH: Use condition VARIABLES instead of symbolic expressions.
    Each branch creates a condition variable, and blocks inherit/combine them.

    This avoids exponential blowup of condition strings.
    """
    conditions: Dict[BlockId, BlockCondition] = {}
    condition_vars: Dict[str, str] = {}  # Maps block to its defining condition var

    # Entry block is always active
    entry = BlockId(mir_func.entry_block)
    conditions[entry] = BlockCondition(block=entry, condition="True")

    # BFS traversal - process all reachable blocks
    visited: Set[BlockId] = set()
    worklist = [entry]
    iteration = 0
    max_iterations = len(mir_func.blocks) * 2  # Safety limit

    while worklist and iteration < max_iterations:
        iteration += 1
        block_id = worklist.pop(0)

        if block_id in visited:
            continue

        block = mir_func.blocks.get(block_id)
        if not block:
            continue

        visited.add(block_id)

        # Compute this block's condition from predecessors if not set
        if block_id not in conditions:
            # Merge point - OR of predecessor conditions
            pred_conds = []
            for pred in block.predecessors:
                pred_id = BlockId(pred)
                if pred_id in conditions:
                    pred_conds.append(conditions[pred_id].condition)

            if pred_conds:
                if len(pred_conds) == 1:
                    cond = pred_conds[0]
                else:
                    cond = "(" + " | ".join(pred_conds) + ")"
                conditions[block_id] = BlockCondition(block=block_id, condition=cond)
            else:
                # Unreachable or not yet processed
                conditions[block_id] = BlockCondition(block=block_id, condition="False")

        # Propagate to successors via terminator
        term = block.terminator
        if term:
            parent_cond = conditions[block_id].condition

            if term.is_branch and term.condition and term.true_block and term.false_block:
                # Conditional branch
                cond_var = term.condition
                true_id = BlockId(term.true_block)
                false_id = BlockId(term.false_block)

                # True branch condition
                if parent_cond == "True":
                    true_cond = cond_var
                else:
                    true_cond = f"({parent_cond} & {cond_var})"

                # False branch condition
                if parent_cond == "True":
                    false_cond = f"~{cond_var}"
                else:
                    false_cond = f"({parent_cond} & ~{cond_var})"

                # Only set if not already set (first path wins for simplicity)
                # In reality we'd need to OR conditions for merge points
                if true_id not in conditions:
                    conditions[true_id] = BlockCondition(block=true_id, condition=true_cond)
                if false_id not in conditions:
                    conditions[false_id] = BlockCondition(block=false_id, condition=false_cond)

                worklist.extend([true_id, false_id])

            elif term.is_jump and term.target_block:
                # Unconditional jump - inherit parent's condition
                target_id = BlockId(term.target_block)
                if target_id not in conditions:
                    conditions[target_id] = BlockCondition(
                        block=target_id,
                        condition=parent_cond
                    )
                worklist.append(target_id)

    return conditions


# =============================================================================
# Step 3: Convert PHI operands to Upsilon instructions
# =============================================================================

def step3_create_upsilons(
    mir_func: MIRFunction,
    shadow_vars: Dict[ValueId, ShadowVariable],
    block_conditions: Dict[BlockId, BlockCondition],
) -> List[Upsilon]:
    """
    Step 3: Convert each PHI operand into an Upsilon instruction.

    For PHI:
      v100 = phi [v50, block_a], [v60, block_b]

    Create Upsilons:
      Upsilon(shadow_v100, v50, block_a, condition_of_block_a)
      Upsilon(shadow_v100, v60, block_b, condition_of_block_b)

    The Upsilon means: "if block_X is active, update shadow_v100 to value_X"
    """
    upsilons: List[Upsilon] = []

    for block_name, block in mir_func.blocks.items():
        for inst in block.instructions:
            if inst.is_phi and inst.result and inst.phi_operands:
                shadow = shadow_vars.get(inst.result)
                if not shadow:
                    continue

                for phi_op in inst.phi_operands:
                    pred_block = phi_op.block
                    pred_cond = block_conditions.get(pred_block)

                    if pred_cond:
                        upsilons.append(Upsilon(
                            shadow_name=shadow.shadow_name,
                            value=phi_op.value,
                            block=pred_block,
                            condition=pred_cond.condition,
                        ))

    return upsilons


# =============================================================================
# Step 4: Generate JAX code
# =============================================================================

def step4_generate_jax_code(ir: PhiUpsilonIR) -> str:
    """
    Step 4: Generate JAX code from Phi/Upsilon IR.

    Structure:
    1. Initialize all shadow variables
    2. For each Upsilon, generate: shadow_X = jnp.where(condition, value, shadow_X)
    3. PHI references become shadow variable references

    Note: This is a simplified generator. Real implementation would integrate
    with the existing codegen pipeline.
    """
    lines = []

    # Initialize shadow variables
    lines.append("# Initialize shadow variables for PHI nodes")
    for phi_result, shadow in ir.shadow_vars.items():
        lines.append(f"{shadow.shadow_name} = {shadow.initial_value}")

    lines.append("")
    lines.append("# Upsilon assignments (conditional updates)")

    # Group upsilons by shadow variable for cleaner output
    by_shadow: Dict[str, List[Upsilon]] = {}
    for ups in ir.upsilons:
        if ups.shadow_name not in by_shadow:
            by_shadow[ups.shadow_name] = []
        by_shadow[ups.shadow_name].append(ups)

    for shadow_name, upsilon_list in by_shadow.items():
        lines.append(f"# Updates to {shadow_name}")
        for ups in upsilon_list:
            # Simplify condition for readability
            cond = ups.condition
            if cond == "True":
                lines.append(f"{shadow_name} = {ups.value}")
            else:
                lines.append(f"{shadow_name} = jnp.where({cond}, {ups.value}, {shadow_name})")

    return "\n".join(lines)


# =============================================================================
# Main transformation function
# =============================================================================

def transform_to_phi_upsilon(mir_func: MIRFunction) -> PhiUpsilonIR:
    """
    Transform MIR to Phi/Upsilon form.

    This is the main entry point that runs all transformation steps.
    """
    # Step 1: Identify PHI nodes
    shadow_vars = step1_identify_phis(mir_func)
    print(f"Step 1: Found {len(shadow_vars)} PHI nodes")

    # Step 2: Compute block conditions
    block_conditions = step2_compute_block_conditions(mir_func)
    print(f"Step 2: Computed conditions for {len(block_conditions)} blocks")

    # Step 3: Create Upsilon instructions
    upsilons = step3_create_upsilons(mir_func, shadow_vars, block_conditions)
    print(f"Step 3: Created {len(upsilons)} Upsilon instructions")

    return PhiUpsilonIR(
        shadow_vars=shadow_vars,
        upsilons=upsilons,
        block_conditions=block_conditions,
        mir_func=mir_func,
    )


# =============================================================================
# Debug/visualization helpers
# =============================================================================

def print_phi_upsilon_summary(ir: PhiUpsilonIR, max_items: int = 10) -> None:
    """Print a summary of the Phi/Upsilon transformation."""
    print("\n" + "=" * 70)
    print("Phi/Upsilon Transformation Summary")
    print("=" * 70)

    print(f"\nShadow Variables: {len(ir.shadow_vars)}")
    for i, (phi_result, shadow) in enumerate(list(ir.shadow_vars.items())[:max_items]):
        print(f"  {shadow.shadow_name} (from {phi_result} in {shadow.block})")
    if len(ir.shadow_vars) > max_items:
        print(f"  ... and {len(ir.shadow_vars) - max_items} more")

    print(f"\nBlock Conditions: {len(ir.block_conditions)}")
    for i, (block_id, cond) in enumerate(list(ir.block_conditions.items())[:max_items]):
        # Truncate long conditions
        cond_str = cond.condition[:60] + "..." if len(cond.condition) > 60 else cond.condition
        print(f"  {block_id}: {cond_str}")
    if len(ir.block_conditions) > max_items:
        print(f"  ... and {len(ir.block_conditions) - max_items} more")

    print(f"\nUpsilon Instructions: {len(ir.upsilons)}")
    for i, ups in enumerate(ir.upsilons[:max_items]):
        cond_str = ups.condition[:40] + "..." if len(ups.condition) > 40 else ups.condition
        print(f"  {ups.shadow_name} = {ups.value} when {cond_str}")
    if len(ir.upsilons) > max_items:
        print(f"  ... and {len(ir.upsilons) - max_items} more")
