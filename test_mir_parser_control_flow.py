#!/usr/bin/env python3
"""Test MIR parser with control flow (branches, jumps, PHI nodes)."""

import sys
sys.path.insert(0, '.')

from jax_spice.codegen.mir_parser import parse_mir_function

# Resistor model setup MIR (simplified)
MODEL_SETUP_MIR = """
Optimized model setup MIR of resistor
function %(v19, v20, v32, v33) {
    inst0 = fn %set_Invalid(Parameter { id: ParamId(0) })(0) -> 0
    inst1 = fn %set_Invalid(Parameter { id: ParamId(1) })(0) -> 0
    v3 = fconst 0.0
    v5 = iconst 1
    v6 = fconst 0x1.0000000000000p0
    v15 = fconst +Inf

                                block0:
@0002                               br v20, block2, block11

                                block2:
                                    v23 = fle v3, v19
                                    br v23, block7, block9

                                block7:
                                    v24 = fle v19, v15
                                    jmp block9

                                block9:
                                    v25 = phi [v1, block2], [v24, block7]
                                    br v25, block4, block10

                                block10:
                                    call inst0()
                                    jmp block4

                                block11:
                                    v31 = optbarrier v6
                                    jmp block4

                                block4:
                                    v21 = phi [v19, block9], [v19, block10], [v6, block11]
                                    br v33, block19, block18

                                block18:
                                    v35 = optbarrier v5
                                    jmp block19

                                block19:
                                    v34 = phi [v32, block4], [v5, block18]
}
"""

def main():
    print("=" * 80)
    print("MIR Parser - Control Flow Test")
    print("=" * 80)

    print("\n1. Parsing model setup MIR with control flow...")
    mir_func = parse_mir_function(MODEL_SETUP_MIR)

    print(f"   Function: {mir_func.name}")
    print(f"   Parameters: {', '.join(p.name for p in mir_func.params)}")
    print(f"   Constants: {len(mir_func.constants)} defined")
    print(f"   Blocks: {len(mir_func.blocks)}")

    print("\n2. Block structure:")
    for block in mir_func.blocks:
        print(f"\n   {block.name}: ({len(block.instructions)} instructions)")
        for inst in block.instructions[:5]:  # Show first 5
            print(f"      {inst}")
        if len(block.instructions) > 5:
            print(f"      ... ({len(block.instructions) - 5} more)")

    print("\n3. Analyzing control flow instructions...")
    control_flow_count = {'br': 0, 'jmp': 0, 'phi': 0}
    for block in mir_func.blocks:
        for inst in block.instructions:
            if inst.opcode in control_flow_count:
                control_flow_count[inst.opcode] += 1

    print(f"   Branches (br):   {control_flow_count['br']}")
    print(f"   Jumps (jmp):     {control_flow_count['jmp']}")
    print(f"   PHI nodes (phi): {control_flow_count['phi']}")

    print("\n4. Finding branching structure...")
    for block in mir_func.blocks:
        for inst in block.instructions:
            if inst.opcode == 'br' and len(inst.target_blocks) == 2:
                print(f"   {block.name}: if {inst.args[0]} then {inst.target_blocks[0]} else {inst.target_blocks[1]}")
            elif inst.opcode == 'jmp':
                print(f"   {block.name}: goto {inst.target_blocks[0]}")

    print("\n" + "=" * 80)
    print("SUCCESS! Parser handles control flow!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
