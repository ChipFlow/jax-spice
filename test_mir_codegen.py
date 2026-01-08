#!/usr/bin/env python3
"""Test MIR→Python code generation with resistor."""

import sys
sys.path.insert(0, '.')

from jax_spice.codegen.mir_parser import parse_mir_function
from jax_spice.codegen.python_codegen import generate_python_eval, analyze_mir_semantics

# Resistor eval MIR from OpenVAF dump
RESISTOR_EVAL_MIR = """
Optimized evaluation MIR of resistor
function %(v16, v17, v20, v21, v27, v51, v48, v70, v73, v53) {
    inst0 = const fn %white_noise(Spur(1))(1) -> 1

                                block5:
@0006                               v18 = fdiv v16, v17
                                    v43 = optbarrier v18
                                    v50 = optbarrier v48
                                    v65 = fmul v51, v18
                                    v54 = optbarrier v65
                                    v55 = fneg v18
                                    v62 = optbarrier v70
                                    v64 = optbarrier v73
                                    v67 = fmul v51, v55
                                    v66 = optbarrier v67
                                    v68 = optbarrier v51
                                    v69 = optbarrier v53
                                    v71 = optbarrier v73
                                    v74 = optbarrier v70
}
"""

def main():
    print("=" * 80)
    print("MIR → Python Code Generation Test")
    print("=" * 80)

    # Parse MIR
    print("\n1. Parsing MIR...")
    mir_func = parse_mir_function(RESISTOR_EVAL_MIR)

    print(f"   Function: {mir_func.name}")
    print(f"   Parameters: {', '.join(p.name for p in mir_func.params)}")
    print(f"   Blocks: {len(mir_func.blocks)}")
    print(f"   Instructions: {sum(len(b.instructions) for b in mir_func.blocks)}")

    # Analyze semantics
    print("\n2. Analyzing semantic mapping...")
    param_map = analyze_mir_semantics(mir_func)
    print("   MIR → Semantic mapping:")
    for mir_name, sem_name in sorted(param_map.items()):
        print(f"     {mir_name} → {sem_name}")

    # Generate Python code
    print("\n3. Generating Python code...")
    python_code = generate_python_eval(mir_func, param_map)

    print("\n" + "=" * 80)
    print("Generated Python Code:")
    print("=" * 80)
    print(python_code)
    print("=" * 80)

    # Test if it compiles
    print("\n4. Testing if generated code compiles...")
    try:
        compile(python_code, '<generated>', 'exec')
        print("   ✓ Code compiles successfully!")
    except SyntaxError as e:
        print(f"   ✗ Syntax error: {e}")
        return 1

    # Execute it
    print("\n5. Testing execution...")
    namespace = {}
    exec(python_code, namespace)

    # Call the generated function
    eval_fn = namespace['eval_device']
    try:
        # Test with simple values: V=1V, R=1000Ω
        result = eval_fn(
            V_br=1.0,           # v16
            r=1000.0,           # v17
            v20=0.0,            # Unknown param
            v21=0.0,            # Unknown param
            v27=0.0,            # Unknown param
            mfactor=1.0,        # v51
            v48=0.0,            # Unknown param
            v70=0.0,            # Unknown param
            v73=0.0,            # Unknown param
            v53=0.0,            # Unknown param
        )
        print(f"   ✓ Execution succeeded!")
        print(f"   Result: {result}")

        # Check if computed current makes sense
        # Expected: I = V/R = 1.0/1000.0 = 0.001 A
        print("\n6. Validating physics...")
        expected_I = 1.0 / 1000.0
        actual_I = result.get('I')
        print(f"   Expected current: {expected_I} A")
        print(f"   Actual current:   {actual_I} A")

        if actual_I is not None and abs(actual_I - expected_I) < 1e-12:
            print("   ✓ Physics validation PASSED!")
        else:
            print(f"   ✗ Physics validation FAILED!")
            return 1

        # Check Jacobian (dI/dV should be 1/R)
        expected_dI_dV = 1.0 / 1000.0
        print(f"\n   Expected dI/dV: {expected_dI_dV} S (Siemens)")
        print(f"   Note: MIR doesn't explicitly compute dI/dV in eval")
        print(f"         (autodiff generates separate Jacobian code)")
        print(f"   I_neg = {result.get('I_neg')} A (for Jacobian assembly)")

    except Exception as e:
        print(f"   ✗ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("SUCCESS! MIR→Python codegen working!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
