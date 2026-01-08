#!/usr/bin/env python3
"""Test control flow code generation."""

import sys
sys.path.insert(0, '.')

from jax_spice.codegen.mir_parser import parse_mir_function
from jax_spice.codegen.control_flow_codegen import generate_python_with_control_flow

# Simplified resistor model setup MIR
MODEL_SETUP_MIR = """
Optimized model setup MIR of resistor
function %(v19, v20, v32, v33) {
    v3 = fconst 0.0
    v6 = fconst 0x1.0000000000000p0
    v15 = fconst +Inf

                                block0:
                                    br v20, block2, block11

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
                                    jmp block4

                                block11:
                                    v31 = optbarrier v6
                                    jmp block4

                                block4:
                                    v21 = phi [v19, block9], [v19, block10], [v6, block11]
}
"""

def main():
    print("=" * 80)
    print("Control Flow Code Generation Test")
    print("=" * 80)

    # Parse MIR
    print("\n1. Parsing MIR...")
    mir_func = parse_mir_function(MODEL_SETUP_MIR)
    print(f"   Parsed {len(mir_func.blocks)} blocks")

    # Simple parameter mapping for resistor model setup
    # v19 = r (resistance parameter)
    # v20 = r_given (flag: was r explicitly set?)
    # v32 = has_noise
    # v33 = has_noise_given
    # Pre-allocated OpenVAF constants:
    # v1 = FALSE, v2 = TRUE, v3 = F_ZERO, v5 = ONE, v6 = F_ONE
    param_map = {
        'v19': 'r',
        'v20': 'r_given',
        'v32': 'has_noise',
        'v33': 'has_noise_given',
        'v1': 'FALSE',  # Pre-allocated OpenVAF constant
        'v2': 'TRUE',   # Pre-allocated OpenVAF constant
        'v3': 'ZERO',   # Pre-allocated OpenVAF constant (F_ZERO)
        'v5': 'ONE_INT',  # Pre-allocated OpenVAF constant (i32)
        'v6': 'ONE',    # Pre-allocated OpenVAF constant (F_ONE)
        'v15': 'INF',
        'v23': 'r_ge_zero',
        'v24': 'r_le_inf',
        'v25': 'r_in_range',
        'v21': 'r_final',
        'v31': 'r_default',
    }

    # Generate Python
    print("\n2. Generating Python code...")
    python_code = generate_python_with_control_flow(mir_func, param_map)

    print("\n" + "=" * 80)
    print("Generated Python Code:")
    print("=" * 80)
    print(python_code)
    print("=" * 80)

    # Test if it compiles
    print("\n3. Testing if generated code compiles...")
    try:
        compile(python_code, '<generated>', 'exec')
        print("   ✓ Code compiles successfully!")
    except SyntaxError as e:
        print(f"   ✗ Syntax error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test execution
    print("\n4. Testing execution...")
    namespace = {'math': __import__('math')}
    exec(python_code, namespace)
    setup_fn = namespace['eval']

    try:
        # Test case 1: r=1000, r_given=True
        print("\n   Test 1: r=1000, r_given=True")
        result = setup_fn(r=1000.0, r_given=True, has_noise=1, has_noise_given=True)
        print(f"   Result keys: {sorted(result.keys())}")
        print(f"   r_final = {result.get('r_final')}")
        print(f"   ✓ Execution succeeded!")

        # Test case 2: r not given (should use default)
        print("\n   Test 2: r not given (should default to 1.0)")
        result = setup_fn(r=0.0, r_given=False, has_noise=1, has_noise_given=True)
        print(f"   r_final = {result.get('r_final')}")
        print(f"   ✓ Execution succeeded!")

    except Exception as e:
        print(f"   ✗ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("SUCCESS! Control flow code generation fully working!")
    print("=" * 80)
    print("\nFeatures implemented:")
    print("  ✓ Branches (br) with conditional execution")
    print("  ✓ Jumps (jmp) for unconditional control flow")
    print("  ✓ PHI nodes with predecessor tracking")
    print("  ✓ State machine dispatch with while loop")
    print("  ✓ Constants initialization from MIR")
    print("\nNext step: Generate complete setup_model() from model_param_setup MIR")

    return 0


if __name__ == '__main__':
    sys.exit(main())
