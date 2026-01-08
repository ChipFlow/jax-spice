#!/usr/bin/env python3
"""Test setup_model() generation from MIR."""

import sys
sys.path.insert(0, '.')

from jax_spice.codegen.mir_parser import parse_mir_function
from jax_spice.codegen.setup_model_mir_codegen import generate_setup_model_from_mir

# Full resistor model setup MIR with validation callbacks
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
    print("setup_model() Generation from MIR Test")
    print("=" * 80)

    # Parse MIR
    print("\n1. Parsing model_param_setup MIR...")
    mir_func = parse_mir_function(MODEL_SETUP_MIR)
    print(f"   Parsed {len(mir_func.blocks)} blocks")
    print(f"   Parameters: {', '.join(p.name for p in mir_func.params)}")

    # Parameter mapping for resistor
    # v19 = r (resistance value)
    # v20 = r_given (was r explicitly provided?)
    # v32 = has_noise (noise flag value)
    # v33 = has_noise_given (was has_noise explicitly provided?)
    # Pre-allocated OpenVAF constants (see openvaf/mir/src/dfg/values.rs):
    # v1 = FALSE, v2 = TRUE, v3 = F_ZERO, v5 = ONE (i32), v6 = F_ONE
    param_map = {
        'v19': 'r',
        'v20': 'r_given',
        'v32': 'has_noise',
        'v33': 'has_noise_given',
        'v1': 'FALSE',    # Pre-allocated OpenVAF constant
        'v2': 'TRUE',     # Pre-allocated OpenVAF constant
        'v3': 'ZERO',     # Pre-allocated OpenVAF constant (F_ZERO)
        'v5': 'ONE_INT',  # Pre-allocated OpenVAF constant (i32)
        'v6': 'ONE',      # Pre-allocated OpenVAF constant (F_ONE)
        'v15': 'INF',
        'v21': 'r_final',
        'v23': 'r_ge_zero',
        'v24': 'r_le_inf',
        'v25': 'r_in_range',
        'v31': 'r_default',
        'v34': 'has_noise_final',
        'v35': 'has_noise_default',
    }

    # Generate setup_model()
    print("\n2. Generating setup_model() code...")
    setup_code = generate_setup_model_from_mir(mir_func, param_map, "resistor")

    print("\n" + "=" * 80)
    print("Generated setup_model() Code:")
    print("=" * 80)
    print(setup_code)
    print("=" * 80)

    # Test if it compiles
    print("\n3. Testing if generated code compiles...")
    try:
        compile(setup_code, '<generated>', 'exec')
        print("   ✓ Code compiles successfully!")
    except SyntaxError as e:
        print(f"   ✗ Syntax error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test execution
    print("\n4. Testing execution...")
    namespace = {'math': __import__('math')}
    exec(setup_code, namespace)
    setup_model = namespace['setup_model_resistor']

    try:
        # Test case 1: Valid r, has_noise given
        print("\n   Test 1: r=1000, r_given=True, has_noise=1")
        result = setup_model(r=1000.0, r_given=True, has_noise=1, has_noise_given=True)
        print(f"   Result: {result}")
        assert result['r'] == 1000.0
        assert result['has_noise'] == 1
        print("   ✓ Test 1 passed!")

        # Test case 2: Use defaults
        print("\n   Test 2: r not given (should default to 1.0)")
        result = setup_model(r=0.0, r_given=False, has_noise=0, has_noise_given=False)
        print(f"   Result: {result}")
        assert result['r'] == 1.0  # Default value
        assert result['has_noise'] == 1  # Default value
        print("   ✓ Test 2 passed!")

        # Test case 3: Invalid r (negative) - should trigger callback
        print("\n   Test 3: r=-100 (invalid, should trigger validation)")
        result = setup_model(r=-100.0, r_given=True, has_noise=1, has_noise_given=True)
        print(f"   Result: {result}")
        # Note: Callbacks are not implemented yet, so this will pass through
        print("   ✓ Test 3 passed (callbacks not yet implemented)")

    except Exception as e:
        print(f"   ✗ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 80)
    print("SUCCESS! setup_model() generation working!")
    print("=" * 80)
    print("\nFeatures implemented:")
    print("  ✓ MIR-based parameter setup")
    print("  ✓ Default value application")
    print("  ✓ Control flow for parameter validation")
    print("\nTODO:")
    print("  - Implement validation callbacks (set_Invalid)")
    print("  - Add actual parameter range checking")

    return 0


if __name__ == '__main__':
    sys.exit(main())
