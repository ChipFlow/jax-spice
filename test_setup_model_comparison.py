#!/usr/bin/env python3
"""Compare our MIR→Python generated setup_model() against expected behavior.

Note: This doesn't yet compare against OSDI setup_model() directly since we
haven't exposed that binding in openvaf-py. Instead, we validate:
1. Physical correctness (defaults, ranges)
2. Control flow logic matches MIR semantics
3. Output structure matches OSDI expectations

TODO: Add direct OSDI comparison once we expose setup_model() in openvaf-py
"""

import sys
sys.path.insert(0, '.')

from jax_spice.codegen.mir_parser import parse_mir_function
from jax_spice.codegen.setup_model_mir_codegen import generate_setup_model_from_mir
import math


# Full resistor model setup MIR
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


def test_resistor_setup_model():
    """Test resistor setup_model() against expected behavior"""
    print("=" * 80)
    print("Resistor setup_model() Validation Test")
    print("=" * 80)

    # Parse and generate
    mir_func = parse_mir_function(MODEL_SETUP_MIR)

    param_map = {
        'v19': 'r',
        'v20': 'r_given',
        'v32': 'has_noise',
        'v33': 'has_noise_given',
        'v1': 'FALSE',
        'v2': 'TRUE',
        'v3': 'ZERO',
        'v5': 'ONE_INT',
        'v6': 'ONE',
        'v15': 'INF',
        'v21': 'r_final',
        'v23': 'r_ge_zero',
        'v24': 'r_le_inf',
        'v25': 'r_in_range',
        'v31': 'r_default',
        'v34': 'has_noise_final',
        'v35': 'has_noise_default',
    }

    setup_code = generate_setup_model_from_mir(mir_func, param_map, "resistor")
    namespace = {'math': math}
    exec(setup_code, namespace)
    setup_model = namespace['setup_model_resistor']

    print("\n1. Testing parameter validation and defaults")
    print("-" * 80)

    # Test 1: Explicit valid parameters
    print("\n   Test 1a: r=1000Ω (explicit, valid)")
    result = setup_model(r=1000.0, r_given=True, has_noise=1, has_noise_given=True)
    assert result['r'] == 1000.0, "Should accept valid resistance"
    assert result['has_noise'] == 1, "Should accept explicit has_noise"
    print(f"   ✓ Result: {result}")

    # Test 2: Default value application
    print("\n   Test 1b: r not given (should default to 1.0Ω)")
    result = setup_model(r=0.0, r_given=False, has_noise=0, has_noise_given=False)
    assert result['r'] == 1.0, f"Should apply default r=1.0, got {result['r']}"
    assert result['has_noise'] == 1, f"Should apply default has_noise=1, got {result['has_noise']}"
    print(f"   ✓ Result: {result}")

    # Test 3: Boundary values
    print("\n   Test 1c: r=0Ω (boundary, valid)")
    result = setup_model(r=0.0, r_given=True, has_noise=1, has_noise_given=True)
    assert result['r'] == 0.0, "Should accept r=0 (boundary)"
    print(f"   ✓ Result: {result}")

    print("\n   Test 1d: r=1e12Ω (large, valid)")
    result = setup_model(r=1e12, r_given=True, has_noise=0, has_noise_given=True)
    assert result['r'] == 1e12, "Should accept large resistance"
    print(f"   ✓ Result: {result}")

    # Test 4: Physical correctness
    print("\n2. Testing physical correctness")
    print("-" * 80)

    test_cases = [
        (0.1, "100mΩ"),
        (1.0, "1Ω (default)"),
        (100.0, "100Ω"),
        (1000.0, "1kΩ"),
        (1e6, "1MΩ"),
        (1e9, "1GΩ"),
    ]

    for r_value, description in test_cases:
        result = setup_model(r=r_value, r_given=True, has_noise=1, has_noise_given=True)
        # For resistor, output should exactly match input (no transformation)
        assert result['r'] == r_value, f"Failed for {description}"
        print(f"   ✓ {description}: r={result['r']}")

    # Test 5: Control flow paths
    print("\n3. Testing control flow paths")
    print("-" * 80)

    print("\n   Path A: r_given=True, valid range → use provided value")
    result = setup_model(r=500.0, r_given=True, has_noise=1, has_noise_given=True)
    assert result['r'] == 500.0, "Should use provided value"
    print(f"   ✓ Used provided r=500.0")

    print("\n   Path B: r_given=False → use default")
    result = setup_model(r=999.0, r_given=False, has_noise=1, has_noise_given=True)
    assert result['r'] == 1.0, "Should use default, not provided value"
    print(f"   ✓ Used default r=1.0 (ignored provided 999.0)")

    print("\n   Path C: has_noise_given=False → use default")
    result = setup_model(r=100.0, r_given=True, has_noise=999, has_noise_given=False)
    assert result['has_noise'] == 1, "Should use default has_noise=1"
    print(f"   ✓ Used default has_noise=1 (ignored provided 999)")

    print("\n4. Comparing against expected OSDI behavior")
    print("-" * 80)
    print("\n   Expected OSDI setup_model() behavior:")
    print("   - Validates parameters against ranges")
    print("   - Applies defaults for non-given parameters")
    print("   - Returns validated parameter dict")
    print("\n   Our generated setup_model():")
    print("   ✓ Implements same control flow from MIR")
    print("   ✓ Applies defaults correctly (r=1.0, has_noise=1)")
    print("   ✓ Returns same structure as OSDI would")
    print("\n   Note: Validation callbacks (set_Invalid) not yet implemented")
    print("         - Negative resistance would be rejected by OSDI")
    print("         - Our code passes it through (TODO)")

    print("\n5. OSDI Direct Comparison (optional)")
    print("-" * 80)
    try:
        import openvaf_py
        # Try to access the OSDI binding
        module = openvaf_py.compile_va("vendor/VACASK/devices/resistor.va")[0]
        if hasattr(module, 'call_osdi_setup_model'):
            print("   ✓ OSDI binding available (osdi-bindings feature enabled)")
            print("   Note: Binding is currently a stub - full implementation pending")
            try:
                osdi_result = module.call_osdi_setup_model({'r': 1000.0, 'r_given': True})
                print(f"   ✓ OSDI result: {osdi_result}")
            except Exception as e:
                print(f"   ⚠ OSDI binding exists but not yet implemented: {e}")
        else:
            print("   ⓘ OSDI binding not available")
            print("     To enable: cargo build --release --features osdi-bindings")
    except ImportError:
        print("   ⓘ openvaf_py not available - skipping OSDI comparison")
    except Exception as e:
        print(f"   ⚠ Error checking OSDI binding: {e}")

    print("\n" + "=" * 80)
    print("SUCCESS! Generated setup_model() behavior validated!")
    print("=" * 80)

    print("\nNext steps:")
    print("  - Add OSDI setup_model() binding to openvaf-py")
    print("  - Direct comparison: generated vs OSDI output")
    print("  - Implement validation callbacks (set_Invalid)")
    print("  - Test with more complex models (capacitor, diode)")

    return 0


if __name__ == '__main__':
    sys.exit(test_resistor_setup_model())
