#!/usr/bin/env python3
"""Test setup_model() code generation."""

import sys
sys.path.insert(0, '.')

from jax_spice.codegen.setup_model_codegen import generate_setup_model, extract_param_info_from_va
from pathlib import Path

def main():
    print("=" * 80)
    print("setup_model() Code Generation Test")
    print("=" * 80)

    # Load resistor Verilog-A source
    resistor_va = Path("vendor/VACASK/devices/resistor.va").read_text()

    # Extract parameter info
    print("\n1. Extracting parameters from Verilog-A...")
    params = extract_param_info_from_va(resistor_va)

    print(f"   Found {len(params)} parameters:")
    for p in params:
        range_str = f"[{p['range'][0]}, {p['range'][1]}]" if p['range'] else "no range"
        print(f"     - {p['name']}: {p['type']} = {p['default']} ({range_str})")

    # Generate setup_model code
    print("\n2. Generating setup_model() code...")
    setup_code = generate_setup_model(params, model_name="resistor")

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
        return 1

    # Execute and test
    print("\n4. Testing execution...")
    namespace = {}
    exec(setup_code, namespace)
    setup_model = namespace['setup_model']

    # Test case 1: Valid parameters explicitly set
    print("\n   Test 1: Valid parameters (r=1000, has_noise=1)")
    try:
        result = setup_model(r=1000.0, r_given=True, has_noise=1, has_noise_given=True)
        print(f"   ✓ Result: {result}")
        assert result['r'] == 1000.0
        assert result['has_noise'] == 1
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return 1

    # Test case 2: Use defaults
    print("\n   Test 2: Use default values")
    try:
        result = setup_model(r=0.0, r_given=False, has_noise=0, has_noise_given=False)
        print(f"   ✓ Result: {result}")
        assert result['r'] == 1.0  # Default from VA source
        assert result['has_noise'] == 1  # Default from VA source
        print(f"   ✓ Defaults applied correctly!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return 1

    # Test case 3: Invalid parameter (negative resistance)
    print("\n   Test 3: Invalid parameter (r=-100)")
    try:
        result = setup_model(r=-100.0, r_given=True, has_noise=1, has_noise_given=True)
        print(f"   ✗ Should have raised ValueError!")
        return 1
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")

    # Test case 4: Boundary values
    print("\n   Test 4: Boundary value (r=0)")
    try:
        result = setup_model(r=0.0, r_given=True, has_noise=1, has_noise_given=True)
        print(f"   ✓ Accepted boundary value: {result}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return 1

    print("\n" + "=" * 80)
    print("SUCCESS! setup_model() generation working!")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
