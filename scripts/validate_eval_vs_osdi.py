#!/usr/bin/env -S uv run --script
"""Validate generated eval function against OSDI reference.

This script generates an eval function from MIR and compares
against the OSDI reference implementation.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
VACASK_DEVICES = REPO_ROOT / "vendor" / "VACASK" / "devices"
OPENVAF = REPO_ROOT / "vendor" / "OpenVAF" / "target" / "debug" / "openvaf-r"


def compile_osdi_if_needed(va_path: Path, osdi_path: Path) -> None:
    """Compile .va to .osdi if needed."""
    if osdi_path.exists():
        return
    if not OPENVAF.exists():
        raise RuntimeError(f"OpenVAF not found at {OPENVAF}")
    print(f"Compiling {va_path.name}...")
    result = subprocess.run(
        [str(OPENVAF), str(va_path), "-o", str(osdi_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stderr}")


def generated_eval_capacitor(cache: list, voltage: float, mfactor: float) -> dict:
    """Generated eval function for capacitor (from MIR).

    Args:
        cache: [effective_c, -effective_c] from init function
        voltage: V(A) - V(B) terminal voltage
        mfactor: instance multiplier

    Returns:
        dict with 'residuals' and 'jacobian' entries
    """
    # Constants from MIR
    v3 = 0.0  # Zero (for resistive parts)

    # Map inputs to MIR variables
    # - c parameter is not directly used in eval (only through cache)
    # - V(A,B) -> v17
    # - mfactor -> v25
    # - cache[0] -> v37 (effective_c = mfactor * c from init)
    # - cache[1] -> v40 (-effective_c from init)

    v17 = voltage
    v25 = mfactor
    v37 = cache[0]  # effective_c
    v40 = cache[1]  # -effective_c

    # MIR instructions:
    # v18 = v16 * v17  -- but v16 (c) isn't directly available
    # We need to derive c from cache: effective_c = mfactor * c, so c = effective_c / mfactor
    # But wait - let me re-examine the MIR...

    # Looking at the actual MIR, v16 IS c and IS a param, but for cache-based eval
    # we should compute Q = effective_c * V = cache[0] * V (not c * V)
    #
    # Actually, looking at the MIR more carefully:
    # v18 = c * V
    # v33 = mfactor * v18 = mfactor * c * V = effective_c * V
    # v26 = v33 (residual[0])
    #
    # So the residual is Q = effective_c * V = cache[0] * V

    # For the MIR-faithful version, we'd need c as input:
    # v16 = c
    # v18 = v16 * v17  # c * V
    # v23 = v18        # optbarrier
    # v33 = v25 * v18  # mfactor * (c * V)
    # v26 = v33        # residual[0]
    # v27 = -v18       # -(c * V)
    # v30 = v37        # cache[0] -> jacobian[0,0]
    # v32 = v40        # cache[1] -> jacobian[1,0]
    # v35 = v25 * v27  # mfactor * (-(c*V))
    # v34 = v35        # residual[1]
    # v36 = v25        # mfactor (unused?)
    # v38 = v40        # cache[1] -> jacobian[0,1]
    # v41 = v37        # cache[0] -> jacobian[1,1]

    # But since effective_c = mfactor * c is precomputed in cache,
    # we can simplify: Q = effective_c * V

    # Compute residuals (reactive charge)
    Q_A = v37 * v17  # effective_c * V
    Q_B = v40 * v17  # -effective_c * V = -Q_A

    residuals_resist = [v3, v3]  # No resistive current
    residuals_react = [Q_A, Q_B]

    # Compute Jacobian (dQ/dV)
    # J[row, col] = d(residual_row) / d(V_col)
    # V(A,B) = V_A - V_B, so dV(A,B)/dV_A = 1, dV(A,B)/dV_B = -1
    #
    # Q_A = C * V(A,B) = C * V_A - C * V_B
    # dQ_A/dV_A = C = effective_c
    # dQ_A/dV_B = -C = -effective_c
    #
    # Q_B = -C * V(A,B) = -C * V_A + C * V_B
    # dQ_B/dV_A = -C = -effective_c
    # dQ_B/dV_B = C = effective_c

    jacobian_resist = [
        [v3, v3],  # Row 0
        [v3, v3],  # Row 1
    ]
    jacobian_react = [
        [v37, v40],  # Row 0: [dQ_A/dV_A, dQ_A/dV_B] = [C, -C]
        [v40, v37],  # Row 1: [dQ_B/dV_A, dQ_B/dV_B] = [-C, C]
    ]

    return {
        'residuals_resist': residuals_resist,
        'residuals_react': residuals_react,
        'jacobian_resist': jacobian_resist,
        'jacobian_react': jacobian_react,
    }


def validate_with_osdi():
    """Compare generated eval vs OSDI reference."""
    import osdi_py

    # Compile capacitor
    va_path = VACASK_DEVICES / "capacitor.va"
    osdi_path = Path("/tmp/capacitor.osdi")
    compile_osdi_if_needed(va_path, osdi_path)

    lib = osdi_py.OsdiLibrary(str(osdi_path))
    print(f"Loaded OSDI model: {lib.name}")

    params = lib.get_params()
    param_names = {p["name"]: i for i, p in enumerate(params)}

    # Test cases: (c, mfactor, V_A, V_B, description)
    test_cases = [
        (1e-9, 1.0, 1.0, 0.0, "1nF, V=1V"),
        (1e-12, 1.0, 0.5, 0.0, "1pF, V=0.5V"),
        (2e-9, 2.0, 1.0, 0.0, "2nF m=2, V=1V"),
        (5e-12, 3.0, -1.0, 0.0, "5pF m=3, V=-1V"),
        (1e-9, 1.0, 2.0, 1.0, "1nF, V_A=2, V_B=1"),
    ]

    print("\n" + "=" * 70)
    print("Validating generated eval vs OSDI reference")
    print("=" * 70)

    all_passed = True
    for c, mfactor, V_A, V_B, desc in test_cases:
        print(f"\nTest: {desc}")
        print(f"  inputs: c={c}, mfactor={mfactor}, V_A={V_A}, V_B={V_B}")

        # Calculate cache values (from init)
        effective_c = mfactor * c
        cache = [effective_c, -effective_c]
        voltage = V_A - V_B

        # Run generated eval
        gen_result = generated_eval_capacitor(cache, voltage, mfactor)
        print(f"  generated residuals (react): {gen_result['residuals_react']}")
        print(f"  generated jacobian (react): {gen_result['jacobian_react']}")

        # Run OSDI reference
        model = lib.create_model()
        model.set_real_param(param_names["c"], c)
        model.set_real_param(param_names["$mfactor"], mfactor)
        model.process_params()

        instance = model.create_instance()
        instance.init_node_mapping([0, 1])
        instance.process_params(300.0, 2)

        flags = (
            osdi_py.CALC_RESIST_RESIDUAL |
            osdi_py.CALC_REACT_RESIDUAL |
            osdi_py.CALC_RESIST_JACOBIAN |
            osdi_py.CALC_REACT_JACOBIAN |
            osdi_py.ANALYSIS_DC
        )
        instance.eval([V_A, V_B], flags, 0.0)

        # Get OSDI results
        osdi_resist = instance.load_residual_resist([0.0, 0.0])
        osdi_react = instance.load_residual_react([0.0, 0.0])
        osdi_jac_resist = instance.write_jacobian_array_resist()
        osdi_jac_react = instance.write_jacobian_array_react()

        print(f"  OSDI residuals (react): {osdi_react}")
        print(f"  OSDI jacobian (react): {osdi_jac_react}")

        # Compare residuals
        tol = 1e-15
        passed = True

        for i, (gen, osdi) in enumerate(zip(gen_result['residuals_react'], osdi_react)):
            diff = abs(gen - osdi)
            if diff > tol:
                print(f"  MISMATCH residual[{i}]: gen={gen}, osdi={osdi}, diff={diff}")
                passed = False

        # Compare Jacobian (reactive)
        # OSDI returns flat array, need to reconstruct
        # For capacitor: entries are (0,0), (0,1), (1,0), (1,1) by convention
        gen_jac_flat = [
            gen_result['jacobian_react'][0][0],  # (0,0)
            gen_result['jacobian_react'][0][1],  # (0,1)
            gen_result['jacobian_react'][1][0],  # (1,0)
            gen_result['jacobian_react'][1][1],  # (1,1)
        ]

        for i, (gen, osdi) in enumerate(zip(gen_jac_flat, osdi_jac_react)):
            diff = abs(gen - osdi)
            if diff > tol:
                print(f"  MISMATCH jacobian[{i}]: gen={gen}, osdi={osdi}, diff={diff}")
                passed = False

        if passed:
            print("  PASSED")
        else:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    try:
        success = validate_with_osdi()
        sys.exit(0 if success else 1)
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure osdi_py is installed.")
        sys.exit(1)
