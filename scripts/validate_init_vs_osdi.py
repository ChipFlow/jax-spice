#!/usr/bin/env -S uv run --script
"""Validate generated init function against OSDI reference.

This script compares the cache values produced by our MIR-generated
Python init function against the reference OSDI implementation.
"""

import subprocess
import sys
import math
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
VACASK_DEVICES = REPO_ROOT / "vendor" / "VACASK" / "devices"
OPENVAF = REPO_ROOT / "vendor" / "OpenVAF" / "target" / "debug" / "openvaf-r"


def compile_osdi_if_needed(va_path: Path, osdi_path: Path) -> None:
    """Compile .va to .osdi if needed."""
    if osdi_path.exists():
        return

    if not OPENVAF.exists():
        raise RuntimeError(
            f"OpenVAF compiler not found at {OPENVAF}. "
            "Build with: cd vendor/OpenVAF && cargo build --bin openvaf-r --features llvm21"
        )

    print(f"Compiling {va_path.name} -> {osdi_path.name}...")
    result = subprocess.run(
        [str(OPENVAF), str(va_path), "-o", str(osdi_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stderr}")


def generated_init_capacitor(c: float, mfactor: float, c_given: bool) -> list:
    """Generated init function for capacitor (from MIR)."""
    # Constants
    v1 = False
    v15 = math.inf
    v38 = 1e-12
    v4 = 0

    # Variables
    v33 = None
    v34 = None
    v35 = None
    v36 = None
    v37 = None
    v39 = None
    v40 = None
    v41 = None
    v42 = None

    # Control flow simulation
    current_block = "block4"
    prev_block = None

    while current_block is not None:
        if current_block == "block4":
            if c_given:
                prev_block, current_block = current_block, "block5"
            else:
                prev_block, current_block = current_block, "block6"
        elif current_block == "block2":
            v17 = -v33
            v19 = mfactor * v33
            v27 = v19  # optbarrier
            v22 = mfactor * v17
            v28 = v22  # optbarrier
            prev_block, current_block = current_block, "block3"
        elif current_block == "block5":
            v34 = float(v4)
            v35 = v34 <= c
            if v35:
                prev_block, current_block = current_block, "block10"
            else:
                prev_block, current_block = current_block, "block11"
        elif current_block == "block10":
            v36 = c <= v15
            prev_block, current_block = current_block, "block12"
        elif current_block == "block11":
            prev_block, current_block = current_block, "block12"
        elif current_block == "block12":
            if prev_block == "block10":
                v37 = v36
            elif prev_block == "block11":
                v37 = v1
            else:
                v37 = None
            if v37:
                prev_block, current_block = current_block, "block9"
            else:
                prev_block, current_block = current_block, "block13"
        elif current_block == "block13":
            # callback (validation) - skip for now
            prev_block, current_block = current_block, "block8"
        elif current_block == "block9":
            prev_block, current_block = current_block, "block8"
        elif current_block == "block8":
            prev_block, current_block = current_block, "block7"
        elif current_block == "block6":
            v39 = float(v4)
            v40 = v39 <= v38
            if v40:
                prev_block, current_block = current_block, "block16"
            else:
                prev_block, current_block = current_block, "block17"
        elif current_block == "block16":
            v41 = v38 <= v15
            prev_block, current_block = current_block, "block18"
        elif current_block == "block17":
            prev_block, current_block = current_block, "block18"
        elif current_block == "block18":
            if prev_block == "block16":
                v42 = v41
            elif prev_block == "block17":
                v42 = v1
            else:
                v42 = None
            if v42:
                prev_block, current_block = current_block, "block15"
            else:
                prev_block, current_block = current_block, "block19"
        elif current_block == "block19":
            prev_block, current_block = current_block, "block14"
        elif current_block == "block15":
            prev_block, current_block = current_block, "block14"
        elif current_block == "block14":
            v33 = v38
            prev_block, current_block = current_block, "block7"
        elif current_block == "block7":
            if prev_block == "block8":
                v33 = c
            elif prev_block == "block14":
                v33 = v38
            else:
                v33 = None
            prev_block, current_block = current_block, "block2"
        elif current_block == "block3":
            current_block = None  # Exit
        else:
            break

    return [v27, v28]


def validate_with_osdi():
    """Compare generated init vs OSDI reference."""
    import osdi_py

    # Compile capacitor if needed
    va_path = VACASK_DEVICES / "capacitor.va"
    osdi_path = Path("/tmp/capacitor.osdi")
    compile_osdi_if_needed(va_path, osdi_path)

    # Load OSDI library
    lib = osdi_py.OsdiLibrary(str(osdi_path))
    print(f"Loaded OSDI model: {lib.name}")
    print(f"  num_nodes: {lib.num_nodes}")
    print(f"  num_terminals: {lib.num_terminals}")
    print(f"  num_params: {lib.num_params}")

    # Get param info
    params = lib.get_params()
    param_names = {p["name"]: i for i, p in enumerate(params)}
    print(f"  parameters: {list(param_names.keys())}")

    # Test cases
    test_cases = [
        {"c": 1e-9, "c_given": True, "mfactor": 1.0, "desc": "1nF, given"},
        {"c": 1e-12, "c_given": False, "mfactor": 1.0, "desc": "default 1pF (not given)"},
        {"c": 2e-9, "c_given": True, "mfactor": 2.0, "desc": "2nF, m=2"},
        {"c": 5e-12, "c_given": True, "mfactor": 3.0, "desc": "5pF, m=3"},
    ]

    print("\n" + "=" * 60)
    print("Validating generated init vs OSDI reference")
    print("=" * 60)

    all_passed = True
    for tc in test_cases:
        print(f"\nTest: {tc['desc']}")
        print(f"  inputs: c={tc['c']}, c_given={tc['c_given']}, mfactor={tc['mfactor']}")

        # Run generated init
        gen_cache = generated_init_capacitor(tc["c"], tc["mfactor"], tc["c_given"])
        print(f"  generated: {gen_cache}")

        # Run OSDI reference
        model = lib.create_model()

        # Set parameters (only if given)
        if tc["c_given"] and "c" in param_names:
            model.set_real_param(param_names["c"], tc["c"])
        if "$mfactor" in param_names:
            model.set_real_param(param_names["$mfactor"], tc["mfactor"])

        model.process_params()

        # Create instance
        instance = model.create_instance()
        instance.init_node_mapping([0, 1])
        instance.process_params(300.0, 2)

        # Evaluate to get reactive residuals (charge = capacitance effect)
        flags = osdi_py.CALC_REACT_RESIDUAL | osdi_py.ANALYSIS_DC
        instance.eval([1.0, 0.0], flags, 0.0)

        # Get reactive residuals (Q = C * V)
        react_residual = instance.load_residual_react([0.0, 0.0])
        osdi_cache = [react_residual[0], react_residual[1]]
        print(f"  OSDI ref:  {osdi_cache}")

        # Compare
        tol = 1e-15
        match = True
        for i, (gen, osdi) in enumerate(zip(gen_cache, osdi_cache)):
            diff = abs(gen - osdi)
            if diff > tol:
                print(f"  MISMATCH at index {i}: gen={gen}, osdi={osdi}, diff={diff}")
                match = False
                all_passed = False

        if match:
            print("  PASSED")

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    try:
        success = validate_with_osdi()
        sys.exit(0 if success else 1)
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure osdi_py is installed. Run: cd osdi-py && maturin develop")
        sys.exit(1)
