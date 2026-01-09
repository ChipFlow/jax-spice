#!/usr/bin/env -S uv run --script
"""Validate MIR code generator against OSDI for diode model."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mir_codegen import generate_init_function, generate_eval_function

REPO_ROOT = Path(__file__).parent.parent
VACASK = REPO_ROOT / "vendor" / "VACASK" / "devices"
OPENVAF = REPO_ROOT / "vendor" / "OpenVAF" / "target" / "debug" / "openvaf-r"


def compile_osdi_if_needed(va_path: Path, osdi_path: Path):
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


def main():
    import openvaf_py
    import osdi_py

    print("=== DIODE MODEL VALIDATION ===\n")

    # Compile models
    va_path = VACASK / "diode.va"
    osdi_path = Path("/tmp/diode.osdi")
    compile_osdi_if_needed(va_path, osdi_path)

    # Load openvaf-py module
    modules = openvaf_py.compile_va(str(va_path))
    diode = modules[0]

    # Generate functions
    init_fn = generate_init_function(diode)
    eval_fn = generate_eval_function(diode)

    # Load OSDI library
    lib = osdi_py.OsdiLibrary(str(osdi_path))
    print(f"OSDI model: {lib.name}")
    print(f"  nodes: {lib.num_nodes}")
    print(f"  params: {lib.num_params}")

    params = lib.get_params()
    param_names = {p["name"]: i for i, p in enumerate(params)}

    # Test parameters
    test_params = {
        "Is": 1e-14,
        "Rs": 10.0,
        "N": 1.0,
        "EG": 1.11,
        "Cjo": 1e-12,
        "Vj": 0.7,
        "M": 0.5,
        "$mfactor": 1.0,
    }

    # Run OSDI
    print("\n--- OSDI Reference ---")
    model = lib.create_model()
    for name, value in test_params.items():
        if name in param_names:
            model.set_real_param(param_names[name], value)
    model.process_params()

    instance = model.create_instance()
    instance.init_node_mapping([0, 1, 2])  # A, CI, C
    instance.process_params(300.15, 3)

    # Evaluate at V(A)=0.6V forward bias
    flags = (
        osdi_py.CALC_RESIST_RESIDUAL |
        osdi_py.CALC_REACT_RESIDUAL |
        osdi_py.CALC_RESIST_JACOBIAN |
        osdi_py.CALC_REACT_JACOBIAN |
        osdi_py.ANALYSIS_DC
    )
    # Node voltages: A=0.6V, CI=0.55V, C=0V (forward bias with small Rs drop)
    V_A, V_CI, V_C = 0.6, 0.55, 0.0
    instance.eval([V_A, V_CI, V_C], flags, 0.0)

    osdi_resist = instance.load_residual_resist([0.0, 0.0, 0.0])
    osdi_react = instance.load_residual_react([0.0, 0.0, 0.0])

    print(f"  V(A,CI,C) = ({V_A}, {V_CI}, {V_C})")
    print(f"  residual_resist: {list(osdi_resist)}")
    print(f"  residual_react: {list(osdi_react)}")

    # Run generated code
    print("\n--- Generated Code ---")

    # Init params for generated code
    gen_init_params = {
        "$temperature": 300.15,
        "Tnom": 300.15,
        "area": 1.0,
        "Is": 1e-14,
        "XTI": 3.0,
        "N": 1.0,
        "EG": 1.11,
        "Rs": 10.0,
        "IBV": 1e-3,
        "Cjo": 1e-12,
        "Vj": 0.7,
        "M": 0.5,
        "FC": 0.5,
        "BV_given": False,
        "area_given": True,
        "mfactor": 1.0,
    }
    cache = init_fn(**gen_init_params)
    print(f"  cache ({len(cache)} values): {cache[:5]}...")

    # Get eval param mapping to see what we need
    metadata = diode.get_codegen_metadata()
    print(f"  eval params needed: {list(metadata['eval_param_mapping'].keys())}")

    # Eval params
    gen_eval_params = {
        "$temperature": 300.15,
        "Tnom": 300.15,
        "V(A,CI)": V_A - V_CI,
        "V(CI,C)": V_CI - V_C,
        "area": 1.0,
        "Is": 1e-14,
        "XTI": 3.0,
        "N": 1.0,
        "EG": 1.11,
        "Rs": 10.0,
        "IBV": 1e-3,
        "Cjo": 1e-12,
        "Vj": 0.7,
        "M": 0.5,
        "FC": 0.5,
        "mfactor": 1.0,
    }
    result = eval_fn(cache, **gen_eval_params)
    print(f"  residual_resist: {result['residuals_resist']}")
    print(f"  residual_react: {result['residuals_react']}")

    # Compare
    print("\n--- Comparison ---")
    tol = 1e-10
    all_match = True

    for i, (gen, osdi) in enumerate(zip(result['residuals_resist'], osdi_resist)):
        diff = abs(gen - osdi)
        match = diff < tol or (gen == osdi)
        status = "OK" if match else "MISMATCH"
        print(f"  resist[{i}]: gen={gen:.6e}, osdi={osdi:.6e} [{status}]")
        if not match:
            all_match = False

    for i, (gen, osdi) in enumerate(zip(result['residuals_react'], osdi_react)):
        diff = abs(gen - osdi)
        match = diff < tol or (gen == osdi)
        status = "OK" if match else "MISMATCH"
        print(f"  react[{i}]: gen={gen:.6e}, osdi={osdi:.6e} [{status}]")
        if not match:
            all_match = False

    print()
    if all_match:
        print("VALIDATION PASSED!")
    else:
        print("VALIDATION FAILED - results differ")

    return all_match


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
