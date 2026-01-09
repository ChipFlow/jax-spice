#!/usr/bin/env python3
"""
Test the osdi-py library with a capacitor model.

This test validates that the OSDI interface correctly loads and evaluates
a simple capacitor device compiled by OpenVAF.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add osdi-py to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent))

import osdi_py


def compile_capacitor_if_needed(osdi_path: str) -> None:
    """Compile capacitor.va to OSDI if the file doesn't exist."""
    if os.path.exists(osdi_path):
        return

    # Find OpenVAF compiler
    repo_root = Path(__file__).parent.parent.parent
    openvaf = repo_root / "vendor" / "OpenVAF" / "target" / "debug" / "openvaf-r"

    if not openvaf.exists():
        raise RuntimeError(
            f"OpenVAF compiler not found at {openvaf}. "
            "Build it with: cd vendor/OpenVAF && cargo build --features llvm21"
        )

    # Find capacitor.va source
    va_source = repo_root / "vendor" / "OpenVAF" / "external" / "vacask" / "devices" / "capacitor.va"
    if not va_source.exists():
        va_source = repo_root / "vendor" / "VACASK" / "devices" / "capacitor.va"

    if not va_source.exists():
        raise RuntimeError(f"capacitor.va not found")

    print(f"Compiling {va_source} to {osdi_path}...")
    result = subprocess.run(
        [str(openvaf), str(va_source), "-o", osdi_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"OpenVAF compilation failed:\n{result.stderr}")


def test_load_library():
    """Test loading the OSDI library."""
    osdi_path = "/tmp/capacitor.osdi"
    compile_capacitor_if_needed(osdi_path)

    lib = osdi_py.OsdiLibrary(osdi_path)

    assert lib.name == "capacitor"
    assert lib.num_nodes == 2
    assert lib.num_terminals == 2
    assert lib.num_params == 2

    terminals = lib.get_terminals()
    assert terminals == ["A", "B"]

    params = lib.get_params()
    param_names = [p["name"] for p in params]
    assert "c" in param_names  # capacitance parameter


def test_model_creation():
    """Test creating a model and setting parameters."""
    osdi_path = "/tmp/capacitor.osdi"
    compile_capacitor_if_needed(osdi_path)

    lib = osdi_py.OsdiLibrary(osdi_path)
    model = lib.create_model()

    # Find the 'c' parameter ID
    params = lib.get_params()
    c_param_id = None
    for i, p in enumerate(params):
        if p["name"] == "c":
            c_param_id = i
            break

    assert c_param_id is not None, "Parameter 'c' not found"

    # Set capacitance to 1pF
    model.set_real_param(c_param_id, 1e-12)
    model.process_params()


def test_instance_eval():
    """Test instance creation and evaluation."""
    osdi_path = "/tmp/capacitor.osdi"
    compile_capacitor_if_needed(osdi_path)

    lib = osdi_py.OsdiLibrary(osdi_path)
    model = lib.create_model()

    # Find and set capacitance parameter
    params = lib.get_params()
    for i, p in enumerate(params):
        if p["name"] == "c":
            model.set_real_param(i, 1e-12)  # 1pF
            break

    model.process_params()

    # Create instance
    instance = model.create_instance()
    instance.init_node_mapping([0, 1])  # A=0, B=1
    instance.process_params(300.0, 2)  # T=300K, 2 terminals

    # Evaluate at V(A,B) = 1V
    flags = (
        osdi_py.CALC_RESIST_RESIDUAL
        | osdi_py.CALC_REACT_RESIDUAL
        | osdi_py.ANALYSIS_DC
    )
    prev_solve = [1.0, 0.0]  # V(A)=1V, V(B)=0V

    ret_flags = instance.eval(prev_solve, flags, 0.0)
    assert ret_flags == 0, f"Eval returned error flags: {ret_flags}"

    # Check residuals
    resist_residual = instance.load_residual_resist([0.0, 0.0])
    react_residual = instance.load_residual_react([0.0, 0.0])

    # Capacitor has no DC resistive current
    assert abs(resist_residual[0]) < 1e-30
    assert abs(resist_residual[1]) < 1e-30

    # Reactive charge: Q = C * V = 1e-12 * 1.0 = 1e-12
    C = 1e-12
    V = 1.0
    expected_Q = C * V

    # Node A: charge flows INTO node from capacitor (positive)
    # Node B: charge flows OUT of node into capacitor (negative)
    assert abs(react_residual[0] - expected_Q) < 1e-20, f"react_residual[0] = {react_residual[0]}, expected {expected_Q}"
    assert abs(react_residual[1] + expected_Q) < 1e-20, f"react_residual[1] = {react_residual[1]}, expected {-expected_Q}"


def test_jacobian():
    """Test Jacobian computation."""
    osdi_path = "/tmp/capacitor.osdi"
    compile_capacitor_if_needed(osdi_path)

    lib = osdi_py.OsdiLibrary(osdi_path)
    model = lib.create_model()

    C = 2e-12  # 2pF for different test value

    # Find and set capacitance parameter
    params = lib.get_params()
    for i, p in enumerate(params):
        if p["name"] == "c":
            model.set_real_param(i, C)
            break

    model.process_params()

    # Create instance
    instance = model.create_instance()
    instance.init_node_mapping([0, 1])
    instance.process_params(300.0, 2)

    # Evaluate
    flags = (
        osdi_py.CALC_RESIST_RESIDUAL
        | osdi_py.CALC_REACT_RESIDUAL
        | osdi_py.CALC_RESIST_JACOBIAN
        | osdi_py.CALC_REACT_JACOBIAN
        | osdi_py.ANALYSIS_DC
    )
    instance.eval([1.0, 0.0], flags, 0.0)

    # Check Jacobian
    resist_jac = instance.write_jacobian_array_resist()
    react_jac = instance.write_jacobian_array_react()

    # Resistive Jacobian should be empty for capacitor
    assert len(resist_jac) == 0

    # Reactive Jacobian: dQ/dV = C
    # Entries: (0,0), (1,0), (0,1), (1,1) based on jacobian_entries order
    assert len(react_jac) == 4

    # The Jacobian should have C on diagonal, -C off-diagonal
    # Due to KCL: d(residual_A)/dV_A = +C, d(residual_A)/dV_B = -C
    expected_values = [C, -C, -C, C]
    for i, (actual, expected) in enumerate(zip(react_jac, expected_values)):
        assert abs(actual - expected) < 1e-20, f"react_jac[{i}] = {actual}, expected {expected}"


def test_mfactor():
    """Test the $mfactor multiplier."""
    osdi_path = "/tmp/capacitor.osdi"
    compile_capacitor_if_needed(osdi_path)

    lib = osdi_py.OsdiLibrary(osdi_path)
    model = lib.create_model()

    C = 1e-12  # Base capacitance
    M = 5.0    # Multiplier

    # Set parameters
    params = lib.get_params()
    for i, p in enumerate(params):
        if p["name"] == "c":
            model.set_real_param(i, C)
        elif p["name"] == "$mfactor":
            model.set_real_param(i, M)

    model.process_params()

    # Create instance
    instance = model.create_instance()
    instance.init_node_mapping([0, 1])
    instance.process_params(300.0, 2)

    # Evaluate
    flags = osdi_py.CALC_REACT_RESIDUAL | osdi_py.ANALYSIS_DC
    instance.eval([1.0, 0.0], flags, 0.0)

    react_residual = instance.load_residual_react([0.0, 0.0])

    # With mfactor=5, effective capacitance is 5*C
    # Q = M * C * V = 5 * 1e-12 * 1.0 = 5e-12
    expected_Q = M * C * 1.0

    assert abs(react_residual[0] - expected_Q) < 1e-20, f"react_residual[0] = {react_residual[0]}, expected {expected_Q}"


if __name__ == "__main__":
    print("Testing osdi-py with capacitor model...")
    print()

    print("test_load_library...")
    test_load_library()
    print("  PASSED")

    print("test_model_creation...")
    test_model_creation()
    print("  PASSED")

    print("test_instance_eval...")
    test_instance_eval()
    print("  PASSED")

    print("test_jacobian...")
    test_jacobian()
    print("  PASSED")

    print("test_mfactor...")
    test_mfactor()
    print("  PASSED")

    print()
    print("All tests passed!")
