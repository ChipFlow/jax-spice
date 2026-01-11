"""OSDI vs openvaf_jax Comparison Tests

This module compares the behavior of openvaf_jax generated models against
OSDI compiled models by sweeping voltages and comparing currents/residuals/Jacobians.

Phase 1: Simple Components (resistor, capacitor, diode)
Phase 2: Complex Transistor Models (PSP103, BSIM4)

The goal is to validate that our JAX translator produces results that match
the reference OSDI implementation.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Add openvaf_jax and openvaf_py to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "openvaf_jax" / "openvaf_py"))

import openvaf_py  # noqa: E402

import openvaf_jax  # noqa: E402

# Try to import osdi_py - skip tests if not available
try:
    import osdi_py
    OSDI_AVAILABLE = True
except ImportError:
    OSDI_AVAILABLE = False


# Paths
VACASK_DEVICES = project_root / "vendor" / "VACASK" / "devices"
OPENVAF_INTEGRATION = project_root / "openvaf-py" / "vendor" / "OpenVAF" / "integration_tests"
OSDI_CACHE = Path("/tmp/osdi_jax_test_cache")


# =============================================================================
# Helper Functions
# =============================================================================


def find_openvaf_compiler() -> Path | None:
    """Find the OpenVAF compiler binary.

    Returns None if not found (tests will be skipped).
    """
    repo_root = project_root

    # Try debug build first
    openvaf = repo_root / "vendor" / "OpenVAF" / "target" / "debug" / "openvaf-r"
    if openvaf.exists():
        return openvaf

    # Try release build
    openvaf = repo_root / "vendor" / "OpenVAF" / "target" / "release" / "openvaf-r"
    if openvaf.exists():
        return openvaf

    return None


# Check if OpenVAF compiler is available
OPENVAF_COMPILER = find_openvaf_compiler()
OPENVAF_AVAILABLE = OPENVAF_COMPILER is not None


def compile_va_to_osdi(va_path: Path, osdi_path: Path) -> None:
    """Compile Verilog-A to OSDI using OpenVAF."""
    if osdi_path.exists():
        return

    if OPENVAF_COMPILER is None:
        raise RuntimeError("OpenVAF compiler not available")

    osdi_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [str(OPENVAF_COMPILER), str(va_path), "-o", str(osdi_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"OpenVAF compilation failed:\n{result.stderr}")


def create_osdi_evaluator(osdi_path: Path, params: dict, temperature: float = 300.0):
    """Create an OSDI evaluator function with fixed params.

    Args:
        osdi_path: Path to compiled OSDI file
        params: Dict of parameter name -> value
        temperature: Device temperature in K

    Returns:
        Tuple of (evaluate_fn, lib, model, instance) where evaluate_fn has signature:
            fn(voltages: list) -> (resist_residuals, react_residuals, resist_jac, react_jac)
    """
    lib = osdi_py.OsdiLibrary(str(osdi_path))
    model = lib.create_model()

    # Set model parameters
    lib_params = lib.get_params()
    for i, p in enumerate(lib_params):
        name = p["name"]
        if name in params:
            model.set_real_param(i, float(params[name]))
        elif name == "$mfactor" and "mfactor" in params:
            model.set_real_param(i, float(params["mfactor"]))

    model.process_params()

    # Create instance
    instance = model.create_instance()
    node_indices = list(range(lib.num_terminals))
    instance.init_node_mapping(node_indices)
    instance.process_params(temperature, lib.num_terminals)

    def evaluate(voltages):
        """Evaluate device at given voltages."""
        flags = (
            osdi_py.CALC_RESIST_RESIDUAL
            | osdi_py.CALC_REACT_RESIDUAL
            | osdi_py.CALC_RESIST_JACOBIAN
            | osdi_py.CALC_REACT_JACOBIAN
            | osdi_py.ANALYSIS_DC
        )
        ret_flags = instance.eval(voltages, flags, 0.0)
        assert ret_flags == 0, f"OSDI eval returned error flags: {ret_flags}"

        # Get residuals (pass zeros array to load)
        n_nodes = lib.num_nodes
        resist_res = instance.load_residual_resist([0.0] * n_nodes)
        react_res = instance.load_residual_react([0.0] * n_nodes)

        # Get Jacobian
        resist_jac = instance.write_jacobian_array_resist()
        react_jac = instance.write_jacobian_array_react()

        return resist_res, react_res, resist_jac, react_jac

    return evaluate, lib, model, instance


def create_jax_evaluator(va_path: Path, params: dict, temperature: float = 300.0):
    """Create a JAX evaluator function with fixed params.

    Args:
        va_path: Path to Verilog-A file
        params: Dict of parameter name -> value
        temperature: Device temperature in K

    Returns:
        Tuple of (evaluate_fn, module, metadata) where evaluate_fn has signature:
            fn(voltages: list) -> (resist_residuals, react_residuals, resist_jac, react_jac)
    """
    modules = openvaf_py.compile_va(str(va_path))
    module = modules[0]
    translator = openvaf_jax.OpenVAFToJAX(module)
    eval_fn, metadata = translator.translate_array()

    # Get param info
    param_names = list(module.param_names)

    def evaluate(voltages):
        """Evaluate device at given voltages."""
        # Build inputs array
        inputs = [0.0] * len(param_names)
        for i, name in enumerate(param_names):
            # Handle voltage params (e.g., V(A,B) or V(A))
            if name.startswith('V('):
                # Parse voltage node(s)
                inner = name[2:-1]  # Remove V( and )
                if ',' in inner:
                    node_pos, node_neg = inner.split(',')
                    # Find voltage indices
                    node_pos = node_pos.strip()
                    node_neg = node_neg.strip()
                    node_order = metadata.get('node_names', [])
                    if node_pos in node_order and node_neg in node_order:
                        idx_pos = node_order.index(node_pos)
                        idx_neg = node_order.index(node_neg)
                        if idx_pos < len(voltages) and idx_neg < len(voltages):
                            inputs[i] = voltages[idx_pos] - voltages[idx_neg]
                else:
                    node = inner.strip()
                    node_order = metadata.get('node_names', [])
                    if node in node_order:
                        idx = node_order.index(node)
                        if idx < len(voltages):
                            inputs[i] = voltages[idx]
            elif name in params:
                inputs[i] = float(params[name])
            elif name == 'mfactor':
                inputs[i] = float(params.get('mfactor', 1.0))
            elif 'temperature' in name.lower():
                inputs[i] = temperature

        # Run eval
        res_resist, res_react, jac_resist, jac_react = eval_fn(inputs)

        # Convert to lists
        return (
            list(np.array(res_resist)),
            list(np.array(res_react)),
            list(np.array(jac_resist)),
            list(np.array(jac_react)),
        )

    return evaluate, module, metadata


def compare_arrays(
    osdi_arr: list,
    jax_arr: list,
    rtol: float = 1e-10,
    atol: float = 1e-15,
) -> tuple:
    """Compare arrays with relative and absolute tolerance.

    Returns:
        Tuple of (passed, max_abs_diff, max_rel_diff)
    """
    osdi = np.array(osdi_arr)
    jax = np.array(jax_arr)

    if osdi.shape != jax.shape:
        return False, float('inf'), float('inf')

    abs_diff = np.abs(osdi - jax)
    max_abs = np.max(abs_diff) if abs_diff.size > 0 else 0.0

    # Relative difference (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(
            np.abs(osdi) > atol,
            abs_diff / np.abs(osdi),
            0.0
        )
    max_rel = np.max(rel_diff) if rel_diff.size > 0 else 0.0

    passed = np.allclose(osdi, jax, rtol=rtol, atol=atol)
    return passed, max_abs, max_rel


# =============================================================================
# Phase 1: Simple Component Tests
# =============================================================================


@pytest.mark.skipif(
    not OSDI_AVAILABLE or not OPENVAF_AVAILABLE,
    reason="osdi_py or OpenVAF compiler not available"
)
class TestResistorComparison:
    """Compare OSDI vs JAX for resistor model."""

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile resistor.va to OSDI once per class."""
        va_path = VACASK_DEVICES / "resistor.va"
        osdi_path = OSDI_CACHE / "resistor.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get resistor.va path."""
        return VACASK_DEVICES / "resistor.va"

    def test_single_point(self, osdi_path, va_path):
        """Compare at a single voltage point."""
        params = {"r": 1000.0, "mfactor": 1.0}

        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        # Evaluate at V(A)=1V, V(B)=0V
        voltages = [1.0, 0.0]

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare resistive residuals
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[0], jax_res[0], rtol=1e-10, atol=1e-15        )
        assert passed, f"Resistive residual mismatch: max_abs={max_abs}, max_rel={max_rel}"

        # Compare reactive residuals (should be 0 for resistor)
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[1], jax_res[1], rtol=1e-10, atol=1e-15        )
        assert passed, f"Reactive residual mismatch: max_abs={max_abs}, max_rel={max_rel}"

    def test_voltage_sweep(self, osdi_path, va_path):
        """Sweep voltage from -1V to +1V, compare currents."""
        params = {"r": 2000.0, "mfactor": 1.0}

        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        max_resist_diff = 0.0

        for v in np.linspace(-1.0, 1.0, 21):
            voltages = [v, 0.0]  # V(A)=v, V(B)=0

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            resist_diff = np.max(np.abs(np.array(osdi_res[0]) - np.array(jax_res[0])))
            max_resist_diff = max(max_resist_diff, resist_diff)

        assert max_resist_diff < 1e-10, f"Max resistive diff over sweep: {max_resist_diff}"

    def test_jacobian_match(self, osdi_path, va_path):
        """Compare Jacobian (dI/dV = 1/R)."""
        r_val = 500.0
        params = {"r": r_val, "mfactor": 1.0}

        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        voltages = [1.0, 0.0]

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare resistive Jacobian
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[2], jax_res[2], rtol=1e-10, atol=1e-15        )
        assert passed, f"Resistive Jacobian mismatch: max_abs={max_abs}, max_rel={max_rel}"


@pytest.mark.skipif(
    not OSDI_AVAILABLE or not OPENVAF_AVAILABLE,
    reason="osdi_py or OpenVAF compiler not available"
)
class TestCapacitorComparison:
    """Compare OSDI vs JAX for capacitor model."""

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile capacitor.va to OSDI once per class."""
        va_path = VACASK_DEVICES / "capacitor.va"
        osdi_path = OSDI_CACHE / "capacitor.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get capacitor.va path."""
        return VACASK_DEVICES / "capacitor.va"

    def test_reactive_charge(self, osdi_path, va_path):
        """Q = C*V comparison."""
        c_val = 1e-12  # 1pF
        params = {"c": c_val, "mfactor": 1.0}

        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        voltages = [1.0, 0.0]  # V = 1V
        expected_q = c_val * 1.0  # Q = C*V

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare reactive residuals (charge)
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[1], jax_res[1], rtol=1e-10, atol=1e-20
        )
        assert passed, f"Reactive residual mismatch: max_abs={max_abs}, max_rel={max_rel}"

        # Verify absolute value
        assert abs(abs(jax_res[1][0]) - expected_q) < 1e-20, \
            f"Expected Q={expected_q}, got {jax_res[1][0]}"

    def test_reactive_jacobian(self, osdi_path, va_path):
        """dQ/dV = C comparison."""
        c_val = 2e-12  # 2pF
        params = {"c": c_val, "mfactor": 1.0}

        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        voltages = [1.0, 0.0]

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare reactive Jacobian (should be C on diagonal, -C off-diagonal)
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[3], jax_res[3], rtol=1e-10, atol=1e-20        )
        assert passed, f"Reactive Jacobian mismatch: max_abs={max_abs}, max_rel={max_rel}"


@pytest.mark.skipif(
    not OSDI_AVAILABLE or not OPENVAF_AVAILABLE,
    reason="osdi_py or OpenVAF compiler not available"
)
class TestDiodeComparison:
    """Compare OSDI vs JAX for diode model."""

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile diode.va to OSDI once per class."""
        va_path = VACASK_DEVICES / "diode.va"
        osdi_path = OSDI_CACHE / "diode.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get diode.va path."""
        return VACASK_DEVICES / "diode.va"

    def test_forward_bias_sweep(self, osdi_path, va_path):
        """Sweep V from 0 to 0.8V in forward bias."""
        params = {
            "is": 1e-12,  # Saturation current
            "n": 1.0,     # Ideality factor
            "rs": 0.0,    # Series resistance (0 for simple test)
            "mfactor": 1.0,
        }

        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        max_resist_diff = 0.0

        for v in np.linspace(0.0, 0.7, 15):  # Forward bias up to 0.7V
            voltages = [v, 0.0]  # V(A)=v, V(C)=0

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            resist_diff = np.max(np.abs(np.array(osdi_res[0]) - np.array(jax_res[0])))
            max_resist_diff = max(max_resist_diff, resist_diff)

        # Diode has exponential behavior, allow slightly larger tolerance
        assert max_resist_diff < 1e-8, f"Max resistive diff over sweep: {max_resist_diff}"

    def test_reverse_bias(self, osdi_path, va_path):
        """Test at V = -1V (reverse bias)."""
        params = {
            "is": 1e-12,
            "n": 1.0,
            "rs": 0.0,
            "mfactor": 1.0,
        }

        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        voltages = [-1.0, 0.0]  # V(A)=-1V, V(C)=0 (reverse bias)

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare resistive residuals
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[0], jax_res[0], rtol=1e-8, atol=1e-15        )
        assert passed, f"Resistive residual mismatch: max_abs={max_abs}, max_rel={max_rel}"

    def test_jacobian_match(self, osdi_path, va_path):
        """Compare Jacobian (conductance) at operating point."""
        params = {
            "is": 1e-12,
            "n": 1.0,
            "rs": 0.0,
            "mfactor": 1.0,
        }

        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, params)
        jax_eval, _, _ = create_jax_evaluator(va_path, params)

        voltages = [0.6, 0.0]  # Forward bias at 0.6V

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare resistive Jacobian
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[2], jax_res[2], rtol=1e-6, atol=1e-12        )
        assert passed, f"Resistive Jacobian mismatch: max_abs={max_abs}, max_rel={max_rel}"


# =============================================================================
# Phase 2: Complex Transistor Model Tests (PSP103, BSIM4)
# =============================================================================


def get_psp103_va_path() -> Path:
    """Find PSP103 Verilog-A file."""
    # Try OpenVAF integration tests location
    psp103_path = OPENVAF_INTEGRATION / "PSP103" / "psp103.va"
    if psp103_path.exists():
        return psp103_path

    # Try alternative location
    alt_path = project_root / "vendor" / "OpenVAF" / "integration_tests" / "PSP103" / "psp103.va"
    if alt_path.exists():
        return alt_path

    raise FileNotFoundError("PSP103 model not found")


def get_bsim4_va_path() -> Path:
    """Find BSIM4 Verilog-A file."""
    # Try VACASK devices location
    bsim4_path = VACASK_DEVICES / "bsim4v8.va"
    if bsim4_path.exists():
        return bsim4_path

    raise FileNotFoundError("BSIM4 model not found")


@pytest.mark.skipif(
    not OSDI_AVAILABLE or not OPENVAF_AVAILABLE,
    reason="osdi_py or OpenVAF compiler not available"
)
class TestPSP103Comparison:
    """Compare OSDI vs JAX for PSP103 MOSFET model."""

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile psp103.va to OSDI once per class."""
        va_path = get_psp103_va_path()
        osdi_path = OSDI_CACHE / "psp103.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get psp103.va path."""
        return get_psp103_va_path()

    @pytest.fixture
    def nmos_params(self):
        """Minimal NMOS parameters for PSP103."""
        return {
            "TYPE": 1,       # NMOS
            "VFB": -0.9,     # Flat-band voltage
            "NEFF": 2.5e17,  # Effective doping
            "NP": 1.0e26,    # Poly doping
            "TOX": 1.8e-9,   # Oxide thickness
            "W": 1e-6,       # Width
            "L": 100e-9,     # Length
            "mfactor": 1.0,
        }

    def test_ids_vs_vgs_sweep(self, osdi_path, va_path, nmos_params):
        """Sweep Vgs at fixed Vds, compare Ids."""
        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, nmos_params)
        jax_eval, _, _ = create_jax_evaluator(va_path, nmos_params)

        max_diff = 0.0
        vds = 0.5  # Fixed Vds

        for vgs in np.linspace(0.0, 1.0, 11):
            # MOSFET terminals: D, G, S, B
            # Voltages referenced to ground (usually S or B)
            voltages = [vds, vgs, 0.0, 0.0]  # Vd, Vg, Vs, Vb

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            # Compare drain current (first residual typically)
            resist_diff = np.max(np.abs(np.array(osdi_res[0]) - np.array(jax_res[0])))
            max_diff = max(max_diff, resist_diff)

        # Complex models may have more numerical variation
        assert max_diff < 1e-6, f"Max Ids diff over Vgs sweep: {max_diff}"

    def test_ids_vs_vds_sweep(self, osdi_path, va_path, nmos_params):
        """Sweep Vds at fixed Vgs, compare Ids (output characteristics)."""
        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, nmos_params)
        jax_eval, _, _ = create_jax_evaluator(va_path, nmos_params)

        max_diff = 0.0
        vgs = 0.6  # Fixed Vgs (above threshold)

        for vds in np.linspace(0.0, 1.0, 11):
            voltages = [vds, vgs, 0.0, 0.0]

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            resist_diff = np.max(np.abs(np.array(osdi_res[0]) - np.array(jax_res[0])))
            max_diff = max(max_diff, resist_diff)

        assert max_diff < 1e-6, f"Max Ids diff over Vds sweep: {max_diff}"

    def test_jacobian_match(self, osdi_path, va_path, nmos_params):
        """Compare transconductance/output conductance."""
        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, nmos_params)
        jax_eval, _, _ = create_jax_evaluator(va_path, nmos_params)

        # Typical operating point
        voltages = [0.5, 0.6, 0.0, 0.0]

        osdi_res = osdi_eval(voltages)
        jax_res = jax_eval(voltages)

        # Compare Jacobian
        passed, max_abs, max_rel = compare_arrays(
            osdi_res[2], jax_res[2], rtol=1e-4, atol=1e-10        )
        assert passed, f"Jacobian mismatch: max_abs={max_abs}, max_rel={max_rel}"


@pytest.mark.skipif(
    not OSDI_AVAILABLE or not OPENVAF_AVAILABLE,
    reason="osdi_py or OpenVAF compiler not available"
)
class TestBSIM4Comparison:
    """Compare OSDI vs JAX for BSIM4 MOSFET model."""

    @pytest.fixture(scope="class")
    def osdi_path(self):
        """Compile bsim4v8.va to OSDI once per class."""
        va_path = get_bsim4_va_path()
        osdi_path = OSDI_CACHE / "bsim4v8.osdi"
        compile_va_to_osdi(va_path, osdi_path)
        return osdi_path

    @pytest.fixture(scope="class")
    def va_path(self):
        """Get bsim4v8.va path."""
        return get_bsim4_va_path()

    @pytest.fixture
    def nmos_params(self):
        """Minimal NMOS parameters for BSIM4."""
        return {
            "L": 100e-9,     # Length
            "W": 1e-6,       # Width
            "NF": 1,         # Number of fingers
            "mfactor": 1.0,
        }

    def test_ids_vs_vgs_sweep(self, osdi_path, va_path, nmos_params):
        """Sweep Vgs at fixed Vds, compare Ids."""
        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, nmos_params)
        jax_eval, _, _ = create_jax_evaluator(va_path, nmos_params)

        max_diff = 0.0
        vds = 0.5

        for vgs in np.linspace(0.0, 1.0, 11):
            voltages = [vds, vgs, 0.0, 0.0]  # D, G, S, B

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            resist_diff = np.max(np.abs(np.array(osdi_res[0]) - np.array(jax_res[0])))
            max_diff = max(max_diff, resist_diff)

        assert max_diff < 1e-6, f"Max Ids diff over Vgs sweep: {max_diff}"

    def test_ids_vs_vds_sweep(self, osdi_path, va_path, nmos_params):
        """Sweep Vds at fixed Vgs, compare Ids."""
        osdi_eval, _, _, _ = create_osdi_evaluator(osdi_path, nmos_params)
        jax_eval, _, _ = create_jax_evaluator(va_path, nmos_params)

        max_diff = 0.0
        vgs = 0.6

        for vds in np.linspace(0.0, 1.0, 11):
            voltages = [vds, vgs, 0.0, 0.0]

            osdi_res = osdi_eval(voltages)
            jax_res = jax_eval(voltages)

            resist_diff = np.max(np.abs(np.array(osdi_res[0]) - np.array(jax_res[0])))
            max_diff = max(max_diff, resist_diff)

        assert max_diff < 1e-6, f"Max Ids diff over Vds sweep: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
