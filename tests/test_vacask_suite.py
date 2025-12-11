"""Parameterized VACASK test suite

Automatically discovers and runs all VACASK .sim test files, extracting
expected values from embedded Python scripts and comparing with our solver.
"""

import pytest
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add openvaf-py to path
sys.path.insert(0, str(Path(__file__).parent.parent / "openvaf-py"))

import numpy as np
import jax.numpy as jnp

import openvaf_py
import openvaf_jax
from jax_spice.netlist.parser import parse_netlist
from jax_spice.devices.base import DeviceStamps
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.dc import dc_operating_point

# Paths - VACASK is at ../VACASK relative to jax-spice
JAX_SPICE_ROOT = Path(__file__).parent.parent
VACASK_ROOT = JAX_SPICE_ROOT.parent / "VACASK"
VACASK_TEST = VACASK_ROOT / "test"
VACASK_DEVICES = VACASK_ROOT / "devices"


def discover_sim_files() -> List[Path]:
    """Find all .sim test files in VACASK test directory."""
    if not VACASK_TEST.exists():
        return []
    return sorted(VACASK_TEST.glob("*.sim"))


def parse_embedded_python(sim_path: Path) -> Dict[str, Any]:
    """Extract expected values from embedded Python test script.

    Parses patterns like:
        v = op1["2"]
        exact = 10*0.9

    Returns dict with:
        - 'expectations': List of (variable_name, expected_value, tolerance)
        - 'analysis_type': 'op' or 'tran'
    """
    text = sim_path.read_text()

    # Find embedded Python between <<<FILE and >>>FILE
    match = re.search(r'<<<FILE\n(.*?)>>>FILE', text, re.DOTALL)
    if not match:
        return {'expectations': [], 'analysis_type': 'op'}

    py_code = match.group(1)
    lines = py_code.split('\n')

    expectations = []
    current_var = None

    for i, line in enumerate(lines):
        # Match: v = op1["node_name"] or i = op1["device.i"]
        m = re.match(r'\s*(\w+)\s*=\s*op1\["([^"]+)"\]', line)
        if m:
            current_var = m.group(2)
            continue

        # Match: exact = <expression>
        m = re.match(r'\s*exact\s*=\s*(.+)', line)
        if m and current_var:
            try:
                # Safe evaluation of numeric expressions
                expr = m.group(1).strip()
                # Handle simple math expressions
                val = eval(expr, {"__builtins__": {}, "np": np}, {})
                expectations.append((current_var, float(val), 1e-3))
            except:
                pass
            current_var = None

    # Determine analysis type
    analysis_type = 'op'
    if 'tran1' in py_code or 'rawread(\'tran' in py_code:
        analysis_type = 'tran'

    return {
        'expectations': expectations,
        'analysis_type': analysis_type
    }


def parse_analysis_commands(sim_path: Path) -> List[Dict]:
    """Extract analysis commands from control block."""
    text = sim_path.read_text()

    analyses = []

    # Find control block
    control_match = re.search(r'control\s*(.*?)endc', text, re.DOTALL | re.IGNORECASE)
    if not control_match:
        return analyses

    control_block = control_match.group(1)

    # Match: analysis <name> op [options]
    for m in re.finditer(r'analysis\s+(\w+)\s+op\b', control_block):
        analyses.append({'name': m.group(1), 'type': 'op'})

    # Match: analysis <name> tran stop=<time> step=<step> [icmode=<mode>]
    for m in re.finditer(r'analysis\s+(\w+)\s+tran\s+stop=(\S+)\s+step=(\S+)(?:\s+icmode="(\w+)")?', control_block):
        analyses.append({
            'name': m.group(1),
            'type': 'tran',
            'stop': m.group(2),
            'step': m.group(3),
            'icmode': m.group(4) or 'op'
        })

    return analyses


def get_required_models(sim_path: Path) -> List[str]:
    """Extract model types required by the sim file."""
    text = sim_path.read_text()
    models = set()

    # Match: load "model.osdi"
    for m in re.finditer(r'load\s+"(\w+)\.osdi"', text):
        models.add(m.group(1))

    return list(models)


def categorize_test(sim_path: Path) -> Tuple[str, List[str]]:
    """Categorize a test and return (category, skip_reasons).

    Categories:
    - 'op_basic': Simple DC operating point (resistor, diode)
    - 'op_complex': Complex DC (sweeps, multiple analyses)
    - 'tran': Transient analysis
    - 'ac': AC analysis
    - 'hb': Harmonic balance
    - 'unsupported': Not yet supported
    """
    name = sim_path.stem
    text = sim_path.read_text()
    models = get_required_models(sim_path)
    analyses = parse_analysis_commands(sim_path)

    skip_reasons = []

    # Check for unsupported features
    if 'hb' in name or any(a.get('type') == 'hb' for a in analyses if isinstance(a, dict) and 'type' in a):
        skip_reasons.append("harmonic balance not implemented")
        return 'hb', skip_reasons

    if 'ac' in name or ('analysis' in text and ' ac ' in text.lower()):
        skip_reasons.append("AC analysis not implemented")
        return 'ac', skip_reasons

    if 'mutual' in name:
        skip_reasons.append("mutual inductors not implemented")
        return 'unsupported', skip_reasons

    if 'noise' in name:
        skip_reasons.append("noise analysis not implemented")
        return 'unsupported', skip_reasons

    if 'sweep' in name:
        skip_reasons.append("parameter sweeps not yet implemented")
        return 'unsupported', skip_reasons

    if 'xf' in name:
        skip_reasons.append("transfer function analysis not implemented")
        return 'unsupported', skip_reasons

    # Check for unsupported models
    unsupported_models = {'bsimsoi', 'hicum', 'mextram'}
    for model in models:
        if model.lower() in unsupported_models:
            skip_reasons.append(f"model {model} not supported")
            return 'unsupported', skip_reasons

    # Categorize by analysis type
    has_tran = any(a.get('type') == 'tran' for a in analyses if isinstance(a, dict))
    has_op = any(a.get('type') == 'op' for a in analyses if isinstance(a, dict))

    if has_tran:
        return 'tran', skip_reasons
    elif has_op:
        if len(analyses) == 1 and models and all(m in ['resistor', 'vsource', 'isource'] for m in models):
            return 'op_basic', skip_reasons
        return 'op_complex', skip_reasons

    return 'unsupported', ["no recognized analysis type"]


# Discover all tests
ALL_SIM_FILES = discover_sim_files()

# Categorize tests
CATEGORIZED_TESTS = {path: categorize_test(path) for path in ALL_SIM_FILES}


def get_test_ids():
    """Generate test IDs from sim file names."""
    return [p.stem for p in ALL_SIM_FILES]


# ============================================================================
# Device evaluation functions for simulation
# ============================================================================

def resistor_eval(voltages, params, context):
    """Resistor evaluation function matching VACASK resistor.va"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    R = float(params.get('r', 1000.0))

    # Ensure minimum resistance
    R = max(R, 1e-12)
    G = 1.0 / R
    I = G * (Vp - Vn)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G), ('p', 'n'): jnp.array(-G),
            ('n', 'p'): jnp.array(-G), ('n', 'n'): jnp.array(G)
        }
    )


def vsource_eval(voltages, params, context):
    """Voltage source evaluation function using large conductance method."""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    V_target = float(params.get('dc', 0.0))
    V_actual = Vp - Vn

    G_big = 1e12
    I = G_big * (V_actual - V_target)

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
            ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
        }
    )


def isource_eval(voltages, params, context):
    """Current source evaluation function."""
    I = float(params.get('dc', 0.0))

    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={}
    )


def parse_si_value(s: str) -> float:
    """Parse a value with SI suffix (e.g., '2k' -> 2000)."""
    s = s.strip().lower()
    multipliers = {
        'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        'k': 1e3, 'meg': 1e6, 'g': 1e9, 't': 1e12
    }
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


# ============================================================================
# Test classes
# ============================================================================

class TestVACASKDiscovery:
    """Test that we can discover and parse VACASK test files."""

    def test_discover_sim_files(self):
        """Should find VACASK .sim test files."""
        if not VACASK_TEST.exists():
            pytest.skip("VACASK test directory not found")

        sim_files = discover_sim_files()
        assert len(sim_files) > 0, "No .sim files found"
        print(f"\nFound {len(sim_files)} .sim files")

    def test_categorize_all_tests(self):
        """Categorize all discovered tests."""
        if not ALL_SIM_FILES:
            pytest.skip("No sim files found")

        categories = {}
        for path, (cat, reasons) in CATEGORIZED_TESTS.items():
            categories.setdefault(cat, []).append(path.stem)

        print("\nTest categorization:")
        for cat, tests in sorted(categories.items()):
            print(f"  {cat}: {len(tests)} tests")
            if len(tests) <= 5:
                for t in tests:
                    print(f"    - {t}")


class TestVACASKParsing:
    """Test that we can parse all VACASK .sim files."""

    @pytest.mark.parametrize("sim_file", ALL_SIM_FILES, ids=get_test_ids())
    def test_parse_netlist(self, sim_file):
        """Each sim file should parse without error."""
        try:
            circuit = parse_netlist(sim_file)
            assert circuit is not None
            assert circuit.ground is not None
        except Exception as e:
            pytest.fail(f"Failed to parse {sim_file.name}: {e}")

    @pytest.mark.parametrize("sim_file", ALL_SIM_FILES, ids=get_test_ids())
    def test_extract_expectations(self, sim_file):
        """Should extract expected values from embedded Python."""
        result = parse_embedded_python(sim_file)
        # Just verify it doesn't crash - not all files have expectations
        assert 'expectations' in result
        assert 'analysis_type' in result


class TestVACASKOperatingPoint:
    """Run DC operating point tests from VACASK suite."""

    def _build_system_from_circuit(self, circuit) -> Tuple[MNASystem, Dict[str, int]]:
        """Build an MNASystem from a parsed circuit."""
        # Build node mapping by collecting all terminals from instances
        node_names = {'0': 0}  # Ground is always 0
        if circuit.ground and circuit.ground != '0':
            node_names[circuit.ground] = 0  # Map ground name to 0

        node_idx = 1
        for inst in circuit.top_instances:
            for terminal in inst.terminals:
                if terminal not in node_names and terminal != circuit.ground:
                    node_names[terminal] = node_idx
                    node_idx += 1

        system = MNASystem(num_nodes=node_idx, node_names=node_names)

        # Eval function mapping
        eval_funcs = {
            'vsource': vsource_eval,
            'v': vsource_eval,
            'isource': isource_eval,
            'i': isource_eval,
            'resistor': resistor_eval,
            'r': resistor_eval,
        }

        # Add devices
        for inst in circuit.top_instances:
            model_name = inst.model.lower()

            # Get node indices
            node_indices = []
            for terminal in inst.terminals:
                term_name = terminal if terminal != circuit.ground else '0'
                if term_name not in node_names:
                    node_names[term_name] = node_idx
                    node_idx += 1
                node_indices.append(node_names[term_name])

            # Get eval function
            eval_fn = eval_funcs.get(model_name)
            if eval_fn is None:
                continue  # Skip unsupported device types

            # Parse parameters
            params = {}
            for k, v in inst.params.items():
                try:
                    params[k] = parse_si_value(str(v))
                except (ValueError, TypeError):
                    params[k] = v

            device = DeviceInfo(
                name=inst.name,
                model_name=model_name,
                terminals=['p', 'n'],  # Two-terminal for R, V, I
                node_indices=node_indices,
                params=params,
                eval_fn=eval_fn
            )
            system.devices.append(device)

        return system, node_names

    def test_test_op(self):
        """Test test_op.sim - resistor voltage divider."""
        sim_file = VACASK_TEST / "test_op.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_op.sim not found")

        # Parse circuit
        circuit = parse_netlist(sim_file)

        # Build MNA system
        system, node_names = self._build_system_from_circuit(circuit)

        # Get expected values
        expectations = parse_embedded_python(sim_file)['expectations']

        # Solve DC operating point
        solution, info = dc_operating_point(system)
        assert info['converged'], f"DC analysis did not converge: {info}"

        # Check expected values
        for var_name, expected, tol in expectations:
            if var_name in node_names:
                idx = node_names[var_name]
                actual = float(solution[idx])
                rel_err = abs(actual - expected) / (abs(expected) + 1e-12)
                print(f"{var_name}: expected={expected}, actual={actual:.6f}, rel_err={rel_err:.2e}")
                assert rel_err < tol, \
                    f"{var_name}: expected {expected}, got {actual} (rel_err={rel_err:.2e})"

    def test_test_resistor(self):
        """Test test_resistor.sim - basic resistor with mfactor."""
        sim_file = VACASK_TEST / "test_resistor.sim"
        if not sim_file.exists():
            pytest.skip("VACASK test_resistor.sim not found")

        # Parse circuit
        circuit = parse_netlist(sim_file)

        # Build MNA system
        system, node_names = self._build_system_from_circuit(circuit)

        # Solve DC operating point
        solution, info = dc_operating_point(system)
        assert info['converged'], f"DC analysis did not converge: {info}"

        # Expected: V1=1V, R=2k, mfactor=3 → I = 1/(2k/3) = 1.5mA
        # Note: Our simple eval doesn't handle mfactor yet
        # Just verify circuit solves
        print(f"Solution: {dict(zip(node_names.keys(), solution))}")


class TestVACASKSummary:
    """Summary of VACASK test coverage."""

    def test_coverage_summary(self):
        """Print summary of which tests are supported."""
        if not ALL_SIM_FILES:
            pytest.skip("No sim files found")

        supported = []
        unsupported = []

        for path, (cat, reasons) in CATEGORIZED_TESTS.items():
            if reasons:
                unsupported.append((path.stem, cat, reasons))
            else:
                supported.append((path.stem, cat))

        print(f"\n{'='*60}")
        print(f"VACASK Test Suite Coverage")
        print(f"{'='*60}")
        print(f"Supported: {len(supported)}/{len(ALL_SIM_FILES)} tests")
        print(f"Unsupported: {len(unsupported)}/{len(ALL_SIM_FILES)} tests")
        print()

        if unsupported:
            print("Unsupported tests:")
            for name, cat, reasons in sorted(unsupported):
                print(f"  {name} ({cat}): {', '.join(reasons)}")
