"""C6288 16x16 Multiplier Benchmark for JAX-SPICE

This script benchmarks transient simulation of the c6288 multiplier circuit
using PSP103 MOSFET models compiled from Verilog-A.

The c6288 is a 16x16 bit multiplier with:
- 2416 gates (256 AND + 2128 NOR + 32 NOT)
- ~10,112 transistors (PSP103 MOSFET)
- 5,123 nodes after flattening

Simulation parameters (from VACASK):
- Stop time: 2ns
- Timestep: 2ps
- 1000 timesteps total

This script can run different test circuits:
- inv_test: Single inverter (6 transistors, ~10 nodes)
- nor_test: Single NOR gate
- and_test: Single AND gate
- c6288_test: Full 16x16 multiplier
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass

# Force CPU backend initially (can switch to CUDA if available)
if 'JAX_PLATFORMS' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve

# Enable float64 for numerical precision
jax.config.update('jax_enable_x64', True)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_spice.netlist.parser import VACASKParser
from jax_spice.netlist.circuit import Circuit, Instance
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.context import AnalysisContext
from jax_spice.devices.base import DeviceStamps


# Paths
VACASK_ROOT = Path(__file__).parent.parent.parent / "VACASK"
C6288_PATH = VACASK_ROOT / "benchmark" / "c6288" / "vacask" / "runme.sim"
PSP103_PATH = VACASK_ROOT / "devices" / "psp103v4" / "psp103.va"


@dataclass
class CompiledModel:
    """A compiled Verilog-A model ready for simulation"""
    name: str
    terminals: List[str]
    param_names: List[str]
    param_kinds: List[str]
    default_params: Dict[str, float]
    eval_fn: Any  # JAX function
    translator: Any = None  # OpenVAFToJAX translator


def compile_psp103(allow_analog_in_cond: bool = True) -> CompiledModel:
    """Compile PSP103 MOSFET model from Verilog-A

    Returns:
        CompiledModel for PSP103
    """
    try:
        import openvaf_py
        import openvaf_jax
    except ImportError:
        # Add openvaf-py to path
        openvaf_py_path = Path(__file__).parent.parent / "openvaf-py"
        if openvaf_py_path.exists() and str(openvaf_py_path) not in sys.path:
            sys.path.insert(0, str(openvaf_py_path))
        import openvaf_py
        import openvaf_jax

    print(f"Compiling PSP103 from {PSP103_PATH}...")
    start = time.perf_counter()

    modules = openvaf_py.compile_va(str(PSP103_PATH), allow_analog_in_cond=allow_analog_in_cond)
    if not modules:
        raise ValueError(f"No modules found in {PSP103_PATH}")

    module = modules[0]
    translator = openvaf_jax.OpenVAFToJAX(module)
    eval_fn = translator.translate()

    elapsed = time.perf_counter() - start
    print(f"  Compiled in {elapsed:.2f}s")
    print(f"  Module: {module.name}")
    print(f"  Terminals: {list(module.nodes)}")
    print(f"  Parameters: {len(module.param_names)}")

    # Build default params
    defaults = {}
    for name, kind in zip(module.param_names, module.param_kinds):
        if kind == 'param':
            defaults[name] = 0.0
        elif kind == 'temperature':
            defaults[name] = 300.15
        elif kind == 'sysfun' and name == 'mfactor':
            defaults[name] = 1.0

    return CompiledModel(
        name=module.name,
        terminals=list(module.nodes),
        param_names=list(module.param_names),
        param_kinds=list(module.param_kinds),
        default_params=defaults,
        eval_fn=eval_fn,
        translator=translator
    )


def create_simple_mosfet_eval(is_pmos: bool = False, vth0: float = 0.4,
                               kp: float = 200e-6, lambda_: float = 0.02):
    """Create a simple Level-1 MOSFET evaluation function

    This is a simplified square-law model for testing purposes.
    PSP103 is too complex for quick integration testing.

    Args:
        is_pmos: True for PMOS, False for NMOS
        vth0: Threshold voltage (V)
        kp: Transconductance parameter (A/V^2)
        lambda_: Channel length modulation

    Returns:
        Function with signature (voltages, params, context) -> DeviceStamps
    """
    sign = -1.0 if is_pmos else 1.0

    def mosfet_eval(voltages: Dict[str, float], inst_params: Dict[str, Any],
                    context: Any) -> DeviceStamps:
        """Evaluate MOSFET using simple square-law model

        Terminals: D, G, S, B
        """
        Vd = float(voltages.get('D', 0.0))
        Vg = float(voltages.get('G', 0.0))
        Vs = float(voltages.get('S', 0.0))
        Vb = float(voltages.get('B', 0.0))

        # Get W/L ratio
        W = eval_param(inst_params.get('w', inst_params.get('W', 1e-6)), _circuit_params)
        L = eval_param(inst_params.get('l', inst_params.get('L', 0.2e-6)), _circuit_params)
        if W == 0:
            W = 1e-6
        if L == 0:
            L = 0.2e-6
        WL = W / L

        # Effective Vth with body effect (simplified)
        gamma = 0.4  # Body effect coefficient
        phi = 0.6    # Surface potential
        Vsb = sign * (Vs - Vb)
        if Vsb > 0:
            Vth = vth0 + gamma * (jnp.sqrt(phi + Vsb) - jnp.sqrt(phi))
        else:
            Vth = vth0

        # Gate overdrive
        Vgs = sign * (Vg - Vs)
        Vds = sign * (Vd - Vs)
        Vov = Vgs - Vth

        # Drain current (square-law model)
        beta = kp * WL

        # Cutoff
        if Vov <= 0:
            Id = 0.0
            gm = 1e-15  # Small conductance to avoid singular matrix
            gds = 1e-15
        # Triode region
        elif Vds < Vov:
            Id = beta * (Vov * Vds - 0.5 * Vds * Vds) * (1 + lambda_ * Vds)
            gm = beta * Vds * (1 + lambda_ * Vds)
            gds = beta * (Vov - Vds) * (1 + lambda_ * Vds) + beta * (Vov * Vds - 0.5 * Vds * Vds) * lambda_
        # Saturation
        else:
            Id = 0.5 * beta * Vov * Vov * (1 + lambda_ * Vds)
            gm = beta * Vov * (1 + lambda_ * Vds)
            gds = 0.5 * beta * Vov * Vov * lambda_

        # Apply sign for PMOS
        Id = sign * Id

        # Small signal conductances
        # For NMOS: d(Id)/d(Vd) = gds, d(Id)/d(Vg) = gm, d(Id)/d(Vs) = -(gm + gds)
        # For PMOS: signs are flipped

        # Current stamps (Id enters D, exits S)
        currents = {
            'D': jnp.array(Id),
            'G': jnp.array(0.0),
            'S': jnp.array(-Id),
            'B': jnp.array(0.0),
        }

        # Conductance stamps (linearized about operating point)
        gm = max(gm, 1e-15)
        gds = max(gds, 1e-15)

        conductances = {
            ('D', 'D'): jnp.array(gds),
            ('D', 'G'): jnp.array(sign * gm),
            ('D', 'S'): jnp.array(-gds - sign * gm),
            ('D', 'B'): jnp.array(0.0),
            ('G', 'D'): jnp.array(0.0),
            ('G', 'G'): jnp.array(1e-15),  # Gate leakage
            ('G', 'S'): jnp.array(0.0),
            ('G', 'B'): jnp.array(0.0),
            ('S', 'D'): jnp.array(-gds),
            ('S', 'G'): jnp.array(-sign * gm),
            ('S', 'S'): jnp.array(gds + sign * gm),
            ('S', 'B'): jnp.array(0.0),
            ('B', 'D'): jnp.array(0.0),
            ('B', 'G'): jnp.array(0.0),
            ('B', 'S'): jnp.array(0.0),
            ('B', 'B'): jnp.array(1e-15),
        }

        return DeviceStamps(currents=currents, conductances=conductances)

    return mosfet_eval


def eval_param(value: Any, circuit_params: Dict[str, str] = None) -> float:
    """Evaluate a parameter value, resolving references

    Args:
        value: Parameter value (may be string expression or number)
        circuit_params: Circuit-level parameters for reference resolution

    Returns:
        Evaluated float value
    """
    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Try direct conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Handle SPICE suffixes (n=1e-9, p=1e-12, etc.)
        suffixes = {
            'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
            'k': 1e3, 'meg': 1e6, 'g': 1e9, 't': 1e12
        }
        for suffix, mult in suffixes.items():
            if value.lower().endswith(suffix):
                try:
                    return float(value[:-len(suffix)]) * mult
                except ValueError:
                    pass

        # Look up in circuit params
        if circuit_params and value.lower() in circuit_params:
            return eval_param(circuit_params[value.lower()], circuit_params)
        if circuit_params and value in circuit_params:
            return eval_param(circuit_params[value], circuit_params)

        # Try to evaluate simple expressions
        try:
            # Handle things like "0.5*vdd"
            expr = value.lower()
            if circuit_params:
                for k, v in circuit_params.items():
                    expr = expr.replace(k.lower(), str(eval_param(v, {})))
            return float(eval(expr))
        except Exception:
            pass

    return 0.0


# Global circuit params (set during build)
_circuit_params: Dict[str, str] = {}


def create_vsource_eval():
    """Create voltage source evaluation function"""
    def vsource_eval(voltages: Dict[str, float], params: Dict[str, Any],
                     context: Any) -> DeviceStamps:
        Vp = voltages.get('p', voltages.get('0', 0.0))
        Vn = voltages.get('n', voltages.get('1', 0.0))

        # Get DC voltage - resolve parameter references
        # For pulse sources, DC = val0 (initial value)
        if 'dc' in params:
            V_dc = eval_param(params['dc'], _circuit_params)
        elif params.get('type', '').strip('"') == 'pulse':
            # For pulse, use val0 for DC operating point
            V_dc = eval_param(params.get('val0', 0.0), _circuit_params)
        else:
            V_dc = eval_param(params.get('v', 0.0), _circuit_params)

        V_target = V_dc

        V_actual = Vp - Vn
        G_big = 1e9  # Reduced conductance for better conditioning (was 1e12)
        I = G_big * (V_actual - V_target)

        return DeviceStamps(
            currents={'p': jnp.array(I), 'n': jnp.array(-I)},
            conductances={
                ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
                ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
            }
        )

    return vsource_eval


def create_resistor_eval():
    """Create resistor evaluation function"""
    def resistor_eval(voltages: Dict[str, float], params: Dict[str, Any],
                      context: Any) -> DeviceStamps:
        Vp = voltages.get('p', voltages.get('0', 0.0))
        Vn = voltages.get('n', voltages.get('1', 0.0))

        # Get resistance - try various parameter names and resolve references
        R_param = params.get('r', params.get('R', params.get('rs', 1000.0)))
        R = eval_param(R_param, _circuit_params)
        if R == 0:
            R = 1000.0  # Default to 1k if unresolved
        G = 1.0 / max(R, 1e-12)
        I = G * (Vp - Vn)

        return DeviceStamps(
            currents={'p': jnp.array(I), 'n': jnp.array(-I)},
            conductances={
                ('p', 'p'): jnp.array(G), ('p', 'n'): jnp.array(-G),
                ('n', 'p'): jnp.array(-G), ('n', 'n'): jnp.array(G)
            }
        )

    return resistor_eval


def parse_c6288() -> Circuit:
    """Parse the c6288 benchmark netlist"""
    print(f"Parsing {C6288_PATH}...")
    start = time.perf_counter()

    parser = VACASKParser()
    circuit = parser.parse_file(C6288_PATH)

    elapsed = time.perf_counter() - start
    print(f"  Parsed in {elapsed:.3f}s")
    print(f"  Subcircuits: {len(circuit.subckts)}")
    print(f"  Models: {list(circuit.models.keys())}")

    return circuit


def flatten_circuit(circuit: Circuit, top_name: str = 'c6288_test') -> Tuple[List[Instance], Dict[str, int]]:
    """Flatten circuit hierarchy"""
    print(f"Flattening {top_name}...")
    start = time.perf_counter()

    instances, nodes = circuit.flatten(top_name)

    elapsed = time.perf_counter() - start
    print(f"  Flattened in {elapsed:.3f}s")
    print(f"  Instances: {len(instances)}")
    print(f"  Nodes: {len(nodes)}")
    # Only show first few node names to avoid flooding output
    node_list = list(nodes.keys())
    print(f"  Sample nodes: {node_list[:10]}...")
    print(f"  Instances details (first 5):")
    for inst in instances[:5]:
        print(f"    {inst.name}: model={inst.model}, terms={inst.terminals}")

    # Count by model type
    by_model = {}
    for inst in instances:
        model = inst.model
        by_model[model] = by_model.get(model, 0) + 1
    print(f"  By model: {by_model}")

    return instances, nodes


def build_mna_system(circuit: Circuit, instances: List[Instance],
                     nodes: Dict[str, int],
                     psp103_model: Optional[CompiledModel] = None) -> MNASystem:
    """Build MNA system from flattened circuit"""
    global _circuit_params

    print("Building MNA system...")
    start = time.perf_counter()

    # Set circuit params for parameter evaluation
    _circuit_params = circuit.params.copy()

    system = MNASystem(num_nodes=len(nodes), node_names=nodes)

    # Create evaluation functions
    vsource_eval = create_vsource_eval()
    resistor_eval = create_resistor_eval()

    psp103_n_eval = None
    psp103_p_eval = None

    # Use simple MOSFET model for now (PSP103 is too complex for quick testing)
    # Parameters tuned for typical 130nm-like process with Vdd=1.2V
    nmos_eval = create_simple_mosfet_eval(is_pmos=False, vth0=0.35, kp=300e-6, lambda_=0.02)
    pmos_eval = create_simple_mosfet_eval(is_pmos=True, vth0=0.35, kp=100e-6, lambda_=0.02)

    skipped = 0
    for inst in instances:
        model_name = inst.model.lower()

        # Map terminals to node indices
        node_indices = [nodes[t] for t in inst.terminals]

        # Select evaluation function based on model
        # IMPORTANT: terminals list must match what the eval_fn returns in DeviceStamps
        if model_name == 'v':
            eval_fn = vsource_eval
            # vsource_eval uses 'p' and 'n', instance terminals are [pos, neg]
            terminals = ['p', 'n']
        elif model_name == 'r':
            eval_fn = resistor_eval
            terminals = ['p', 'n']
        elif model_name == 'psp103n':
            eval_fn = nmos_eval
            # PSP103 has D, G, S, B terminals - instance terminals are in this order
            terminals = ['D', 'G', 'S', 'B']
        elif model_name == 'psp103p':
            eval_fn = pmos_eval
            terminals = ['D', 'G', 'S', 'B']
        else:
            skipped += 1
            continue

        # Ensure we have the right number of node indices
        if len(node_indices) != len(terminals):
            print(f"Warning: Instance {inst.name} has {len(node_indices)} nodes but model needs {len(terminals)}")
            skipped += 1
            continue

        device = DeviceInfo(
            name=inst.name,
            model_name=inst.model,
            terminals=terminals,
            node_indices=node_indices,
            params=inst.params,
            eval_fn=eval_fn
        )
        system.devices.append(device)

    elapsed = time.perf_counter() - start
    print(f"  Built in {elapsed:.3f}s")
    print(f"  Devices: {len(system.devices)}")
    if skipped > 0:
        print(f"  Skipped: {skipped}")

    return system


def dc_operating_point(system: MNASystem, max_iterations: int = 100,
                       abstol: float = 1e-9, reltol: float = 1e-3,
                       verbose: bool = False) -> Tuple[Array, Dict]:
    """Find DC operating point using damped Newton-Raphson iteration

    Uses adaptive damping to improve convergence for nonlinear circuits.

    Args:
        system: MNA system
        max_iterations: Maximum NR iterations
        abstol: Absolute convergence tolerance
        reltol: Relative convergence tolerance
        verbose: Print iteration details

    Returns:
        (voltages, info) tuple
    """
    n = system.num_nodes
    dtype = jnp.float64

    # Initial guess: set supply nodes to expected values
    V = jnp.zeros(n, dtype=dtype)

    # Set initial voltages for known supply nodes
    for name, idx in system.node_names.items():
        name_lower = name.lower()
        if 'vdd' in name_lower:
            # Get Vdd value from circuit params
            vdd = eval_param(_circuit_params.get('vdd', 1.2), _circuit_params)
            V = V.at[idx].set(vdd)
        # vss stays at 0

    context = AnalysisContext(
        time=0.0,
        dt=1e-9,
        analysis_type='dc',
        c0=0.0,
        c1=0.0,
        rhs_correction=0.0,
    )

    converged = False
    residual_norm = 1e20
    prev_residual_norm = 1e20

    for iteration in range(max_iterations):
        # Build Jacobian and residual
        J, f = system.build_jacobian_and_residual(V, context)

        # Check convergence
        residual_norm = float(jnp.max(jnp.abs(f)))

        if verbose and iteration < 10:
            print(f"    Iter {iteration}: residual={residual_norm:.2e}, V_max={float(jnp.max(V[1:])):.4f}, V_min={float(jnp.min(V[1:])):.4f}")

        if residual_norm < abstol:
            converged = True
            break

        # Regularize for numerical stability
        reg = 1e-10 * jnp.eye(J.shape[0], dtype=dtype)

        # Solve for update
        try:
            delta_V = solve(J + reg, -f)
        except Exception as e:
            print(f"  Warning: Linear solve failed at iteration {iteration}: {e}")
            break

        # Limit voltage step (damping) - adaptive
        max_step = 2.0  # Allow larger steps now that supplies are initialized
        max_delta = float(jnp.max(jnp.abs(delta_V)))
        step_scale = float(jnp.minimum(1.0, max_step / (max_delta + 1e-15)))

        # Apply damped update (skip ground node at index 0)
        V = V.at[1:].add(step_scale * delta_V)

        # Clamp voltages to reasonable range for this technology
        vdd = eval_param(_circuit_params.get('vdd', 1.2), _circuit_params)
        V = jnp.clip(V, -0.5, vdd + 0.5)

        # Check delta convergence
        delta_norm = float(jnp.max(jnp.abs(step_scale * delta_V)))
        v_norm = float(jnp.max(jnp.abs(V[1:])))

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break

        prev_residual_norm = residual_norm

    return V, {
        'converged': converged,
        'iterations': iteration + 1,
        'residual_norm': residual_norm
    }


def run_benchmark(circuit_name: str = 'inv_test'):
    """Run the benchmark

    Args:
        circuit_name: Which circuit to simulate ('inv_test', 'nor_test', 'and_test', 'c6288_test')
    """
    print("=" * 70)
    print(f"JAX-SPICE Benchmark: {circuit_name}")
    print("=" * 70)
    print()

    # System info
    print(f"JAX Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    # Step 1: Parse netlist
    circuit = parse_c6288()
    print()

    # Step 2: Flatten hierarchy
    instances, nodes = flatten_circuit(circuit, circuit_name)
    print()

    # Step 3: Compile PSP103 model (optional - skip if not available)
    psp103_model = None
    try:
        psp103_model = compile_psp103()
        print()
    except Exception as e:
        print(f"Warning: Could not compile PSP103: {e}")
        print("Continuing with simplified device models...")
        print()

    # Step 4: Build MNA system
    system = build_mna_system(circuit, instances, nodes, psp103_model)
    print()

    # Step 5: DC operating point
    print("Finding DC operating point...")
    start = time.perf_counter()
    V_dc, dc_info = dc_operating_point(system, verbose=True)
    dc_time = time.perf_counter() - start

    if dc_info['converged']:
        print(f"  Converged in {dc_info['iterations']} iterations, {dc_time*1000:.2f}ms")
        print(f"  Residual norm: {dc_info['residual_norm']:.2e}")

        # Print some node voltages
        print("  Node voltages:")
        for name, idx in sorted(nodes.items(), key=lambda x: x[1])[:10]:
            print(f"    {name}: {float(V_dc[idx]):.4f} V")
    else:
        print(f"  Did not converge after {dc_info['iterations']} iterations")
        print(f"  Residual norm: {dc_info['residual_norm']:.2e}")
    print()

    # Summary
    print("-" * 70)
    print("Summary:")
    print(f"  Circuit: {circuit_name}")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Total devices: {len(system.devices)}")

    # Count device types
    by_model = {}
    for d in system.devices:
        m = d.model_name.lower()
        by_model[m] = by_model.get(m, 0) + 1
    print(f"  Device breakdown: {by_model}")
    print("-" * 70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='JAX-SPICE Benchmark')
    parser.add_argument('--circuit', '-c', default='gatedrv_test',
                        choices=['inv_test', 'nor_test', 'and_test', 'gatedrv_test', 'c6288_test'],
                        help='Circuit to simulate')
    args = parser.parse_args()
    run_benchmark(args.circuit)
