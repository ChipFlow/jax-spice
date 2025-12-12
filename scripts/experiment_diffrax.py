#!/usr/bin/env python3
"""Diffrax experiments: RC circuit validation and multi-node extension.

Tests if diffrax can solve circuit ODEs using Kvaerno5 (implicit solver).

Experiment 1: Single RC circuit (analytical validation)
Experiment 2: Two-node RC ladder (matrix ODE: dV/dt = C_inv @ (I - G @ V))

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/experiment_diffrax.py
"""

import jax
import jax.numpy as jnp
import diffrax

jax.config.update('jax_enable_x64', True)


# =============================================================================
# Experiment 1: Single RC Circuit
# =============================================================================

def rc_circuit_ode(t, V, args):
    """dV/dt for RC charging circuit."""
    Vs, R, C = args
    return (Vs - V) / (R * C)


def experiment_single_rc():
    """Single RC circuit with analytical validation."""
    print("=" * 60)
    print("Experiment 1: Single RC Circuit")
    print("=" * 60)
    print()

    # Circuit: Vs --[R]-- V --[C]-- GND
    Vs = 5.0        # Voltage source: 5V
    R = 1000.0      # Resistance: 1kΩ
    C = 1e-6        # Capacitance: 1µF
    tau = R * C     # Time constant: 1ms

    print(f"Circuit: Vs={Vs}V --[R={R}Ω]-- V --[C={C*1e6}µF]-- GND")
    print(f"Time constant τ = {tau*1000}ms")
    print()

    # Solve with diffrax
    term = diffrax.ODETerm(rc_circuit_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    t_end = 5 * tau
    save_times = jnp.linspace(0, t_end, 100)

    solution = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=t_end, dt0=tau/100,
        y0=jnp.array(0.0),
        args=(Vs, R, C),
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=save_times)
    )

    # Compare to analytical
    V_analytical = Vs * (1 - jnp.exp(-solution.ts / tau))
    max_error = float(jnp.max(jnp.abs(solution.ys - V_analytical)))

    print(f"Final V: {float(solution.ys[-1]):.6f}V (expected: {float(Vs * (1 - jnp.exp(-5))):.6f}V)")
    print(f"Max error: {max_error:.2e}")

    success = max_error < 1e-5
    print(f"{'✓ PASS' if success else '✗ FAIL'}")
    print()
    return success


# =============================================================================
# Experiment 2: Two-Node RC Ladder
# =============================================================================

def multinode_ode(t, V, args):
    """dV/dt for multi-node circuit.

    Circuit:
        Vs --[R1]-- V1 --[R2]-- V2 --[R3]-- GND
                    |           |
                   [C1]        [C2]
                    |           |
                   GND         GND

    Node equations (KCL):
        C1 * dV1/dt = (Vs - V1)/R1 - (V1 - V2)/R2
        C2 * dV2/dt = (V1 - V2)/R2 - V2/R3

    Matrix form: C @ dV/dt = I - G @ V
    So: dV/dt = C_inv @ (I - G @ V)
    """
    Vs, C_inv, G, I_src = args
    return C_inv @ (I_src - G @ V)


def experiment_multinode():
    """Two-node RC ladder circuit."""
    print("=" * 60)
    print("Experiment 2: Two-Node RC Ladder")
    print("=" * 60)
    print()

    # Circuit parameters
    Vs = 5.0
    R1, R2, R3 = 1000.0, 1000.0, 1000.0  # 1kΩ each
    C1, C2 = 1e-6, 1e-6                   # 1µF each

    print(f"Circuit: Vs={Vs}V --[R1={R1}Ω]-- V1 --[R2={R2}Ω]-- V2 --[R3={R3}Ω]-- GND")
    print(f"                              |                |")
    print(f"                            [C1={C1*1e6}µF]          [C2={C2*1e6}µF]")
    print(f"                              |                |")
    print(f"                             GND              GND")
    print()

    # Build matrices
    # G matrix (conductance): G @ V gives currents out of each node
    G = jnp.array([
        [1/R1 + 1/R2, -1/R2],
        [-1/R2, 1/R2 + 1/R3]
    ])

    # C matrix (capacitance) - diagonal for this circuit
    C = jnp.diag(jnp.array([C1, C2]))
    C_inv = jnp.diag(jnp.array([1/C1, 1/C2]))

    # Source current vector (Vs/R1 flows into node 1)
    I_src = jnp.array([Vs/R1, 0.0])

    print(f"G matrix:\n{G}")
    print(f"C matrix: diag([{C1}, {C2}])")
    print(f"I_src: {I_src}")
    print()

    # Solve with diffrax
    term = diffrax.ODETerm(multinode_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)

    # Time scale: RC ladder has multiple time constants
    tau_fast = min(R1*C1, R2*C2, R3*C2)
    tau_slow = max(R1*C1, R2*C2) * 2  # Approximate
    t_end = 20e-3  # 20ms for full steady state

    print(f"Simulating for {t_end*1000}ms...")
    solution = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=t_end, dt0=tau_fast/10,
        y0=jnp.zeros(2),
        args=(Vs, C_inv, G, I_src),
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0, t_end, 200))
    )

    # Steady-state solution (solve G @ V_ss = I_src)
    V_steady = jnp.linalg.solve(G, I_src)
    print(f"Steady-state (analytical): V1={float(V_steady[0]):.4f}V, V2={float(V_steady[1]):.4f}V")
    print(f"Final (diffrax):           V1={float(solution.ys[-1, 0]):.4f}V, V2={float(solution.ys[-1, 1]):.4f}V")

    # Check steady-state error
    ss_error = jnp.max(jnp.abs(solution.ys[-1] - V_steady))
    print(f"Steady-state error: {float(ss_error):.2e}")

    # Print trajectory at key times
    print()
    print("Trajectory:")
    for t_ms in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        idx = int(jnp.argmin(jnp.abs(solution.ts - t_ms/1000)))
        V1, V2 = solution.ys[idx]
        print(f"  t={t_ms}ms: V1={float(V1):.4f}V, V2={float(V2):.4f}V")

    success = ss_error < 1e-4
    print()
    print(f"{'✓ PASS' if success else '✗ FAIL'}")
    print()
    return success


# =============================================================================
# Experiment 3: Parse VACASK RC Circuit
# =============================================================================

def extract_linear_matrices(runner):
    """Extract G (conductance) and C (capacitance) matrices from parsed circuit.

    Works for circuits with only resistors and capacitors (linear devices).
    Returns (G, C, node_count) where matrices are (n-1) x (n-1) excluding ground.
    """
    n = runner.num_nodes  # Includes ground at index 0
    n_unknowns = n - 1  # Exclude ground

    # Initialize matrices
    G = jnp.zeros((n_unknowns, n_unknowns))
    C = jnp.zeros((n_unknowns, n_unknowns))

    for dev in runner.devices:
        model = dev['model']
        nodes = dev['nodes']
        params = dev['params']

        if model == 'resistor':
            # Resistor: stamp conductance G = 1/R
            R = params.get('r', 1000.0)
            g = 1.0 / R

            p, n = nodes[0], nodes[1]  # Positive, negative terminals
            # Stamp into G matrix (excluding ground at index 0)
            if p > 0 and n > 0:  # Both non-ground
                G = G.at[p-1, p-1].add(g)
                G = G.at[p-1, n-1].add(-g)
                G = G.at[n-1, p-1].add(-g)
                G = G.at[n-1, n-1].add(g)
            elif p > 0:  # n is ground
                G = G.at[p-1, p-1].add(g)
            elif n > 0:  # p is ground
                G = G.at[n-1, n-1].add(g)

        elif model == 'capacitor':
            # Capacitor: stamp capacitance
            cap = params.get('c', 1e-6)

            p, n = nodes[0], nodes[1]
            if p > 0 and n > 0:
                C = C.at[p-1, p-1].add(cap)
                C = C.at[p-1, n-1].add(-cap)
                C = C.at[n-1, p-1].add(-cap)
                C = C.at[n-1, n-1].add(cap)
            elif p > 0:
                C = C.at[p-1, p-1].add(cap)
            elif n > 0:
                C = C.at[n-1, n-1].add(cap)

    return G, C, n_unknowns


def extract_source_info(runner):
    """Extract voltage source information from parsed circuit.

    Returns dict mapping node indices to source parameters.
    """
    sources = {}
    for dev in runner.devices:
        if dev['model'] != 'vsource':
            continue

        nodes = dev['nodes']
        params = dev['params']

        sources[dev['name']] = {
            'node_p': nodes[0],
            'node_n': nodes[1],
            'dc': params.get('dc', 0.0),
            'type': params.get('type', 'dc'),
            'val0': params.get('val0', 0.0),
            'val1': params.get('val1', 1.0),
            'rise': params.get('rise', 1e-9),
            'fall': params.get('fall', 1e-9),
            'width': params.get('width', 1e-6),
            'period': params.get('period', 2e-6),
        }

    return sources


def build_source_current_fn(sources, G, n_unknowns):
    """Build function that returns source current vector I_src(t).

    For voltage sources modeled as high-conductance stamps:
    I_src[node] = G_vs * V_source(t)
    """
    G_vs = 1e12  # High conductance for voltage source stamp

    def source_fn(t):
        I_src = jnp.zeros(n_unknowns)

        for name, src in sources.items():
            node_p, node_n = src['node_p'], src['node_n']

            # Get source voltage at time t
            if src['type'] == 'dc':
                V_src = src['dc']
            elif src['type'] == 'pulse':
                v0, v1 = src['val0'], src['val1']
                rise, fall = src['rise'], src['fall']
                width, period = src['width'], src['period']
                t_mod = t % period
                # Simple pulse waveform
                V_src = jnp.where(
                    t_mod < rise, v0 + (v1 - v0) * t_mod / rise,
                    jnp.where(
                        t_mod < rise + width, v1,
                        jnp.where(
                            t_mod < rise + width + fall,
                            v1 - (v1 - v0) * (t_mod - rise - width) / fall,
                            v0
                        )
                    )
                )
            else:
                V_src = src['dc']

            # Current injection: I = G_vs * V_src
            # Flows into node_p, out of node_n
            if node_p > 0:
                I_src = I_src.at[node_p - 1].add(G_vs * V_src)
            if node_n > 0:
                I_src = I_src.at[node_n - 1].add(-G_vs * V_src)

        return I_src

    return source_fn, G_vs


def experiment_vacask_rc():
    """Parse VACASK RC circuit and solve with diffrax."""
    from pathlib import Path
    import sys

    # Add parent to path for imports
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from jax_spice.benchmarks.runner import VACASKBenchmarkRunner

    print("=" * 60)
    print("Experiment 3: VACASK RC Circuit via Parser")
    print("=" * 60)
    print()

    # Parse VACASK RC circuit
    sim_path = project_root / "vendor/VACASK/benchmark/rc/vacask/runme.sim"
    if not sim_path.exists():
        print(f"SKIP: VACASK RC circuit not found at {sim_path}")
        return None

    print(f"Parsing: {sim_path}")
    runner = VACASKBenchmarkRunner(sim_path)
    runner.parse()

    print(f"Nodes: {runner.num_nodes} (including ground)")
    print(f"Devices: {len(runner.devices)}")
    for dev in runner.devices:
        print(f"  {dev['name']}: {dev['model']} nodes={dev['nodes']} params={dev['params']}")
    print()

    # Extract linear matrices
    G, C, n_unknowns = extract_linear_matrices(runner)
    print(f"G matrix (conductance):\n{G}")
    print(f"C matrix (capacitance):\n{C}")
    print()

    # Extract source info and build source function
    sources = extract_source_info(runner)
    print(f"Sources: {list(sources.keys())}")
    for name, src in sources.items():
        print(f"  {name}: type={src['type']}, nodes=({src['node_p']}, {src['node_n']})")
    print()

    # For voltage sources, add their conductance to G matrix
    source_fn, G_vs = build_source_current_fn(sources, G, n_unknowns)
    for name, src in sources.items():
        node_p, node_n = src['node_p'], src['node_n']
        if node_p > 0:
            G = G.at[node_p-1, node_p-1].add(G_vs)
        if node_n > 0:
            G = G.at[node_n-1, node_n-1].add(G_vs)
        if node_p > 0 and node_n > 0:
            G = G.at[node_p-1, node_n-1].add(-G_vs)
            G = G.at[node_n-1, node_p-1].add(-G_vs)

    # Check for zero diagonal in C (nodes without capacitance)
    # These are algebraic constraints, need different handling
    C_diag = jnp.diag(C)
    has_capacitance = C_diag > 0
    print(f"Nodes with capacitance: {jnp.sum(has_capacitance).item()}/{n_unknowns}")

    if not jnp.all(has_capacitance):
        # For nodes without capacitance, we need DAE handling
        # For now, add small parasitic capacitance
        C_parasitic = 1e-15  # 1fF parasitic
        C = C + jnp.diag(jnp.where(~has_capacitance, C_parasitic, 0.0))
        print(f"Added {C_parasitic}F parasitic to nodes without capacitance")

    # Invert C for ODE: dV/dt = C_inv @ (I_src - G @ V)
    C_inv = jnp.linalg.inv(C)

    # Define ODE for diffrax
    def circuit_ode(t, V, args):
        G, C_inv, source_fn = args
        I_src = source_fn(t)
        return C_inv @ (I_src - G @ V)

    # Solve with diffrax
    term = diffrax.ODETerm(circuit_ode)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-9)

    # Simulation parameters (from VACASK: step=1u, stop=1s, but we'll do shorter)
    t_end = 10e-3  # 10ms (5 time constants for RC with tau=1ms)
    dt0 = 1e-6

    print(f"Simulating for {t_end*1000}ms...")
    solution = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=t_end, dt0=dt0,
        y0=jnp.zeros(n_unknowns),
        args=(G, C_inv, source_fn),
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0, t_end, 200))
    )

    print(f"Solution shape: {solution.ys.shape}")
    print()

    # Print trajectory at key times
    print("Trajectory (V at node 2, capacitor voltage):")
    for t_ms in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        idx = int(jnp.argmin(jnp.abs(solution.ts - t_ms/1000)))
        t_actual = float(solution.ts[idx]) * 1000
        # Node 2 is index 1 (0-indexed after removing ground)
        if n_unknowns > 1:
            V_cap = float(solution.ys[idx, 1])  # Node 2 voltage
        else:
            V_cap = float(solution.ys[idx, 0])
        print(f"  t={t_actual:.2f}ms: V_cap={V_cap:.4f}V")

    # For pulse train input with period=2ms, width=1ms (50% duty cycle):
    # - Capacitor charges during "on" phase (1ms)
    # - Capacitor discharges during "off" phase (1ms)
    # After several cycles, reaches periodic steady state
    # At start of period (t=0, 2ms, 4ms, etc): minimum voltage
    # At end of "on" phase (t=1ms, 3ms, 5ms, etc): maximum voltage
    #
    # For tau=RC=1ms and period=2ms:
    # - Charge for 1 tau, discharge for 1 tau
    # - V_max ≈ 1 * (1 - e^-1) ≈ 0.632 (from 0 to 1V)
    # - V_min ≈ V_max * e^-1 ≈ 0.232 (discharge back)
    # In steady state: V oscillates between ~0.27V and ~0.73V

    # Check that we see the expected oscillation pattern
    # Find peak voltage (should be around 0.6-0.8V)
    V_cap_series = solution.ys[:, 1] if n_unknowns > 1 else solution.ys[:, 0]
    V_max = float(jnp.max(V_cap_series))
    V_min = float(jnp.min(V_cap_series[50:]))  # Skip initial transient

    print()
    print(f"Pulse train response (period=2ms, width=1ms, tau=1ms):")
    print(f"  V_max (peak): {V_max:.4f}V")
    print(f"  V_min (trough): {V_min:.4f}V")
    print(f"  V_swing: {V_max - V_min:.4f}V")

    # Success criteria: reasonable oscillation pattern
    # V_max should be in range [0.5, 0.9]
    # V_min should be in range [0.1, 0.4]
    # Swing should be > 0.2V
    success = (0.5 < V_max < 0.9) and (0.1 < V_min < 0.4) and (V_max - V_min > 0.2)
    print(f"{'✓ PASS' if success else '✗ FAIL'}")
    print()

    return success


# =============================================================================
# Main
# =============================================================================

def main():
    results = []

    results.append(("Single RC", experiment_single_rc()))
    results.append(("Multi-node RC", experiment_multinode()))

    # Experiment 3: Parse VACASK circuit
    result = experiment_vacask_rc()
    if result is not None:
        results.append(("VACASK RC", result))

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        print(f"  {name}: {'✓ PASS' if passed else '✗ FAIL'}")

    return 0 if all(p for _, p in results) else 1


if __name__ == "__main__":
    exit(main())
