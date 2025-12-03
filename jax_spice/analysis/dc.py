"""DC operating point analysis for JAX-SPICE

Uses Newton-Raphson iteration to find the DC operating point
where all capacitor currents are zero and the circuit is in equilibrium.
"""

from typing import Dict, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve
import numpy as np

from jax_spice.analysis.mna import MNASystem
from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.sparse import sparse_solve_csr


def dc_operating_point(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 50,
    abstol: float = 1e-12,
    reltol: float = 1e-3,
    damping: float = 1.0,
) -> Tuple[Array, Dict]:
    """Find DC operating point using Newton-Raphson iteration
    
    Solves the nonlinear system: f(V) = 0
    where f is the sum of currents at each node (KCL).
    
    Newton-Raphson update:
        J(V_k) * delta_V = -f(V_k)
        V_{k+1} = V_k + damping * delta_V
    
    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
                      If None, starts from zero
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1)
        
    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information
    """
    n = system.num_nodes

    # Use float32 on Metal (no float64 support), float64 elsewhere
    dtype = jnp.float32 if jax.default_backend() == 'METAL' else jnp.float64

    # Initialize solution
    if initial_guess is not None:
        V = jnp.array(initial_guess, dtype=dtype)
    else:
        V = jnp.zeros(n, dtype=dtype)
    
    # Create DC context
    context = AnalysisContext(
        time=None,  # DC analysis
        dt=None,
        analysis_type='dc'
    )
    
    converged = False
    iterations = 0
    residual_history = []
    
    for iteration in range(max_iterations):
        context.iteration = iteration
        
        # Build Jacobian and residual
        J, f = system.build_jacobian_and_residual(V, context)
        
        # Check residual norm for convergence
        residual_norm = jnp.max(jnp.abs(f))
        residual_history.append(float(residual_norm))
        
        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break
        
        # Solve for Newton update: J * delta_V = -f
        try:
            delta_V = solve(J, -f)
        except Exception as e:
            # Matrix is singular - try with regularization
            reg = 1e-12 * jnp.eye(J.shape[0])
            delta_V = solve(J + reg, -f)
        
        # Update solution with damping FIRST
        # Note: V[0] is ground, stays at 0
        V = V.at[1:].add(damping * delta_V)
        iterations = iteration + 1

        # THEN check delta for convergence
        delta_norm = jnp.max(jnp.abs(delta_V))
        v_norm = jnp.max(jnp.abs(V[1:]))  # Exclude ground

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break
    
    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': float(residual_norm),
        'delta_norm': float(delta_norm) if 'delta_norm' in dir() else 0.0,
        'residual_history': residual_history,
    }

    return V, info


def dc_operating_point_sparse(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    max_iterations: int = 100,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    vdd: float = 1.2,
    init_supplies: bool = True,
    verbose: bool = False,
    source_stepping: bool = False,
    source_steps: int = 5,
) -> Tuple[Array, Dict]:
    """Find DC operating point using sparse Newton-Raphson iteration

    This version uses sparse matrix assembly and sparse linear solvers,
    which is much more memory-efficient for large circuits (>1000 nodes).

    Memory comparison for 5000-node circuit:
    - Dense: ~200MB per Jacobian matrix
    - Sparse: ~5MB (assuming 0.1% fill)

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
                      If None, starts from zero with supply nodes initialized
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1), or 'auto' for adaptive
        vdd: Supply voltage for initialization and clamping
        init_supplies: If True, initialize nodes with 'vdd' in name to vdd
        verbose: Print iteration details

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information
    """
    n = system.num_nodes

    # Initialize solution
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

        # Initialize supply nodes if requested
        if init_supplies:
            for name, idx in system.node_names.items():
                name_lower = name.lower()
                if 'vdd' in name_lower:
                    V[idx] = vdd

    # Find supply node indices (these have fixed voltage via voltage sources)
    # Their residual represents supply current, not an error
    supply_node_indices = set()
    for name, idx in system.node_names.items():
        name_lower = name.lower()
        if name_lower in ('vdd', 'vss', 'gnd', '0') or 'vdd' in name_lower:
            if idx > 0:  # Skip ground (index 0)
                supply_node_indices.add(idx - 1)  # Convert to reduced index

    # Create DC context
    context = AnalysisContext(
        time=0.0,
        dt=1e-9,
        analysis_type='dc',
        c0=0.0,
        c1=0.0,
        rhs_correction=0.0,
    )

    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    for iteration in range(max_iterations):
        context.iteration = iteration

        # Build sparse Jacobian and residual
        (data, indices, indptr, shape), f = system.build_sparse_jacobian_and_residual(
            jnp.array(V), context
        )

        # Check residual norm for convergence, excluding supply nodes
        # Supply nodes have residual = supply current, not an error
        f_check = f.copy()
        for idx in supply_node_indices:
            f_check[idx] = 0.0
        residual_norm = float(np.max(np.abs(f_check)))
        residual_norm_full = float(np.max(np.abs(f)))
        residual_history.append(residual_norm)

        if verbose and iteration < 20:
            print(f"    Iter {iteration}: residual={residual_norm:.2e} (full={residual_norm_full:.2e}), "
                  f"V_max={np.max(V[1:]):.4f}, V_min={np.min(V[1:]):.4f}")

        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break

        # Solve for Newton update using sparse solver
        # J * delta_V = -f
        delta_V = sparse_solve_csr(
            jnp.array(data),
            jnp.array(indices),
            jnp.array(indptr),
            jnp.array(-f),
            shape
        )
        delta_V = np.array(delta_V)

        # Apply damping with voltage step limiting
        max_step = 2.0  # Maximum voltage change per iteration
        max_delta = np.max(np.abs(delta_V))
        step_scale = min(damping, max_step / (max_delta + 1e-15))

        # Update solution (skip ground node at index 0)
        V[1:] += step_scale * delta_V

        # Clamp voltages to reasonable range for CMOS circuits
        # Use +/- 2*Vdd to allow for some overshoot while preventing runaway
        v_clamp = vdd * 2.0
        V = np.clip(V, -v_clamp, v_clamp)

        iterations = iteration + 1

        # Check delta for convergence
        delta_norm = float(np.max(np.abs(step_scale * delta_V)))
        v_norm = float(np.max(np.abs(V[1:])))

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break

    # Convert back to JAX array
    V_jax = jnp.array(V)

    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': residual_norm,
        'delta_norm': delta_norm,
        'residual_history': residual_history,
    }

    return V_jax, info


def dc_operating_point_gmin_stepping(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    start_gmin: float = 1e-2,
    target_gmin: float = 1e-12,
    gmin_factor: float = 10.0,
    max_gmin_steps: int = 20,
    max_iterations_per_step: int = 50,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    vdd: float = 1.2,
    init_supplies: bool = True,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """Find DC operating point using GMIN stepping for difficult circuits

    GMIN stepping is a homotopy method that starts with a large minimum
    conductance (GMIN) from each node to ground, making the matrix well-
    conditioned. The GMIN is then gradually reduced to the target value,
    using the previous solution as a warm start.

    This is particularly effective for:
    - Large digital circuits with many cascaded stages
    - Circuits with floating nodes
    - Circuits that fail to converge with standard Newton-Raphson

    Args:
        system: MNA system with devices
        initial_guess: Initial voltage estimate (shape: [num_nodes])
        start_gmin: Initial large GMIN value (default 1e-2)
        target_gmin: Final small GMIN value (default 1e-12)
        gmin_factor: Factor to reduce GMIN by each step (default 10.0)
        max_gmin_steps: Maximum number of GMIN stepping iterations
        max_iterations_per_step: Max NR iterations per GMIN step
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
        damping: Damping factor (0 < damping <= 1)
        vdd: Supply voltage for initialization and clamping
        init_supplies: If True, initialize nodes with 'vdd' in name to vdd
        verbose: Print progress information

    Returns:
        Tuple of (solution, info) where:
            solution: Node voltages (shape: [num_nodes])
            info: Dict with convergence information including gmin_steps
    """
    n = system.num_nodes

    # Initialize solution
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

        # Initialize supply nodes if requested
        if init_supplies:
            for name, idx in system.node_names.items():
                name_lower = name.lower()
                if 'vdd' in name_lower:
                    V[idx] = vdd

    gmin = start_gmin
    total_iterations = 0
    gmin_steps = 0
    all_residual_history = []

    if verbose:
        print(f"GMIN stepping: {start_gmin:.0e} -> {target_gmin:.0e} (factor {gmin_factor})", flush=True)

    while gmin >= target_gmin:
        gmin_steps += 1

        if verbose:
            print(f"  GMIN step {gmin_steps}: gmin={gmin:.0e}", flush=True)

        # Run sparse Newton-Raphson with current GMIN
        # We need to pass gmin via the context, which is created inside dc_operating_point_sparse
        # But dc_operating_point_sparse doesn't accept gmin, so we need to call a lower-level
        # function or modify dc_operating_point_sparse. Let's create an internal version.
        V_jax, info = _dc_solve_with_gmin(
            system,
            initial_guess=V,
            gmin=gmin,
            max_iterations=max_iterations_per_step,
            abstol=abstol,
            reltol=reltol,
            damping=damping,
            vdd=vdd,
            verbose=False,
        )

        V = np.array(V_jax)
        total_iterations += info['iterations']
        all_residual_history.extend(info['residual_history'])

        if verbose:
            print(f"    -> iter={info['iterations']}, residual={info['residual_norm']:.2e}, "
                  f"converged={info['converged']}", flush=True)

        if not info['converged']:
            # Back off: increase GMIN slightly and retry
            gmin = min(gmin * 2.0, start_gmin)
            if verbose:
                print(f"    Not converged, backing off to gmin={gmin:.0e}", flush=True)

            if gmin_steps > max_gmin_steps:
                if verbose:
                    print(f"  GMIN stepping failed after {max_gmin_steps} steps", flush=True)
                break
            continue

        # Converged at this GMIN level
        if gmin <= target_gmin:
            # Reached target GMIN
            break

        # Reduce GMIN for next step
        gmin = max(gmin / gmin_factor, target_gmin)

    final_converged = info['converged'] and gmin <= target_gmin

    result_info = {
        'converged': final_converged,
        'iterations': total_iterations,
        'gmin_steps': gmin_steps,
        'final_gmin': gmin,
        'residual_norm': info['residual_norm'],
        'delta_norm': info.get('delta_norm', 0.0),
        'residual_history': all_residual_history,
    }

    if verbose:
        print(f"  GMIN stepping complete: gmin_steps={gmin_steps}, "
              f"total_iter={total_iterations}, converged={final_converged}", flush=True)

    return jnp.array(V), result_info


def _dc_solve_with_gmin(
    system: MNASystem,
    initial_guess: Optional[Array] = None,
    gmin: float = 1e-12,
    max_iterations: int = 100,
    abstol: float = 1e-9,
    reltol: float = 1e-3,
    damping: float = 1.0,
    vdd: float = 1.2,
    verbose: bool = False,
) -> Tuple[Array, Dict]:
    """Internal DC solver with explicit GMIN parameter

    This is a variant of dc_operating_point_sparse that accepts an explicit
    GMIN value for use with GMIN stepping.
    """
    n = system.num_nodes

    # Initialize solution
    if initial_guess is not None:
        V = np.array(initial_guess, dtype=np.float64)
    else:
        V = np.zeros(n, dtype=np.float64)

    # Find supply node indices (these have fixed voltage via voltage sources)
    supply_node_indices = set()
    for name, idx in system.node_names.items():
        name_lower = name.lower()
        if name_lower in ('vdd', 'vss', 'gnd', '0') or 'vdd' in name_lower:
            if idx > 0:
                supply_node_indices.add(idx - 1)

    # Create DC context with specified GMIN
    context = AnalysisContext(
        time=0.0,
        dt=1e-9,
        analysis_type='dc',
        c0=0.0,
        c1=0.0,
        rhs_correction=0.0,
        gmin=gmin,  # Pass explicit GMIN
    )

    converged = False
    iterations = 0
    residual_norm = 1e20
    delta_norm = 0.0
    residual_history = []

    for iteration in range(max_iterations):
        context.iteration = iteration

        # Build sparse Jacobian and residual
        (data, indices, indptr, shape), f = system.build_sparse_jacobian_and_residual(
            jnp.array(V), context
        )

        # Check residual norm for convergence, excluding supply nodes
        f_check = f.copy()
        for idx in supply_node_indices:
            f_check[idx] = 0.0
        residual_norm = float(np.max(np.abs(f_check)))
        residual_history.append(residual_norm)

        if verbose and iteration < 20:
            print(f"      Iter {iteration}: residual={residual_norm:.2e}")

        if residual_norm < abstol:
            converged = True
            iterations = iteration + 1
            break

        # Solve for Newton update using sparse solver
        delta_V = sparse_solve_csr(
            jnp.array(data),
            jnp.array(indices),
            jnp.array(indptr),
            jnp.array(-f),
            shape
        )
        delta_V = np.array(delta_V)

        # Apply damping with voltage step limiting
        max_step = 2.0
        max_delta = np.max(np.abs(delta_V))
        step_scale = min(damping, max_step / (max_delta + 1e-15))

        # Update solution (skip ground node at index 0)
        V[1:] += step_scale * delta_V

        # Clamp voltages to reasonable range
        v_clamp = vdd * 2.0
        V = np.clip(V, -v_clamp, v_clamp)

        iterations = iteration + 1

        # Check delta for convergence
        delta_norm = float(np.max(np.abs(step_scale * delta_V)))
        v_norm = float(np.max(np.abs(V[1:])))

        if delta_norm < abstol + reltol * max(v_norm, 1.0):
            converged = True
            break

    V_jax = jnp.array(V)

    info = {
        'converged': converged,
        'iterations': iterations,
        'residual_norm': residual_norm,
        'delta_norm': delta_norm,
        'residual_history': residual_history,
    }

    return V_jax, info
