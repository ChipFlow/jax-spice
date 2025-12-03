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

        # Check residual norm for convergence
        residual_norm = float(np.max(np.abs(f)))
        residual_history.append(residual_norm)

        if verbose and iteration < 20:
            print(f"    Iter {iteration}: residual={residual_norm:.2e}, "
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

        # Clamp voltages to reasonable range (relaxed to allow Newton-Raphson exploration)
        V = np.clip(V, -100, 100)

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
