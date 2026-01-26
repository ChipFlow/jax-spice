"""Adaptive timestep configuration and utilities.

This module provides LTE-based adaptive timestep control utilities used by
FullMNAStrategy. The algorithm:

1. Use polynomial extrapolation to predict solution at next timestep
2. Solve with Newton-Raphson (corrector step)
3. Estimate Local Truncation Error (LTE) from predictor-corrector difference
4. Adjust timestep based on LTE and tolerance requirements
5. Accept or reject the step based on error threshold

Reference: VACASK coretran.cpp lines 1070-1220
"""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive timestep control.

    Attributes:
        lte_ratio: LTE tolerance multiplier (tran_lteratio). Higher values
            allow larger LTE before reducing timestep. Default 3.5.
        redo_factor: Rejection threshold (tran_redofactor). If current_dt /
            new_dt exceeds this, the timestep is rejected. Default 2.5.
        reltol: Relative tolerance for LTE comparison. Default 1e-3.
        abstol: Absolute tolerance for LTE comparison. Default 1e-12.
        min_dt: Minimum allowed timestep. Default 1e-18 seconds.
        max_dt: Maximum allowed timestep. Default infinity (no limit).
        warmup_steps: Number of fixed-dt steps before enabling LTE control.
            Need at least 2 past solutions for linear extrapolation. Default 2.
        max_order: Maximum order for polynomial predictor. Default 2 (quadratic).
        grow_factor: Maximum factor by which timestep can grow per step. Default 2.0.
        nr_convtol: NR convergence tolerance factor (multiplier on abstol). Default 1.0.
        gshunt_init: Initial gshunt value for convergence aid. Default 0.0.
        gshunt_steps: Number of steps to ramp gshunt from init to target. Default 5.
        gshunt_target: Final gshunt value after ramping. Default 0.0.
    """

    lte_ratio: float = 3.5
    redo_factor: float = 2.5
    reltol: float = 1e-3
    abstol: float = 1e-12
    min_dt: float = 1e-18
    max_dt: float = float("inf")
    warmup_steps: int = 2
    max_order: int = 2
    grow_factor: float = 2.0
    nr_convtol: float = 1.0
    gshunt_init: float = 0.0
    gshunt_steps: int = 5
    gshunt_target: float = 0.0


def predict_voltage_jax(
    V_history: jax.Array,           # (max_history, n_total)
    dt_history: jax.Array,          # (max_history,)
    history_count: jax.Array,       # Scalar
    new_dt: jax.Array,              # Scalar
    max_order: int,
) -> Tuple[jax.Array, jax.Array]:
    """Compute predictor using fixed-size arrays (JAX-compatible).

    This is the shared predictor function used by FullMNAStrategy.

    Args:
        V_history: Ring buffer of past voltage vectors (max_history, n_total)
        dt_history: Ring buffer of past timesteps (max_history,)
        history_count: Number of valid history entries
        new_dt: Timestep to predict for
        max_order: Maximum polynomial order (0=constant, 1=linear, 2=quadratic)

    Returns:
        Tuple of (V_predicted, error_coeff)
    """
    # Compute actual order to use (min of history_count-1 and max_order)
    order = jnp.minimum(history_count - 1, max_order)
    order = jnp.maximum(order, 0)  # At least 0

    # Compute normalized timepoints: tau_k = (t_{n-k} - t_{n+1}) / h_n
    # t_n is at -new_dt, t_{n-1} is at -(new_dt + dt_history[0]), etc.
    cumsum_dt = jnp.cumsum(dt_history)  # Cumulative sum of past timesteps
    tau = -(new_dt + jnp.concatenate([jnp.array([0.0]), cumsum_dt[:-1]])) / new_dt

    # Build Vandermonde-like matrix for polynomial interpolation
    # For order p, we need p+1 points and solve for coefficients
    # Use order=2 (quadratic) as max, pad smaller orders with zeros

    # Simple approach: compute coefficients for each possible order
    # and select based on actual order

    # Order 0 (constant): a = [1]
    a0 = jnp.array([1.0, 0.0, 0.0])
    err0 = 1.0

    # Order 1 (linear): V_pred = V_n + (V_n - V_{n-1}) * new_dt / dt_{n-1}
    # Coefficients: a_0 = 1 + new_dt/dt_0, a_1 = -new_dt/dt_0
    ratio1 = new_dt / jnp.maximum(dt_history[0], 1e-30)
    a1 = jnp.array([1.0 + ratio1, -ratio1, 0.0])
    err1 = -ratio1 * (1 + ratio1) / 2  # Approximate error coefficient

    # Order 2 (quadratic): more complex, use Lagrange interpolation
    # tau values for 3 points (tau[0] = t_n position, tau[1] = t_{n-1}, tau[2] = t_{n-2})
    tau0 = -1.0  # t_n relative to t_{n+1}, normalized (same as tau[0])
    tau1 = tau[1]  # t_{n-1}
    tau2 = tau[2]  # t_{n-2}

    # Lagrange coefficients for interpolating to tau=0 (t_{n+1})
    denom0 = (tau0 - tau1) * (tau0 - tau2)
    denom1 = (tau1 - tau0) * (tau1 - tau2)
    denom2 = (tau2 - tau0) * (tau2 - tau1)

    # Avoid division by zero
    denom0 = jnp.where(jnp.abs(denom0) < 1e-30, 1e-30, denom0)
    denom1 = jnp.where(jnp.abs(denom1) < 1e-30, 1e-30, denom1)
    denom2 = jnp.where(jnp.abs(denom2) < 1e-30, 1e-30, denom2)

    l0 = (0 - tau1) * (0 - tau2) / denom0  # For V_n
    l1 = (0 - tau0) * (0 - tau2) / denom1  # For V_{n-1}
    l2 = (0 - tau0) * (0 - tau1) / denom2  # For V_{n-2}

    a2 = jnp.array([l0, l1, l2])
    err2 = -tau0 * tau1 * tau2 / 6  # Third derivative contribution

    # Select coefficients based on order
    a = jnp.where(order == 0, a0, jnp.where(order == 1, a1, a2))
    err_coeff = jnp.where(order == 0, err0, jnp.where(order == 1, err1, err2))

    # Compute prediction: V_pred = sum_i a_i * V_history[i]
    # Only use valid entries based on order
    mask = jnp.arange(3) <= order
    a_masked = jnp.where(mask, a, 0.0)

    # V_history is (max_history, n_total), we need first 3 rows
    V_pred = jnp.einsum('i,ij->j', a_masked[:3], V_history[:3])

    return V_pred, err_coeff


def compute_lte_timestep_jax(
    V_new: jax.Array,
    V_pred: jax.Array,
    pred_err_coeff: jax.Array,
    dt: jax.Array,
    history_count: jax.Array,
    config: AdaptiveConfig,
    error_coeff_integ: float = -0.5,
) -> Tuple[jax.Array, jax.Array]:
    """Compute LTE-based new timestep (JAX-compatible).

    This is the shared LTE calculation used by FullMNAStrategy.

    Args:
        V_new: Corrected solution from Newton-Raphson
        V_pred: Predicted solution from extrapolation
        pred_err_coeff: Predictor error coefficient
        dt: Current timestep
        history_count: Number of valid history entries
        config: Adaptive timestep configuration
        error_coeff_integ: Integration method error coefficient (default -0.5 for BE)

    Returns:
        Tuple of (dt_new, lte_norm)
    """
    # Estimate LTE from predictor-corrector difference
    lte = (V_new - V_pred) * (error_coeff_integ / (error_coeff_integ - pred_err_coeff))
    lte_norm = jnp.max(jnp.abs(lte) / (config.reltol * jnp.abs(V_new) + config.abstol))

    # Compute new timestep: dt_new = dt * (lte_ratio / lte_norm)^(1/(order+2))
    order = jnp.minimum(history_count - 1, config.max_order)
    factor = jnp.power(config.lte_ratio / jnp.maximum(lte_norm, 1e-10), 1.0 / (order + 2))
    factor = jnp.clip(factor, 0.1, config.grow_factor)
    dt_new = dt * factor

    return dt_new, lte_norm
