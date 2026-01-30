"""SPICE-compatible voltage limiting functions for NR convergence.

These functions implement the classic SPICE limiting algorithms (pnjlim, fetlim)
that compress large voltage changes during Newton-Raphson iteration to improve
convergence. Without limiting, large voltage steps can cause device models to
evaluate at unrealistic operating points, leading to poor convergence.

The key insight is that PN junction currents are exponential in voltage:
    I = Is * (exp(V/Vt) - 1)

A 1V change in junction voltage can cause current to change by a factor of
e^(1/0.026) ≈ 2e16. By using logarithmic compression, we limit the step size
while still making progress toward the solution.

Reference: SPICE3 source code, lib/dev/diode/diotemp.c
"""

import jax.numpy as jnp
from jax import Array


def pnjlim(vnew: Array, vold: Array, vt: float = 0.026, vcrit: float = 0.6) -> Array:
    """PN junction voltage limiting (logarithmic damping).

    Applies logarithmic compression to large voltage changes across PN junctions.
    This is the classic SPICE pnjlim algorithm.

    The algorithm:
    1. If the new voltage is above vcrit AND the change exceeds 2*vt, apply limiting
    2. For positive changes: vnew = vold + vt * log(1 + (vnew - vold)/vt)
    3. For negative changes: vnew = vold - vt * log(1 - (vnew - vold)/vt)

    This compresses large steps while preserving the sign and direction of change.

    Args:
        vnew: Proposed new voltage (from NR step)
        vold: Previous voltage
        vt: Thermal voltage (kT/q ≈ 0.026V at 300K)
        vcrit: Critical voltage above which limiting is applied (typ. 0.6V)

    Returns:
        Limited voltage that is closer to vold than vnew when the step is large
    """
    delta_v = vnew - vold

    # Condition: above critical voltage AND large change
    large_positive = (vnew > vcrit) & (delta_v > 2 * vt)
    large_negative = (vnew > vcrit) & (delta_v < -2 * vt)

    # Logarithmic compression for positive changes
    # vnew = vold + vt * log(1 + delta/vt)
    # This maps delta -> vt * log(1 + delta/vt), compressing large deltas
    arg_pos = delta_v / vt
    limited_pos = vold + vt * jnp.log1p(jnp.maximum(arg_pos, 0))

    # For negative changes, use symmetric formula
    # vnew = vold - vt * log(1 - delta/vt) = vold - vt * log(1 + |delta|/vt)
    arg_neg = -delta_v / vt
    limited_neg = vold - vt * jnp.log1p(jnp.maximum(arg_neg, 0))

    # Apply limiting where needed
    result = jnp.where(large_positive & (vold > 0), limited_pos, vnew)
    result = jnp.where(large_negative & (vold > 0), limited_neg, result)

    return result


def fetlim(vnew: Array, vold: Array, vto: float = 0.5) -> Array:
    """FET gate-source voltage limiting (region-based).

    Limits gate-source voltage changes based on operating region.
    In the on-region (Vgs > Vto), larger steps are allowed.
    In the off-region, steps are limited to 0.5V.

    Args:
        vnew: Proposed new Vgs voltage
        vold: Previous Vgs voltage
        vto: Threshold voltage (default 0.5V for NMOS)

    Returns:
        Limited voltage
    """
    # Determine operating region
    in_on_region = vold >= vto

    # Allow larger steps when well into on-region
    # Step limit = 2 * |vold - vto| + 2 when on, 0.5 when off
    max_step = jnp.where(
        in_on_region,
        2 * jnp.abs(vold - vto) + 2,
        0.5
    )

    delta = vnew - vold
    limited = vold + jnp.clip(delta, -max_step, max_step)

    return limited


def apply_voltage_damping(
    V_new: Array,
    V_old: Array,
    vt: float = 0.026,
    vcrit: float = 0.6,
    max_step: float = 0.3
) -> Array:
    """Apply voltage damping to all voltage updates.

    This combines two damping strategies:
    1. pnjlim-style logarithmic compression for PN junction regions
       (where vnew > vcrit and vold > 0)
    2. General step limiting for large voltage changes that don't
       trigger pnjlim (important for MOSFETs where Vgs may start at 0)

    The general step limiter caps voltage changes at max_step volts,
    which prevents the solver from making huge jumps that could cause
    device evaluation at unrealistic operating points.

    Args:
        V_new: Proposed new voltage vector (from NR step, excluding ground)
        V_old: Previous voltage vector (excluding ground)
        vt: Thermal voltage (kT/q ≈ 0.026V at 300K)
        vcrit: Critical voltage above which pnjlim is applied
        max_step: Maximum voltage step for general limiting (default 0.3V)

    Returns:
        Damped voltage vector
    """
    # First apply pnjlim for PN junction regions
    result = pnjlim(V_new, V_old, vt, vcrit)

    # Then apply general step limiting for nodes where pnjlim didn't help
    # This catches MOSFET Vgs starting from 0V where vold > 0 condition fails
    delta = result - V_old
    abs_delta = jnp.abs(delta)

    # Only apply step limiting where:
    # 1. The change is large (> max_step)
    # 2. pnjlim didn't already reduce it significantly
    needs_limiting = abs_delta > max_step

    # Logarithmic compression for large steps (like pnjlim but without vold>0 condition)
    # This compresses delta while preserving direction
    compressed_delta = jnp.sign(delta) * max_step * jnp.log1p(abs_delta / max_step)
    limited = V_old + compressed_delta

    # Apply step limiting only where needed
    result = jnp.where(needs_limiting, limited, result)

    return result
