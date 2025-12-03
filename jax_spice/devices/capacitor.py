"""Capacitor device model for JAX-SPICE

Two-terminal capacitor with optional initial condition support.
Critical for transient analysis and MOSFET junction capacitances.
"""

from typing import Dict, Optional, Tuple, TYPE_CHECKING
import jax.numpy as jnp
from jax import Array

from jax_spice.devices.base import DeviceStamps

if TYPE_CHECKING:
    from jax_spice.analysis.context import AnalysisContext


class Capacitor:
    """Two-terminal capacitor device

    For DC analysis: open circuit (no current)
    For transient: I = C * dV/dt, modeled as companion resistor

    Using backward Euler discretization:
        Q(n+1) = C * V(n+1)
        I(n+1) = (Q(n+1) - Q(n)) / dt = C * (V(n+1) - V(n)) / dt

    Companion model (Norton equivalent):
        G_eq = C / dt (equivalent conductance)
        I_eq = C * V(n) / dt (history current source)
        I = G_eq * V(n+1) - I_eq

    Parameters:
        C: Capacitance in Farads (default: 1e-12, 1pF)
        ic: Initial condition voltage (default: None, use DC OP)
        vc1: First-order voltage coefficient (default: 0)
        vc2: Second-order voltage coefficient (default: 0)

    Terminals:
        p: Positive terminal
        n: Negative terminal
    """

    terminals: Tuple[str, str] = ('p', 'n')

    def __init__(self, C: float = 1e-12, ic: Optional[float] = None,
                 vc1: float = 0.0, vc2: float = 0.0):
        """Initialize capacitor with parameters

        Args:
            C: Capacitance value in Farads
            ic: Initial condition voltage (None = use DC operating point)
            vc1: First-order voltage coefficient for C(V)
            vc2: Second-order voltage coefficient for C(V)
        """
        self.C = C
        self.ic = ic
        self.vc1 = vc1
        self.vc2 = vc2

    def evaluate(
        self,
        voltages: Dict[str, float],
        params: Optional[Dict[str, float]] = None,
        context: Optional["AnalysisContext"] = None,
    ) -> DeviceStamps:
        """Evaluate capacitor at given terminal voltages

        Args:
            voltages: Dictionary with 'p' and 'n' terminal voltages
            params: Optional parameter overrides
            context: Analysis context (provides dt for transient, is_dc flag)

        Returns:
            DeviceStamps with current, conductance, and charge
        """
        # Get parameters (allow override)
        if params is None:
            params = {}
        C = params.get('C', self.C)
        vc1 = params.get('vc1', self.vc1)
        vc2 = params.get('vc2', self.vc2)

        # Get terminal voltages
        Vp = voltages.get('p', 0.0)
        Vn = voltages.get('n', 0.0)
        V = Vp - Vn

        # Voltage-dependent capacitance: C(V) = C * (1 + vc1*V + vc2*V^2)
        C_eff = C * (1.0 + vc1 * V + vc2 * V * V)
        C_eff = jnp.maximum(C_eff, 1e-18)  # Minimum capacitance

        # Charge stored
        Q = C_eff * V

        # Check analysis type
        is_dc = True
        dt = None
        V_prev = 0.0

        if context is not None:
            is_dc = getattr(context, 'is_dc', True)
            dt = getattr(context, 'dt', None)
            # Get previous voltage from context for transient
            V_prev = getattr(context, 'v_prev', {}).get('capacitor_V', 0.0)

        # For DC analysis: capacitor is open circuit
        # For transient: return capacitance for companion model in MNA system
        # The transient analysis module will handle the companion model conversion

        # Zero current for DC, small conductance for numerical stability
        I = jnp.array(0.0)
        G_small = jnp.array(1e-15)  # Very small leakage for numerical stability

        return DeviceStamps(
            currents={'p': I, 'n': -I},
            conductances={
                ('p', 'p'): G_small, ('p', 'n'): -G_small,
                ('n', 'p'): -G_small, ('n', 'n'): G_small
            },
            charges={'p': Q, 'n': -Q},
            capacitances={
                ('p', 'p'): C_eff, ('p', 'n'): -C_eff,
                ('n', 'p'): -C_eff, ('n', 'n'): C_eff
            }
        )


def capacitor(Vp: Array, Vn: Array, C: Array) -> Tuple[Array, Array]:
    """Functional capacitor model for charge calculation

    Pure function for use in JAX computations.
    Returns charge (not current) - use with ddt() for transient.

    Args:
        Vp: Positive terminal voltage
        Vn: Negative terminal voltage
        C: Capacitance in Farads

    Returns:
        Tuple of (charge, capacitance) at p terminal
    """
    C_safe = jnp.maximum(C, 1e-18)
    V = Vp - Vn
    Q = C_safe * V
    return Q, C_safe


def capacitor_companion(V: Array, V_prev: Array, C: Array, dt: Array) -> Tuple[Array, Array, Array]:
    """Backward Euler companion model for capacitor

    Used in transient analysis with fixed timestep.

    Args:
        V: Current voltage across capacitor
        V_prev: Previous voltage across capacitor
        C: Capacitance in Farads
        dt: Timestep in seconds

    Returns:
        Tuple of (current, equivalent_conductance, history_current)
    """
    C_safe = jnp.maximum(C, 1e-18)
    dt_safe = jnp.maximum(dt, 1e-18)

    G_eq = C_safe / dt_safe
    I_eq = G_eq * V_prev  # History current source
    I = G_eq * V - I_eq    # Total current

    return I, G_eq, I_eq
