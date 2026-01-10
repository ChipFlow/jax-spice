"""Modified Nodal Analysis (MNA) types for JAX-SPICE

This module provides core types used by the VACASK benchmark runner.
The actual MNA matrix assembly is done in the runner using OpenVAF-compiled models.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# =============================================================================
# Parameter Evaluation
# =============================================================================


def eval_param_simple(value, vdd: float = 1.2, defaults: dict = None):
    """Simple parameter evaluation for common cases.

    Handles:
    - Numbers (int, float)
    - String 'vdd' -> vdd value
    - String '0' -> 0.0
    - String 'w', 'l' -> default MOSFET dimensions
    - SPICE number suffixes (1u, 100n, etc.)
    """
    if defaults is None:
        defaults = {
            'w': 1e-6,      # Default MOSFET width = 1u
            'l': 0.2e-6,    # Default MOSFET length = 0.2u
            'ld': 0.5e-6,   # Default drain extension
            'ls': 0.5e-6,   # Default source extension
        }

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value_lower = value.lower().strip()

        # Common parameter references
        if value_lower == 'vdd':
            return vdd
        if value_lower in ('0', '0.0', 'vss', 'gnd'):
            return 0.0
        if value_lower in defaults:
            return defaults[value_lower]

        # SPICE number suffixes
        suffixes = {
            't': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3,
            'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15
        }
        for suffix, mult in sorted(suffixes.items(), key=lambda x: -len(x[0])):
            if value_lower.endswith(suffix):
                try:
                    return float(value_lower[:-len(suffix)]) * mult
                except ValueError:
                    pass

        # Try direct conversion
        try:
            return float(value)
        except ValueError:
            pass

    return 0.0


# =============================================================================
# Device Type Enumeration
# =============================================================================


class DeviceType(Enum):
    """Enumeration of supported device types.

    All non-source devices use OpenVAF (VERILOG_A).
    VSOURCE and ISOURCE are handled with simple large-conductance models.
    """
    VSOURCE = 'vsource'
    ISOURCE = 'isource'
    VERILOG_A = 'verilog_a'  # All OpenVAF-compiled devices (resistor, capacitor, diode, psp103, etc.)


# =============================================================================
# Device Info
# =============================================================================


@dataclass
class DeviceInfo:
    """Information about a device instance for simulation."""
    name: str
    model_name: str
    terminals: List[str]  # Terminal names
    node_indices: List[int]  # Corresponding node indices
    params: Dict[str, Any]  # Instance parameters
    eval_fn: Optional[Callable] = None  # Device evaluation function
    is_openvaf: bool = False  # True if device uses OpenVAF-compiled Verilog-A model
