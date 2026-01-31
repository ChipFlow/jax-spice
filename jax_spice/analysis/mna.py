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
            "w": 1e-6,  # Default MOSFET width = 1u
            "l": 0.2e-6,  # Default MOSFET length = 0.2u
            "ld": 0.5e-6,  # Default drain extension
            "ls": 0.5e-6,  # Default source extension
        }

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value_lower = value.lower().strip()

        # Common parameter references
        if value_lower == "vdd":
            return vdd
        if value_lower in ("0", "0.0", "vss", "gnd"):
            return 0.0
        if value_lower in defaults:
            return defaults[value_lower]

        # SPICE number suffixes
        suffixes = {
            "t": 1e12,
            "g": 1e9,
            "meg": 1e6,
            "k": 1e3,
            "m": 1e-3,
            "u": 1e-6,
            "n": 1e-9,
            "p": 1e-12,
            "f": 1e-15,
        }
        for suffix, mult in sorted(suffixes.items(), key=lambda x: -len(x[0])):
            if value_lower.endswith(suffix):
                try:
                    return float(value_lower[: -len(suffix)]) * mult
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

    VSOURCE = "vsource"
    ISOURCE = "isource"
    VERILOG_A = (
        "verilog_a"  # All OpenVAF-compiled devices (resistor, capacitor, diode, psp103, etc.)
    )


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


# =============================================================================
# Full MNA Branch Current Structures
# =============================================================================


@dataclass
class BranchInfo:
    """Information about a branch current unknown in full MNA formulation.

    In full MNA, voltage sources have their currents as explicit unknowns.
    The augmented system becomes:

        ┌───────────────┐   ┌───┐   ┌───┐
        │     G     B   │   │ V │   │ I │
        │               │ × │   │ = │   │
        │    B^T    0   │   │ J │   │ E │
        └───────────────┘   └───┘   └───┘

    Where:
    - G = device conductance matrix (n×n)
    - B = incidence matrix mapping branch currents to nodes (n×m)
    - V = node voltages (n×1)
    - J = branch currents (m×1) - these are the primary unknowns for vsources
    - I = device current sources (n×1)
    - E = voltage source values (m×1)

    Attributes:
        name: Source device name (e.g., 'vdd', 'v1')
        node_p: Positive terminal node index
        node_n: Negative terminal node index
        branch_idx: Index into the branch current solution vector
        dc_value: DC voltage value (for initialization)
    """

    name: str
    node_p: int
    node_n: int
    branch_idx: int
    dc_value: float = 0.0


@dataclass
class MNABranchData:
    """Data structure for full MNA branch currents.

    Contains all information needed to:
    1. Build the augmented MNA matrix with B and B^T blocks
    2. Extract branch currents from the solution vector
    3. Map source names to their branch current indices

    Attributes:
        n_branches: Number of branch current unknowns (= number of vsources)
        branches: List of BranchInfo for each voltage source
        name_to_idx: Mapping from source name to branch index
        node_p: Array of positive terminal indices for all vsources
        node_n: Array of negative terminal indices for all vsources
        dc_values: Array of DC voltage values
    """

    n_branches: int
    branches: List[BranchInfo]
    name_to_idx: Dict[str, int]
    node_p: List[int]
    node_n: List[int]
    dc_values: List[float]

    @classmethod
    def from_devices(cls, devices: List[Dict], node_names: Dict[str, int]) -> "MNABranchData":
        """Create MNABranchData from device list.

        Args:
            devices: List of device dicts with 'model', 'name', 'nodes', 'params'
            node_names: Mapping from node name to node index

        Returns:
            MNABranchData with branch info for all voltage sources
        """
        branches = []
        name_to_idx = {}
        node_p_list = []
        node_n_list = []
        dc_values = []

        branch_idx = 0
        for dev in devices:
            if dev.get("model") == "vsource":
                name = dev["name"]
                nodes = dev["nodes"]
                params = dev.get("params", {})

                # Get node indices (nodes are already indices in most cases)
                if len(nodes) >= 2:
                    p_node = (
                        nodes[0] if isinstance(nodes[0], int) else node_names.get(str(nodes[0]), 0)
                    )
                    n_node = (
                        nodes[1] if isinstance(nodes[1], int) else node_names.get(str(nodes[1]), 0)
                    )
                else:
                    p_node = (
                        nodes[0] if isinstance(nodes[0], int) else node_names.get(str(nodes[0]), 0)
                    )
                    n_node = 0  # Ground

                dc_val = float(params.get("dc", 0.0))

                branch = BranchInfo(
                    name=name, node_p=p_node, node_n=n_node, branch_idx=branch_idx, dc_value=dc_val
                )
                branches.append(branch)
                name_to_idx[name] = branch_idx
                node_p_list.append(p_node)
                node_n_list.append(n_node)
                dc_values.append(dc_val)
                branch_idx += 1

        return cls(
            n_branches=len(branches),
            branches=branches,
            name_to_idx=name_to_idx,
            node_p=node_p_list,
            node_n=node_n_list,
            dc_values=dc_values,
        )
