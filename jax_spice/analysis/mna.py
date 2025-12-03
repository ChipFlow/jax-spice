"""Modified Nodal Analysis (MNA) matrix assembly for JAX-SPICE

The MNA formulation:
    G * V + C * dV/dt = I
    
Where:
    G = conductance matrix (from resistive elements)
    C = capacitance matrix (from reactive elements) 
    V = node voltage vector
    I = current source vector

For Newton-Raphson iteration, we solve:
    J * delta_V = -residual
    
Where J is the Jacobian (partial derivatives of residual w.r.t. voltages).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jax_spice.netlist.circuit import Circuit, Instance, Model
from jax_spice.analysis.context import AnalysisContext


# =============================================================================
# Device Type Enumeration
# =============================================================================


class DeviceType(Enum):
    """Enumeration of supported device types for vectorized evaluation"""
    VSOURCE = 'vsource'
    ISOURCE = 'isource'
    RESISTOR = 'resistor'
    CAPACITOR = 'capacitor'
    MOSFET = 'mosfet'
    VERILOG_A = 'verilog_a'
    UNKNOWN = 'unknown'


# =============================================================================
# Vectorized Device Group
# =============================================================================


@dataclass
class VectorizedDeviceGroup:
    """Group of devices of the same type for vectorized evaluation

    This enables GPU-friendly batch processing of device evaluations.
    Instead of iterating over devices one-by-one with Python loops,
    we evaluate all devices of the same type in parallel using JAX.

    Attributes:
        device_type: Type of devices in this group
        n_devices: Number of devices in the group
        device_names: List of device instance names (for debugging)
        node_indices: (n_devices, n_terminals) int array of node indices
        params: Dict of parameter arrays, each shape (n_devices,)
    """
    device_type: DeviceType
    n_devices: int
    device_names: List[str]
    node_indices: Array  # (n_devices, n_terminals) int32 array
    params: Dict[str, Array]  # Parameter name -> (n_devices,) array

    @property
    def n_terminals(self) -> int:
        """Number of terminals per device"""
        return self.node_indices.shape[1] if self.n_devices > 0 else 0


@dataclass
class DeviceInfo:
    """Information about a device instance for simulation"""
    name: str
    model_name: str
    terminals: List[str]  # Terminal names
    node_indices: List[int]  # Corresponding node indices
    params: Dict[str, Any]  # Instance parameters
    eval_fn: Optional[Callable] = None  # Device evaluation function


@dataclass 
class MNASystem:
    """MNA system for circuit simulation
    
    Manages the mapping between circuit topology and matrix entries.
    Supports both dense and sparse matrix assembly.
    
    Attributes:
        num_nodes: Number of nodes (including ground = 0)
        node_names: Mapping from node name to index
        devices: List of device instances
        ground_node: Index of ground node (always 0)
    """
    num_nodes: int
    node_names: Dict[str, int]
    devices: List[DeviceInfo] = field(default_factory=list)
    ground_node: int = 0

    # Model registry: model_name -> (device_class, default_params)
    model_registry: Dict[str, Tuple[Any, Dict]] = field(default_factory=dict)

    # Vectorized device groups for GPU-friendly batch evaluation
    device_groups: List[VectorizedDeviceGroup] = field(default_factory=list)
    
    @classmethod
    def from_circuit(
        cls,
        circuit: Circuit,
        top_subckt: str,
        model_registry: Dict[str, Tuple[Any, Dict]]
    ) -> 'MNASystem':
        """Create MNA system from parsed circuit
        
        Args:
            circuit: Parsed circuit from netlist
            top_subckt: Name of top-level subcircuit to simulate
            model_registry: Mapping from model names to device classes
            
        Returns:
            MNASystem ready for simulation
        """
        # Flatten the circuit hierarchy
        flat_instances, nodes = circuit.flatten(top_subckt)
        
        # Create the system
        system = cls(
            num_nodes=len(nodes),
            node_names=nodes,
            model_registry=model_registry
        )
        
        # Add devices
        for inst in flat_instances:
            # Look up the model
            model_info = circuit.models.get(inst.model)
            if model_info is None:
                raise ValueError(f"Model '{inst.model}' not found for instance '{inst.name}'")
            
            # Look up the device class
            if model_info.module not in model_registry:
                raise ValueError(f"Unknown device module '{model_info.module}' for model '{inst.model}'")
            
            device_class, default_params = model_registry[model_info.module]
            
            # Merge parameters: defaults < model params < instance params
            params = {**default_params, **model_info.params, **inst.params}
            
            # Map terminal names to node indices
            node_indices = [nodes[t] for t in inst.terminals]
            
            device = DeviceInfo(
                name=inst.name,
                model_name=inst.model,
                terminals=inst.terminals,
                node_indices=node_indices,
                params=params,
                eval_fn=device_class
            )
            system.devices.append(device)
        
        return system
    
    def build_jacobian_and_residual(
        self,
        voltages: Array,
        context: AnalysisContext
    ) -> Tuple[Array, Array]:
        """Build Jacobian matrix and residual vector
        
        Args:
            voltages: Current node voltage estimates (shape: [num_nodes])
            context: Analysis context with time, etc.
            
        Returns:
            Tuple of (jacobian, residual) where:
                jacobian: Shape [num_nodes-1, num_nodes-1] (ground eliminated)
                residual: Shape [num_nodes-1]
        """
        n = self.num_nodes - 1  # Exclude ground

        # Use float32 on Metal (no float64 support), float64 elsewhere
        dtype = jnp.float32 if jax.default_backend() == 'METAL' else jnp.float64

        # Initialize Jacobian and residual
        jacobian = jnp.zeros((n, n), dtype=dtype)
        residual = jnp.zeros(n, dtype=dtype)
        
        # Evaluate each device and stamp into matrix
        for device in self.devices:
            if device.eval_fn is None:
                continue
                
            # Build voltage dictionary for this device
            dev_voltages = {}
            for i, (term, node_idx) in enumerate(zip(device.terminals, device.node_indices)):
                dev_voltages[term] = voltages[node_idx]
            
            # Evaluate device
            stamps = device.eval_fn(dev_voltages, device.params, context)
            
            # Stamp currents into residual
            for term, current in stamps.currents.items():
                term_idx = device.terminals.index(term)
                node_idx = device.node_indices[term_idx]
                if node_idx != self.ground_node:
                    residual = residual.at[node_idx - 1].add(current)
            
            # Stamp conductances into Jacobian
            for (term_i, term_j), conductance in stamps.conductances.items():
                idx_i = device.terminals.index(term_i)
                idx_j = device.terminals.index(term_j)
                node_i = device.node_indices[idx_i]
                node_j = device.node_indices[idx_j]
                
                # Skip ground entries
                if node_i != self.ground_node and node_j != self.ground_node:
                    jacobian = jacobian.at[node_i - 1, node_j - 1].add(conductance)
        
        return jacobian, residual
    
    def get_node_voltage(self, solution: Array, node_name: str) -> float:
        """Get voltage at a named node from solution vector
        
        Args:
            solution: Solution vector (length num_nodes - 1, ground excluded)
            node_name: Name of node
            
        Returns:
            Voltage at node (0.0 for ground)
        """
        node_idx = self.node_names.get(node_name)
        if node_idx is None:
            raise ValueError(f"Unknown node: {node_name}")
        if node_idx == self.ground_node:
            return 0.0
        return float(solution[node_idx - 1])
    
    def full_voltage_vector(self, solution: Array) -> Array:
        """Convert solution to full voltage vector including ground

        Args:
            solution: Solution vector (length num_nodes - 1)

        Returns:
            Full voltage vector (length num_nodes) with ground = 0
        """
        return jnp.concatenate([jnp.array([0.0]), solution])

    def build_sparse_jacobian_and_residual(
        self,
        voltages: Array,
        context: AnalysisContext
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]], np.ndarray]:
        """Build Jacobian matrix and residual vector in sparse CSR format

        This is more memory-efficient for large circuits. Uses COO accumulation
        then converts to CSR for the sparse solver.

        Args:
            voltages: Current node voltage estimates (shape: [num_nodes])
            context: Analysis context with time, etc.

        Returns:
            Tuple of ((data, indices, indptr, shape), residual) where:
                data, indices, indptr: CSR format arrays
                shape: (n, n) matrix shape
                residual: Shape [num_nodes-1] numpy array
        """
        n = self.num_nodes - 1  # Exclude ground

        # Accumulate triplets (row, col, value) for COO format
        rows = []
        cols = []
        values = []
        residual = np.zeros(n, dtype=np.float64)

        # Evaluate each device and accumulate stamps
        for device in self.devices:
            if device.eval_fn is None:
                continue

            # Build voltage dictionary for this device
            dev_voltages = {}
            for i, (term, node_idx) in enumerate(zip(device.terminals, device.node_indices)):
                dev_voltages[term] = float(voltages[node_idx])

            # Evaluate device
            stamps = device.eval_fn(dev_voltages, device.params, context)

            # Stamp currents into residual
            for term, current in stamps.currents.items():
                term_idx = device.terminals.index(term)
                node_idx = device.node_indices[term_idx]
                if node_idx != self.ground_node:
                    residual[node_idx - 1] += float(current)

            # Stamp conductances into Jacobian (triplet format)
            for (term_i, term_j), conductance in stamps.conductances.items():
                idx_i = device.terminals.index(term_i)
                idx_j = device.terminals.index(term_j)
                node_i = device.node_indices[idx_i]
                node_j = device.node_indices[idx_j]

                # Skip ground entries
                if node_i != self.ground_node and node_j != self.ground_node:
                    rows.append(node_i - 1)
                    cols.append(node_j - 1)
                    values.append(float(conductance))

        # Add GMIN to all diagonal entries for numerical stability
        # This creates a small conductance from each node to ground
        # Read gmin from context (allows GMIN stepping for convergence)
        gmin = context.gmin if hasattr(context, 'gmin') else 1e-12
        for i in range(n):
            rows.append(i)
            cols.append(i)
            values.append(gmin)

        # Convert to CSR format
        from jax_spice.analysis.sparse import build_csr_arrays

        rows_arr = np.array(rows, dtype=np.int32)
        cols_arr = np.array(cols, dtype=np.int32)
        values_arr = np.array(values, dtype=np.float64)

        data, indices, indptr = build_csr_arrays(
            rows_arr, cols_arr, values_arr, (n, n)
        )

        return (data, indices, indptr, (n, n)), residual

    def build_vectorized_jacobian_and_residual(
        self,
        voltages: Array,
        context: AnalysisContext
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]], np.ndarray]:
        """Build Jacobian matrix and residual using vectorized device evaluation

        GPU-friendly version that evaluates all devices of the same type in parallel
        using JAX operations instead of Python loops.

        Requires device_groups to be populated (via build_device_groups()).

        Args:
            voltages: Current node voltage estimates (shape: [num_nodes])
            context: Analysis context with time, etc.

        Returns:
            Tuple of ((data, indices, indptr, shape), residual) where:
                data, indices, indptr: CSR format arrays
                shape: (n, n) matrix shape
                residual: Shape [num_nodes-1] numpy array
        """
        from jax_spice.devices.vsource import vsource_batch
        from jax_spice.devices.resistor import resistor_batch

        n = self.num_nodes - 1  # Exclude ground

        # Accumulate triplets (row, col, value) for COO format
        rows = []
        cols = []
        values = []
        residual = np.zeros(n, dtype=np.float64)

        # Process each device group with vectorized evaluation
        for group in self.device_groups:
            if group.n_devices == 0:
                continue

            # Get terminal voltages for all devices in this group
            # node_indices is (n_devices, n_terminals)
            V_batch = voltages[group.node_indices]  # (n_devices, n_terminals)

            if group.device_type == DeviceType.VSOURCE:
                # Vectorized voltage source evaluation
                V_target = group.params.get('v', group.params.get('dc', jnp.zeros(group.n_devices)))
                I_batch, G_batch = vsource_batch(V_batch, V_target)

                # Stamp into matrix: 2-terminal device with stamps:
                # Residual: f[p] += I, f[n] -= I
                # Jacobian: G[p,p] += G, G[p,n] -= G, G[n,p] -= G, G[n,n] += G
                node_p = np.array(group.node_indices[:, 0])  # (n_devices,)
                node_n = np.array(group.node_indices[:, 1])  # (n_devices,)
                I_np = np.array(I_batch)
                G_np = np.array(G_batch)

                # Stamp residuals (KCL current sum at each node)
                for i in range(group.n_devices):
                    np_idx, nn_idx = int(node_p[i]), int(node_n[i])
                    if np_idx != self.ground_node:
                        residual[np_idx - 1] += I_np[i]
                    if nn_idx != self.ground_node:
                        residual[nn_idx - 1] -= I_np[i]

                # Stamp Jacobian conductances
                for i in range(group.n_devices):
                    np_idx, nn_idx = int(node_p[i]), int(node_n[i])
                    g = G_np[i]

                    # G[p,p] += G
                    if np_idx != self.ground_node:
                        rows.append(np_idx - 1)
                        cols.append(np_idx - 1)
                        values.append(g)

                    # G[p,n] -= G
                    if np_idx != self.ground_node and nn_idx != self.ground_node:
                        rows.append(np_idx - 1)
                        cols.append(nn_idx - 1)
                        values.append(-g)

                    # G[n,p] -= G
                    if nn_idx != self.ground_node and np_idx != self.ground_node:
                        rows.append(nn_idx - 1)
                        cols.append(np_idx - 1)
                        values.append(-g)

                    # G[n,n] += G
                    if nn_idx != self.ground_node:
                        rows.append(nn_idx - 1)
                        cols.append(nn_idx - 1)
                        values.append(g)

            elif group.device_type == DeviceType.RESISTOR:
                # Vectorized resistor evaluation
                R_batch = group.params.get('r', group.params.get('R', jnp.ones(group.n_devices) * 1000.0))
                I_batch, G_batch = resistor_batch(V_batch, R_batch)

                node_p = np.array(group.node_indices[:, 0])
                node_n = np.array(group.node_indices[:, 1])
                I_np = np.array(I_batch)
                G_np = np.array(G_batch)

                # Same stamping pattern as voltage sources
                for i in range(group.n_devices):
                    np_idx, nn_idx = int(node_p[i]), int(node_n[i])
                    if np_idx != self.ground_node:
                        residual[np_idx - 1] += I_np[i]
                    if nn_idx != self.ground_node:
                        residual[nn_idx - 1] -= I_np[i]

                for i in range(group.n_devices):
                    np_idx, nn_idx = int(node_p[i]), int(node_n[i])
                    g = G_np[i]

                    if np_idx != self.ground_node:
                        rows.append(np_idx - 1)
                        cols.append(np_idx - 1)
                        values.append(g)
                    if np_idx != self.ground_node and nn_idx != self.ground_node:
                        rows.append(np_idx - 1)
                        cols.append(nn_idx - 1)
                        values.append(-g)
                    if nn_idx != self.ground_node and np_idx != self.ground_node:
                        rows.append(nn_idx - 1)
                        cols.append(np_idx - 1)
                        values.append(-g)
                    if nn_idx != self.ground_node:
                        rows.append(nn_idx - 1)
                        cols.append(nn_idx - 1)
                        values.append(g)

            else:
                # Fallback to scalar evaluation for unsupported device types
                # (MOSFET, Verilog-A, etc. - to be added later)
                for idx, device_name in enumerate(group.device_names):
                    # Find matching device in self.devices
                    for device in self.devices:
                        if device.name == device_name:
                            dev_voltages = {}
                            for j, (term, node_idx) in enumerate(zip(device.terminals, device.node_indices)):
                                dev_voltages[term] = float(voltages[node_idx])
                            stamps = device.eval_fn(dev_voltages, device.params, context)

                            for term, current in stamps.currents.items():
                                term_idx = device.terminals.index(term)
                                node_idx = device.node_indices[term_idx]
                                if node_idx != self.ground_node:
                                    residual[node_idx - 1] += float(current)

                            for (term_i, term_j), conductance in stamps.conductances.items():
                                idx_i = device.terminals.index(term_i)
                                idx_j = device.terminals.index(term_j)
                                node_i = device.node_indices[idx_i]
                                node_j = device.node_indices[idx_j]
                                if node_i != self.ground_node and node_j != self.ground_node:
                                    rows.append(node_i - 1)
                                    cols.append(node_j - 1)
                                    values.append(float(conductance))
                            break

        # Add GMIN to diagonal
        gmin = context.gmin if hasattr(context, 'gmin') else 1e-12
        for i in range(n):
            rows.append(i)
            cols.append(i)
            values.append(gmin)

        # Convert to CSR format
        from jax_spice.analysis.sparse import build_csr_arrays

        rows_arr = np.array(rows, dtype=np.int32)
        cols_arr = np.array(cols, dtype=np.int32)
        values_arr = np.array(values, dtype=np.float64)

        data, indices, indptr = build_csr_arrays(
            rows_arr, cols_arr, values_arr, (n, n)
        )

        return (data, indices, indptr, (n, n)), residual

    def build_device_groups(self) -> None:
        """Build vectorized device groups from the devices list

        Groups devices by type and creates VectorizedDeviceGroup instances
        with pre-computed node indices and parameter arrays.
        """
        from collections import defaultdict

        # Group devices by type
        type_to_devices: Dict[DeviceType, List[DeviceInfo]] = defaultdict(list)

        for device in self.devices:
            # Determine device type from model name or eval_fn
            model_lower = device.model_name.lower()

            if 'vsource' in model_lower or 'vdc' in model_lower:
                dtype = DeviceType.VSOURCE
            elif 'isource' in model_lower or 'idc' in model_lower:
                dtype = DeviceType.ISOURCE
            elif 'resistor' in model_lower or model_lower.startswith('r'):
                dtype = DeviceType.RESISTOR
            elif 'capacitor' in model_lower or model_lower.startswith('c'):
                dtype = DeviceType.CAPACITOR
            elif 'nmos' in model_lower or 'pmos' in model_lower or 'mosfet' in model_lower:
                dtype = DeviceType.MOSFET
            elif 'psp' in model_lower:  # PSP103 MOSFET model
                dtype = DeviceType.MOSFET
            else:
                dtype = DeviceType.UNKNOWN

            type_to_devices[dtype].append(device)

        # Build VectorizedDeviceGroup for each type
        self.device_groups = []

        for dtype, devices in type_to_devices.items():
            if not devices:
                continue

            n_devices = len(devices)
            device_names = [d.name for d in devices]

            # Build node_indices array
            # All devices of same type should have same number of terminals
            n_terminals = len(devices[0].node_indices)
            node_indices = np.array(
                [d.node_indices for d in devices],
                dtype=np.int32
            )

            # Build parameter arrays
            # Collect all unique parameter names
            all_params: Dict[str, List[float]] = defaultdict(list)
            for device in devices:
                for key, val in device.params.items():
                    if isinstance(val, (int, float)):
                        all_params[key].append(float(val))

            # Convert to JAX arrays
            params = {}
            for key, vals in all_params.items():
                if len(vals) == n_devices:
                    params[key] = jnp.array(vals)

            group = VectorizedDeviceGroup(
                device_type=dtype,
                n_devices=n_devices,
                device_names=device_names,
                node_indices=jnp.array(node_indices),
                params=params
            )
            self.device_groups.append(group)
