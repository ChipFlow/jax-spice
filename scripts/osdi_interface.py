"""OSDI (Open Simulator Device Interface) Python interface.

This module provides Python bindings to OSDI 0.4 compiled Verilog-A models
(.osdi shared libraries) via ctypes.

Reference: VACASK/include/osdi.h for structure definitions.

Usage:
    from osdi_interface import OsdiModel

    # Load a compiled OSDI model
    model = OsdiModel('/path/to/capacitor.osdi')

    # Initialize model and instance
    model.setup_model(temperature=300.15)
    model.setup_instance(temperature=300.15)

    # Set parameters
    model.set_param('c', 1e-9)

    # Evaluate at a given operating point
    voltages = {0: 1.0, 1: 0.0}  # Node voltages
    result = model.eval(voltages)

    print(f"Residuals: {result['residuals']}")
    print(f"Jacobian (resist): {result['jacobian_resist']}")
"""

import ctypes
from ctypes import (
    POINTER, Structure, Union, c_char_p, c_double, c_uint32, c_int32,
    c_void_p, c_bool, c_size_t, CFUNCTYPE, cast, byref
)
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


# =============================================================================
# OSDI Constants (from osdi.h)
# =============================================================================

OSDI_VERSION_MAJOR_CURR = 0
OSDI_VERSION_MINOR_CURR = 4

# Parameter type masks
PARA_TY_MASK = 3
PARA_TY_REAL = 0
PARA_TY_INT = 1
PARA_TY_STR = 2
PARA_KIND_MASK = 3 << 30
PARA_KIND_MODEL = 0 << 30
PARA_KIND_INST = 1 << 30
PARA_KIND_OPVAR = 2 << 30

# Access flags
ACCESS_FLAG_READ = 0
ACCESS_FLAG_SET = 1
ACCESS_FLAG_INSTANCE = 4

# Jacobian entry flags
JACOBIAN_ENTRY_RESIST_CONST = 1
JACOBIAN_ENTRY_REACT_CONST = 2
JACOBIAN_ENTRY_RESIST = 4
JACOBIAN_ENTRY_REACT = 8

# Calculation flags
CALC_RESIST_RESIDUAL = 1
CALC_REACT_RESIDUAL = 2
CALC_RESIST_JACOBIAN = 4
CALC_REACT_JACOBIAN = 8
CALC_NOISE = 16
CALC_OP = 32
CALC_RESIST_LIM_RHS = 64
CALC_REACT_LIM_RHS = 128
ENABLE_LIM = 256
INIT_LIM = 512
ANALYSIS_NOISE = 1024
ANALYSIS_DC = 2048
ANALYSIS_AC = 4096
ANALYSIS_TRAN = 8192
ANALYSIS_IC = 16384
ANALYSIS_STATIC = 32768
ANALYSIS_NODESET = 65536

# Eval return flags
EVAL_RET_FLAG_LIM = 1
EVAL_RET_FLAG_FATAL = 2
EVAL_RET_FLAG_FINISH = 4
EVAL_RET_FLAG_STOP = 8


# =============================================================================
# OSDI Structures (from osdi.h)
# =============================================================================

class OsdiSimParas(Structure):
    """Simulation parameters passed to model functions."""
    _fields_ = [
        ('names', POINTER(c_char_p)),      # Parameter names
        ('vals', POINTER(c_double)),       # Parameter values
        ('names_str', POINTER(c_char_p)),  # String parameter names
        ('vals_str', POINTER(c_char_p)),   # String parameter values
    ]


class OsdiSimInfo(Structure):
    """Simulation info passed to eval function."""
    _fields_ = [
        ('paras', OsdiSimParas),
        ('abstime', c_double),
        ('prev_solve', POINTER(c_double)),
        ('prev_state', POINTER(c_double)),
        ('next_state', POINTER(c_double)),
        ('flags', c_uint32),
    ]


class OsdiInitErrorPayload(Union):
    """Init error payload."""
    _fields_ = [
        ('parameter_id', c_uint32),
    ]


class OsdiInitError(Structure):
    """Init error info."""
    _fields_ = [
        ('code', c_uint32),
        ('payload', OsdiInitErrorPayload),
    ]


class OsdiInitInfo(Structure):
    """Init function result info."""
    _fields_ = [
        ('flags', c_uint32),
        ('num_errors', c_uint32),
        ('errors', POINTER(OsdiInitError)),
    ]


class OsdiNodePair(Structure):
    """A pair of node indices."""
    _fields_ = [
        ('node_1', c_uint32),
        ('node_2', c_uint32),
    ]


class OsdiJacobianEntry(Structure):
    """Jacobian matrix entry metadata."""
    _fields_ = [
        ('nodes', OsdiNodePair),
        ('react_ptr_off', c_uint32),
        ('flags', c_uint32),
    ]


class OsdiNode(Structure):
    """Node descriptor."""
    _fields_ = [
        ('name', c_char_p),
        ('units', c_char_p),
        ('residual_units', c_char_p),
        ('resist_residual_off', c_uint32),
        ('react_residual_off', c_uint32),
        ('resist_limit_rhs_off', c_uint32),
        ('react_limit_rhs_off', c_uint32),
        ('is_flow', c_bool),
    ]


class OsdiParamOpvar(Structure):
    """Parameter or output variable descriptor."""
    _fields_ = [
        ('name', POINTER(c_char_p)),  # Array of names (aliases)
        ('num_alias', c_uint32),
        ('description', c_char_p),
        ('units', c_char_p),
        ('flags', c_uint32),
        ('len', c_uint32),  # Array length (1 for scalar)
    ]


class OsdiNoiseSource(Structure):
    """Noise source descriptor."""
    _fields_ = [
        ('name', c_char_p),
        ('nodes', OsdiNodePair),
    ]


class OsdiNatureRef(Structure):
    """Nature reference."""
    _fields_ = [
        ('ref_type', c_uint32),
        ('index', c_uint32),
    ]


# Forward declaration for function pointers
class OsdiDescriptor(Structure):
    """OSDI model descriptor - the main interface to the compiled model."""
    pass


# Define function pointer types
AccessFunc = CFUNCTYPE(c_void_p, c_void_p, c_void_p, c_uint32, c_uint32)
SetupModelFunc = CFUNCTYPE(None, c_void_p, c_void_p, POINTER(OsdiSimParas), POINTER(OsdiInitInfo))
SetupInstanceFunc = CFUNCTYPE(None, c_void_p, c_void_p, c_void_p, c_double, c_uint32, POINTER(OsdiSimParas), POINTER(OsdiInitInfo))
EvalFunc = CFUNCTYPE(c_uint32, c_void_p, c_void_p, c_void_p, POINTER(OsdiSimInfo))
LoadNoiseFunc = CFUNCTYPE(None, c_void_p, c_void_p, c_double, POINTER(c_double))
LoadResidualFunc = CFUNCTYPE(None, c_void_p, c_void_p, POINTER(c_double))
LoadSpiceRhsDcFunc = CFUNCTYPE(None, c_void_p, c_void_p, POINTER(c_double), POINTER(c_double))
LoadSpiceRhsTranFunc = CFUNCTYPE(None, c_void_p, c_void_p, POINTER(c_double), POINTER(c_double), c_double)
LoadJacobianFunc = CFUNCTYPE(None, c_void_p, c_void_p)
LoadJacobianAlphaFunc = CFUNCTYPE(None, c_void_p, c_void_p, c_double)
GivenFlagFunc = CFUNCTYPE(c_uint32, c_void_p, c_uint32)
WriteJacobianArrayFunc = CFUNCTYPE(None, c_void_p, c_void_p, POINTER(c_double))
LoadJacobianOffsetFunc = CFUNCTYPE(None, c_void_p, c_void_p, c_size_t)

# Complete descriptor fields
OsdiDescriptor._fields_ = [
    ('name', c_char_p),

    ('num_nodes', c_uint32),
    ('num_terminals', c_uint32),
    ('nodes', POINTER(OsdiNode)),

    ('num_jacobian_entries', c_uint32),
    ('jacobian_entries', POINTER(OsdiJacobianEntry)),

    ('num_collapsible', c_uint32),
    ('collapsible', POINTER(OsdiNodePair)),
    ('collapsed_offset', c_uint32),

    ('noise_sources', POINTER(OsdiNoiseSource)),
    ('num_noise_src', c_uint32),

    ('num_params', c_uint32),
    ('num_instance_params', c_uint32),
    ('num_opvars', c_uint32),
    ('param_opvar', POINTER(OsdiParamOpvar)),

    ('node_mapping_offset', c_uint32),
    ('jacobian_ptr_resist_offset', c_uint32),

    ('num_states', c_uint32),
    ('state_idx_off', c_uint32),

    ('bound_step_offset', c_uint32),

    ('instance_size', c_uint32),
    ('model_size', c_uint32),

    ('access', AccessFunc),
    ('setup_model', SetupModelFunc),
    ('setup_instance', SetupInstanceFunc),
    ('eval', EvalFunc),
    ('load_noise', LoadNoiseFunc),
    ('load_residual_resist', LoadResidualFunc),
    ('load_residual_react', LoadResidualFunc),
    ('load_limit_rhs_resist', LoadResidualFunc),
    ('load_limit_rhs_react', LoadResidualFunc),
    ('load_spice_rhs_dc', LoadSpiceRhsDcFunc),
    ('load_spice_rhs_tran', LoadSpiceRhsTranFunc),
    ('load_jacobian_resist', LoadJacobianFunc),
    ('load_jacobian_react', LoadJacobianAlphaFunc),
    ('load_jacobian_tran', LoadJacobianAlphaFunc),
    ('given_flag_model', GivenFlagFunc),
    ('given_flag_instance', GivenFlagFunc),
    ('num_resistive_jacobian_entries', c_uint32),
    ('num_reactive_jacobian_entries', c_uint32),
    ('write_jacobian_array_resist', WriteJacobianArrayFunc),
    ('write_jacobian_array_react', WriteJacobianArrayFunc),
    ('num_inputs', c_uint32),
    ('inputs', POINTER(OsdiNodePair)),
    ('load_jacobian_with_offset_resist', LoadJacobianOffsetFunc),
    ('load_jacobian_with_offset_react', LoadJacobianOffsetFunc),
    ('unknown_nature', POINTER(OsdiNatureRef)),
    ('residual_nature', POINTER(OsdiNatureRef)),
]


# =============================================================================
# OSDI Model Wrapper
# =============================================================================

class OsdiModel:
    """Python wrapper for an OSDI compiled model.

    This class loads an OSDI shared library and provides methods to:
    - Query model metadata (nodes, parameters, Jacobian structure)
    - Initialize model and instance data structures
    - Set/get parameters
    - Evaluate the model at given operating points
    - Extract residuals and Jacobian matrices
    """

    def __init__(self, osdi_path: str):
        """Load an OSDI shared library.

        Args:
            osdi_path: Path to the .osdi shared library file.
        """
        self.path = Path(osdi_path)
        if not self.path.exists():
            raise FileNotFoundError(f"OSDI library not found: {osdi_path}")

        # Load the shared library
        self.lib = ctypes.CDLL(str(self.path))

        # Get global symbols using in_dll() - these are data, not functions
        # OSDI_NUM_DESCRIPTORS is a uint32 global variable
        num_descriptors = c_uint32.in_dll(self.lib, "OSDI_NUM_DESCRIPTORS").value

        if num_descriptors < 1:
            raise RuntimeError(f"No descriptors found in {osdi_path}")

        # OSDI_DESCRIPTORS is an INLINE array of OsdiDescriptor structs (not pointers)
        # The descriptors are stored directly at the symbol address
        descriptors_array_type = OsdiDescriptor * num_descriptors
        descriptors_array = descriptors_array_type.in_dll(self.lib, "OSDI_DESCRIPTORS")

        # Use the first descriptor (direct access, not pointer dereference)
        self.descriptor = descriptors_array[0]
        self.name = self.descriptor.name.decode('utf-8') if self.descriptor.name else "unknown"

        # Allocate model and instance data structures
        self._model_data = (ctypes.c_byte * self.descriptor.model_size)()
        self._instance_data = (ctypes.c_byte * self.descriptor.instance_size)()

        # Initialize state arrays
        self.num_states = self.descriptor.num_states
        self._prev_state = (c_double * max(1, self.num_states))()
        self._next_state = (c_double * max(1, self.num_states))()

        # Allocate solution vector (for node voltages)
        self.num_nodes = self.descriptor.num_nodes
        self.num_terminals = self.descriptor.num_terminals
        self._prev_solve = (c_double * max(1, self.num_nodes))()

        # Cache parameter info
        self._param_info = self._extract_param_info()

        # Cache node info
        self._node_info = self._extract_node_info()

        # Cache Jacobian structure
        self._jacobian_info = self._extract_jacobian_info()

        # Allocate Jacobian arrays for write_jacobian_array functions
        self._jacobian_resist_array = (c_double * max(1, self.descriptor.num_resistive_jacobian_entries))()
        self._jacobian_react_array = (c_double * max(1, self.descriptor.num_reactive_jacobian_entries))()

        # Track initialization status
        self._model_initialized = False
        self._instance_initialized = False

    def _extract_param_info(self) -> Dict[str, Dict]:
        """Extract parameter metadata from descriptor."""
        params = {}
        total = self.descriptor.num_params + self.descriptor.num_instance_params + self.descriptor.num_opvars

        for i in range(total):
            pov = self.descriptor.param_opvar[i]

            # Get primary name
            if pov.name and pov.name[0]:
                name = pov.name[0].decode('utf-8')
            else:
                name = f"param_{i}"

            # Get aliases
            aliases = []
            for j in range(pov.num_alias):
                if pov.name[j]:
                    aliases.append(pov.name[j].decode('utf-8'))

            # Determine kind
            flags = pov.flags
            kind_bits = flags & PARA_KIND_MASK
            if kind_bits == PARA_KIND_MODEL:
                kind = 'model'
            elif kind_bits == PARA_KIND_INST:
                kind = 'instance'
            else:
                kind = 'opvar'

            # Determine type
            type_bits = flags & PARA_TY_MASK
            if type_bits == PARA_TY_REAL:
                ptype = 'real'
            elif type_bits == PARA_TY_INT:
                ptype = 'int'
            else:
                ptype = 'str'

            params[name] = {
                'id': i,
                'name': name,
                'aliases': aliases,
                'kind': kind,
                'type': ptype,
                'len': pov.len,
                'description': pov.description.decode('utf-8') if pov.description else '',
                'units': pov.units.decode('utf-8') if pov.units else '',
            }

        return params

    def _extract_node_info(self) -> List[Dict]:
        """Extract node metadata from descriptor."""
        nodes = []
        for i in range(self.num_nodes):
            node = self.descriptor.nodes[i]
            nodes.append({
                'id': i,
                'name': node.name.decode('utf-8') if node.name else f"node_{i}",
                'units': node.units.decode('utf-8') if node.units else '',
                'residual_units': node.residual_units.decode('utf-8') if node.residual_units else '',
                'resist_residual_off': node.resist_residual_off,
                'react_residual_off': node.react_residual_off,
                'is_flow': bool(node.is_flow),
                'is_terminal': i < self.num_terminals,
            })
        return nodes

    def _extract_jacobian_info(self) -> List[Dict]:
        """Extract Jacobian structure from descriptor."""
        entries = []
        for i in range(self.descriptor.num_jacobian_entries):
            entry = self.descriptor.jacobian_entries[i]
            entries.append({
                'id': i,
                'row': entry.nodes.node_1,
                'col': entry.nodes.node_2,
                'react_ptr_off': entry.react_ptr_off,
                'has_resist': bool(entry.flags & JACOBIAN_ENTRY_RESIST),
                'has_react': bool(entry.flags & JACOBIAN_ENTRY_REACT),
                'resist_const': bool(entry.flags & JACOBIAN_ENTRY_RESIST_CONST),
                'react_const': bool(entry.flags & JACOBIAN_ENTRY_REACT_CONST),
            })
        return entries

    @property
    def model_ptr(self) -> c_void_p:
        """Get pointer to model data."""
        return cast(self._model_data, c_void_p)

    @property
    def instance_ptr(self) -> c_void_p:
        """Get pointer to instance data."""
        return cast(self._instance_data, c_void_p)

    def info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': self.name,
            'path': str(self.path),
            'num_nodes': self.num_nodes,
            'num_terminals': self.num_terminals,
            'num_params': len([p for p in self._param_info.values() if p['kind'] in ('model', 'instance')]),
            'num_opvars': len([p for p in self._param_info.values() if p['kind'] == 'opvar']),
            'num_states': self.num_states,
            'num_jacobian_entries': self.descriptor.num_jacobian_entries,
            'num_resistive_jacobian_entries': self.descriptor.num_resistive_jacobian_entries,
            'num_reactive_jacobian_entries': self.descriptor.num_reactive_jacobian_entries,
            'model_size': self.descriptor.model_size,
            'instance_size': self.descriptor.instance_size,
        }

    def get_param_names(self, kind: Optional[str] = None) -> List[str]:
        """Get parameter names, optionally filtered by kind."""
        names = []
        for name, info in self._param_info.items():
            if kind is None or info['kind'] == kind:
                names.append(name)
        return names

    def get_param_info(self, name: str) -> Optional[Dict]:
        """Get info for a specific parameter."""
        return self._param_info.get(name)

    def get_node_names(self) -> List[str]:
        """Get node names."""
        return [n['name'] for n in self._node_info]

    def get_terminal_names(self) -> List[str]:
        """Get terminal (external node) names."""
        return [n['name'] for n in self._node_info if n['is_terminal']]

    def _create_sim_paras(self, **kwargs) -> OsdiSimParas:
        """Create simulation parameters structure.

        Common parameters:
            gmin: Minimum conductance (default: 1e-12)
            tnom: Nominal temperature in K (default: 300.15)
            scale: Scale factor (default: 1.0)
            iteration: Iteration count (default: 0)
        """
        # Default simulation parameters
        defaults = {
            'gmin': 1e-12,
            'tnom': 300.15,
            'scale': 1.0,
            'shrink': 0.0,
            'iteration': 0.0,
        }
        defaults.update(kwargs)

        # Create arrays
        names = list(defaults.keys())
        vals = list(defaults.values())
        n = len(names)

        name_array = (c_char_p * n)(*[k.encode('utf-8') for k in names])
        val_array = (c_double * n)(*vals)

        paras = OsdiSimParas()
        paras.names = cast(name_array, POINTER(c_char_p))
        paras.vals = cast(val_array, POINTER(c_double))
        paras.names_str = None
        paras.vals_str = None

        # Store references to prevent garbage collection
        self._sim_para_names = name_array
        self._sim_para_vals = val_array

        return paras

    def setup_model(self, temperature: float = 300.15, **sim_params) -> Dict:
        """Initialize the model data structure.

        Args:
            temperature: Temperature in Kelvin (default: 300.15K = 27C)
            **sim_params: Additional simulation parameters (gmin, tnom, etc.)

        Returns:
            Dict with initialization result info.
        """
        sim_params.setdefault('tnom', temperature)
        paras = self._create_sim_paras(**sim_params)

        init_info = OsdiInitInfo()
        init_info.flags = 0
        init_info.num_errors = 0
        init_info.errors = None

        # Create handle (null for now - VACASK uses this for logging)
        handle = c_void_p(0)

        # Call setup_model
        self.descriptor.setup_model(handle, self.model_ptr, byref(paras), byref(init_info))

        self._model_initialized = True

        return {
            'flags': init_info.flags,
            'num_errors': init_info.num_errors,
        }

    def setup_instance(self, temperature: float = 300.15, **sim_params) -> Dict:
        """Initialize the instance data structure.

        Must be called after setup_model().

        Args:
            temperature: Temperature in Kelvin (default: 300.15K = 27C)
            **sim_params: Additional simulation parameters.

        Returns:
            Dict with initialization result info.
        """
        if not self._model_initialized:
            raise RuntimeError("setup_model() must be called before setup_instance()")

        sim_params.setdefault('tnom', temperature)
        paras = self._create_sim_paras(**sim_params)

        init_info = OsdiInitInfo()
        init_info.flags = 0
        init_info.num_errors = 0
        init_info.errors = None

        handle = c_void_p(0)

        # Call setup_instance
        self.descriptor.setup_instance(
            handle,
            self.instance_ptr,
            self.model_ptr,
            c_double(temperature),
            c_uint32(self.num_terminals),
            byref(paras),
            byref(init_info)
        )

        self._instance_initialized = True

        # Set up node mapping (identity mapping for testing)
        node_mapping_off = self.descriptor.node_mapping_offset
        if node_mapping_off > 0:
            node_mapping = cast(
                ctypes.addressof(self._instance_data) + node_mapping_off,
                POINTER(c_uint32)
            )
            for i in range(self.num_nodes):
                node_mapping[i] = i

        return {
            'flags': init_info.flags,
            'num_errors': init_info.num_errors,
        }

    def set_param(self, name: str, value: float) -> bool:
        """Set a parameter value.

        Args:
            name: Parameter name.
            value: Parameter value.

        Returns:
            True if parameter was set successfully.
        """
        info = self._param_info.get(name)
        if info is None:
            # Try case-insensitive match
            for pname, pinfo in self._param_info.items():
                if pname.lower() == name.lower():
                    info = pinfo
                    break

        if info is None:
            return False

        # Determine flags
        flags = ACCESS_FLAG_SET
        if info['kind'] == 'instance':
            flags |= ACCESS_FLAG_INSTANCE

        # Get value pointer via access function
        ptr = self.descriptor.access(
            self.instance_ptr,
            self.model_ptr,
            c_uint32(info['id']),
            c_uint32(flags)
        )

        if ptr:
            if info['type'] == 'real':
                cast(ptr, POINTER(c_double))[0] = value
            elif info['type'] == 'int':
                cast(ptr, POINTER(c_int32))[0] = int(value)
            return True

        return False

    def get_param(self, name: str) -> Optional[float]:
        """Get a parameter value.

        Args:
            name: Parameter name.

        Returns:
            Parameter value or None if not found.
        """
        info = self._param_info.get(name)
        if info is None:
            # Try case-insensitive match
            for pname, pinfo in self._param_info.items():
                if pname.lower() == name.lower():
                    info = pinfo
                    break

        if info is None:
            return None

        # Determine flags
        flags = ACCESS_FLAG_READ
        if info['kind'] == 'instance':
            flags |= ACCESS_FLAG_INSTANCE

        # Get value pointer via access function
        ptr = self.descriptor.access(
            self.instance_ptr,
            self.model_ptr,
            c_uint32(info['id']),
            c_uint32(flags)
        )

        if ptr:
            if info['type'] == 'real':
                return cast(ptr, POINTER(c_double))[0]
            elif info['type'] == 'int':
                return float(cast(ptr, POINTER(c_int32))[0])

        return None

    def eval(
        self,
        voltages: Dict[int, float],
        time: float = 0.0,
        flags: int = CALC_RESIST_RESIDUAL | CALC_RESIST_JACOBIAN | ANALYSIS_DC,
        **sim_params
    ) -> Dict[str, Any]:
        """Evaluate the model at given node voltages.

        Args:
            voltages: Dict mapping node index to voltage value.
            time: Simulation time (for transient analysis).
            flags: Calculation flags (what to compute).
            **sim_params: Additional simulation parameters.

        Returns:
            Dict containing:
                'eval_flags': Return flags from eval function
                'residuals': Dict mapping node name to residual value
                'jacobian_resist': 2D array of resistive Jacobian values
                'jacobian_react': 2D array of reactive Jacobian values (if computed)
        """
        if not self._instance_initialized:
            raise RuntimeError("setup_instance() must be called before eval()")

        # Set up solution vector
        for node_idx, voltage in voltages.items():
            if 0 <= node_idx < self.num_nodes:
                self._prev_solve[node_idx] = voltage

        # Create sim info
        paras = self._create_sim_paras(**sim_params)

        sim_info = OsdiSimInfo()
        sim_info.paras = paras
        sim_info.abstime = time
        sim_info.prev_solve = cast(self._prev_solve, POINTER(c_double))
        sim_info.prev_state = cast(self._prev_state, POINTER(c_double))
        sim_info.next_state = cast(self._next_state, POINTER(c_double))
        sim_info.flags = flags

        handle = c_void_p(0)

        # Call eval
        eval_flags = self.descriptor.eval(
            handle,
            self.instance_ptr,
            self.model_ptr,
            byref(sim_info)
        )

        result = {
            'eval_flags': eval_flags,
            'limiting_applied': bool(eval_flags & EVAL_RET_FLAG_LIM),
            'fatal_error': bool(eval_flags & EVAL_RET_FLAG_FATAL),
        }

        # Extract residuals
        result['residuals'] = self._extract_residuals(flags)

        # Extract Jacobian
        if flags & CALC_RESIST_JACOBIAN:
            result['jacobian_resist'] = self._extract_jacobian_resist()

        if flags & CALC_REACT_JACOBIAN:
            result['jacobian_react'] = self._extract_jacobian_react()

        return result

    def _extract_residuals(self, flags: int) -> Dict[str, float]:
        """Extract residual values from instance data."""
        residuals = {}

        for node_info in self._node_info:
            name = node_info['name']

            # Get resistive residual
            if flags & CALC_RESIST_RESIDUAL:
                off = node_info['resist_residual_off']
                if off > 0:
                    ptr = cast(
                        ctypes.addressof(self._instance_data) + off,
                        POINTER(c_double)
                    )
                    residuals[f"{name}_resist"] = ptr[0]

            # Get reactive residual
            if flags & CALC_REACT_RESIDUAL:
                off = node_info['react_residual_off']
                if off > 0:
                    ptr = cast(
                        ctypes.addressof(self._instance_data) + off,
                        POINTER(c_double)
                    )
                    residuals[f"{name}_react"] = ptr[0]

        return residuals

    def _extract_jacobian_resist(self) -> np.ndarray:
        """Extract resistive Jacobian as numpy array."""
        # Use write_jacobian_array_resist to get all entries
        self.descriptor.write_jacobian_array_resist(
            self.instance_ptr,
            self.model_ptr,
            cast(self._jacobian_resist_array, POINTER(c_double))
        )

        # Convert to numpy
        n = self.descriptor.num_resistive_jacobian_entries
        flat = np.array([self._jacobian_resist_array[i] for i in range(n)])

        # Also build a sparse representation
        jacobian = np.zeros((self.num_nodes, self.num_nodes))

        resist_idx = 0
        for entry in self._jacobian_info:
            if entry['has_resist']:
                row, col = entry['row'], entry['col']
                if resist_idx < len(flat):
                    jacobian[row, col] = flat[resist_idx]
                    resist_idx += 1

        return jacobian

    def _extract_jacobian_react(self) -> np.ndarray:
        """Extract reactive Jacobian as numpy array."""
        # Use write_jacobian_array_react to get all entries
        self.descriptor.write_jacobian_array_react(
            self.instance_ptr,
            self.model_ptr,
            cast(self._jacobian_react_array, POINTER(c_double))
        )

        # Convert to numpy
        n = self.descriptor.num_reactive_jacobian_entries
        flat = np.array([self._jacobian_react_array[i] for i in range(n)])

        # Build sparse representation
        jacobian = np.zeros((self.num_nodes, self.num_nodes))

        react_idx = 0
        for entry in self._jacobian_info:
            if entry['has_react']:
                row, col = entry['row'], entry['col']
                if react_idx < len(flat):
                    jacobian[row, col] = flat[react_idx]
                    react_idx += 1

        return jacobian

    def load_residual_resist(self) -> np.ndarray:
        """Load resistive residual into an array and return it."""
        rhs = (c_double * self.num_nodes)()
        self.descriptor.load_residual_resist(
            self.instance_ptr,
            self.model_ptr,
            cast(rhs, POINTER(c_double))
        )
        return np.array([rhs[i] for i in range(self.num_nodes)])

    def load_residual_react(self) -> np.ndarray:
        """Load reactive residual into an array and return it."""
        rhs = (c_double * self.num_nodes)()
        self.descriptor.load_residual_react(
            self.instance_ptr,
            self.model_ptr,
            cast(rhs, POINTER(c_double))
        )
        return np.array([rhs[i] for i in range(self.num_nodes)])

    def get_state(self) -> np.ndarray:
        """Get internal state values."""
        return np.array([self._prev_state[i] for i in range(self.num_states)])

    def set_state(self, state: np.ndarray):
        """Set internal state values."""
        for i in range(min(len(state), self.num_states)):
            self._prev_state[i] = state[i]

    def __repr__(self) -> str:
        return f"OsdiModel('{self.name}', nodes={self.num_nodes}, terminals={self.num_terminals})"


# =============================================================================
# Convenience functions
# =============================================================================

def load_model(osdi_path: str) -> OsdiModel:
    """Load an OSDI model and initialize it with default settings.

    Args:
        osdi_path: Path to the .osdi file.

    Returns:
        Initialized OsdiModel.
    """
    model = OsdiModel(osdi_path)
    model.setup_model()
    model.setup_instance()
    return model


def list_models(osdi_path: str) -> List[str]:
    """List all models in an OSDI library.

    Args:
        osdi_path: Path to the .osdi file.

    Returns:
        List of model names.
    """
    lib = ctypes.CDLL(osdi_path)
    lib.OSDI_DESCRIPTORS.restype = POINTER(POINTER(OsdiDescriptor))
    lib.OSDI_NUM_DESCRIPTORS.restype = c_uint32

    n = lib.OSDI_NUM_DESCRIPTORS()
    descriptors = lib.OSDI_DESCRIPTORS()

    names = []
    for i in range(n):
        name = descriptors[i].contents.name
        if name:
            names.append(name.decode('utf-8'))

    return names


if __name__ == '__main__':
    # Quick test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python osdi_interface.py <path/to/model.osdi>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading OSDI model: {path}")

    model = OsdiModel(path)
    print(f"\nModel info:")
    for k, v in model.info().items():
        print(f"  {k}: {v}")

    print(f"\nTerminals: {model.get_terminal_names()}")
    print(f"All nodes: {model.get_node_names()}")

    print(f"\nModel parameters: {model.get_param_names('model')[:10]}...")
    print(f"Instance parameters: {model.get_param_names('instance')[:10]}...")
