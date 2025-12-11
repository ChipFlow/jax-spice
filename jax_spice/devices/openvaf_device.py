"""OpenVAF-based device evaluation with analytical Jacobians

This module provides device evaluation functions compiled from Verilog-A
models using openvaf_jax. The key advantage is that these functions return
BOTH residuals AND analytical Jacobians in a single call, avoiding the
autodiff issues that cause GPU solver convergence problems.

Usage:
    from jax_spice.devices.openvaf_device import compile_va_model, VADevice

    # Compile a model once
    device = VADevice.from_va_file("path/to/model.va")

    # Evaluate at given voltages
    residuals, jacobian = device.evaluate(voltages, params)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array

# Add openvaf-py to path
_openvaf_path = Path(__file__).parent.parent.parent / "openvaf-py"
if str(_openvaf_path) not in sys.path:
    sys.path.insert(0, str(_openvaf_path))

try:
    import openvaf_py
    import openvaf_jax
    HAS_OPENVAF = True
except ImportError:
    HAS_OPENVAF = False
    openvaf_py = None
    openvaf_jax = None


@dataclass
class VADevice:
    """Wrapper for an OpenVAF-compiled Verilog-A device model.

    Provides evaluation with analytical Jacobians suitable for Newton-Raphson.
    """
    name: str
    module: Any  # openvaf_py.VaModule
    eval_fn: Callable
    param_names: List[str]
    param_kinds: List[str]
    nodes: List[str]

    @classmethod
    def from_va_file(cls, va_path: str, allow_analog_in_cond: bool = False) -> 'VADevice':
        """Compile a Verilog-A file and create a device wrapper.

        Args:
            va_path: Path to .va file
            allow_analog_in_cond: Allow analog statements in conditionals

        Returns:
            VADevice instance ready for evaluation
        """
        if not HAS_OPENVAF:
            raise ImportError("openvaf_py and openvaf_jax required. "
                            "Ensure openvaf-py is built and in path.")

        modules = openvaf_py.compile_va(str(va_path), allow_analog_in_cond)
        if not modules:
            raise ValueError(f"No modules found in {va_path}")

        module = modules[0]
        translator = openvaf_jax.OpenVAFToJAX(module)
        eval_fn = translator.translate()

        return cls(
            name=module.name,
            module=module,
            eval_fn=eval_fn,
            param_names=list(module.param_names),
            param_kinds=list(module.param_kinds),
            nodes=list(module.nodes),
        )

    def build_inputs(self, voltages: Dict[str, float], params: Dict[str, Any],
                     temperature: float = 300.0) -> List[float]:
        """Build input array for evaluation.

        Args:
            voltages: Terminal voltages by name (e.g., {'V(A,B)': 1.0})
            params: Device parameters (e.g., {'r': 1000.0})
            temperature: Device temperature in Kelvin

        Returns:
            Input array matching param_names order
        """
        inputs = []
        for name, kind in zip(self.param_names, self.param_kinds):
            if kind == 'voltage':
                # Try exact match first, then pattern matching
                if name in voltages:
                    inputs.append(float(voltages[name]))
                else:
                    inputs.append(0.0)
            elif kind == 'temperature' or 'temperature' in name.lower():
                inputs.append(temperature)
            elif kind in ('param', 'sysfun'):
                # Look up in params dict (sysfun includes mfactor)
                param_lower = name.lower()
                # Default mfactor to 1.0
                default = 1.0 if 'mfactor' in param_lower else 1.0
                value = params.get(name, params.get(param_lower, default))
                inputs.append(float(value))
            elif kind == 'hidden_state':
                inputs.append(0.0)
            else:
                # Unknown kind - check if it's in params anyway
                if name in params:
                    inputs.append(float(params[name]))
                elif name.lower() in params:
                    inputs.append(float(params[name.lower()]))
                else:
                    inputs.append(0.0)
        return inputs

    def evaluate(self, voltages: Dict[str, float], params: Dict[str, Any],
                 temperature: float = 300.0) -> Tuple[Dict, Dict]:
        """Evaluate device and return residuals and Jacobian.

        Args:
            voltages: Terminal voltages
            params: Device parameters
            temperature: Temperature in Kelvin

        Returns:
            Tuple of (residuals, jacobian) where:
                residuals: Dict[node, {'resist': float, 'react': float}]
                jacobian: Dict[(row, col), {'resist': float, 'react': float}]
        """
        inputs = self.build_inputs(voltages, params, temperature)
        return self.eval_fn(inputs)


# Cache for compiled models
_model_cache: Dict[str, VADevice] = {}


def get_vacask_resistor() -> VADevice:
    """Get compiled VACASK resistor model."""
    key = 'vacask_resistor'
    if key not in _model_cache:
        va_path = Path(__file__).parent.parent.parent / "vendor" / "VACASK" / "devices" / "resistor.va"
        _model_cache[key] = VADevice.from_va_file(str(va_path))
    return _model_cache[key]


def get_vacask_diode() -> VADevice:
    """Get compiled VACASK diode model."""
    key = 'vacask_diode'
    if key not in _model_cache:
        va_path = Path(__file__).parent.parent.parent / "vendor" / "VACASK" / "devices" / "diode.va"
        _model_cache[key] = VADevice.from_va_file(str(va_path))
    return _model_cache[key]


def get_vacask_capacitor() -> VADevice:
    """Get compiled VACASK capacitor model."""
    key = 'vacask_capacitor'
    if key not in _model_cache:
        va_path = Path(__file__).parent.parent.parent / "vendor" / "VACASK" / "devices" / "capacitor.va"
        _model_cache[key] = VADevice.from_va_file(str(va_path))
    return _model_cache[key]


def get_mosfet_level1() -> VADevice:
    """Get compiled level-1 MOSFET model."""
    key = 'mosfet_level1'
    if key not in _model_cache:
        va_path = Path(__file__).parent.parent.parent / "tests" / "models" / "mosfet_level1.va"
        _model_cache[key] = VADevice.from_va_file(str(va_path))
    return _model_cache[key]


def stamp_device_into_system(
    residual: Array,
    jacobian_data: List[float],
    jacobian_rows: List[int],
    jacobian_cols: List[int],
    device: VADevice,
    node_map: Dict[str, int],
    voltages: Dict[str, float],
    params: Dict[str, Any],
    temperature: float = 300.0,
) -> Tuple[Array, List[float], List[int], List[int]]:
    """Stamp device residuals and Jacobian into circuit system.

    Args:
        residual: Circuit residual vector to update
        jacobian_data/rows/cols: COO format Jacobian to update
        device: VADevice to evaluate
        node_map: Maps device terminal names to circuit node indices
        voltages: Terminal voltages
        params: Device parameters
        temperature: Temperature

    Returns:
        Updated (residual, jacobian_data, jacobian_rows, jacobian_cols)
    """
    dev_residuals, dev_jacobian = device.evaluate(voltages, params, temperature)

    # Stamp residuals
    for node_name, res in dev_residuals.items():
        if node_name in node_map:
            node_idx = node_map[node_name]
            if node_idx > 0:  # Skip ground
                residual = residual.at[node_idx].add(float(res['resist']))

    # Stamp Jacobian
    for (row_name, col_name), jac in dev_jacobian.items():
        if row_name in node_map and col_name in node_map:
            row_idx = node_map[row_name]
            col_idx = node_map[col_name]
            if row_idx > 0 and col_idx > 0:  # Skip ground
                jacobian_data.append(float(jac['resist']))
                jacobian_rows.append(row_idx - 1)  # 0-indexed for reduced system
                jacobian_cols.append(col_idx - 1)

    return residual, jacobian_data, jacobian_rows, jacobian_cols
