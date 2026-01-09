"""OpenVAF to JAX translator

Translates OpenVAF MIR to JAX-compatible functions for JIT compilation.
Uses the MIR interpreter for init (one-time setup) and generates
traced JAX code for eval (hot path).
"""

import math
from typing import Any, Callable, Dict, List, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax


# JAX-compatible MIR operations
def jax_fadd(a, b):
    return a + b

def jax_fsub(a, b):
    return a - b

def jax_fmul(a, b):
    return a * b

def jax_fdiv(a, b):
    return jnp.where(b != 0, a / b, jnp.where(a > 0, jnp.inf, jnp.where(a < 0, -jnp.inf, jnp.nan)))

def jax_fneg(a):
    return -a

def jax_flt(a, b):
    return a < b

def jax_fle(a, b):
    return a <= b

def jax_fgt(a, b):
    return a > b

def jax_fge(a, b):
    return a >= b

def jax_feq(a, b):
    return a == b

def jax_fne(a, b):
    return a != b

def jax_pow(base, exp):
    return jnp.power(base, exp)

def jax_ln(x):
    return jnp.log(jnp.maximum(x, 1e-300))  # Safe log

def jax_exp(x):
    return jnp.exp(jnp.clip(x, -700, 700))  # Safe exp

def jax_sqrt(x):
    return jnp.sqrt(jnp.maximum(x, 0.0))

def jax_sin(x):
    return jnp.sin(x)

def jax_cos(x):
    return jnp.cos(x)

def jax_tan(x):
    return jnp.tan(x)

def jax_atan(x):
    return jnp.arctan(x)

def jax_atan2(y, x):
    return jnp.arctan2(y, x)

def jax_abs(x):
    return jnp.abs(x)

def jax_min(a, b):
    return jnp.minimum(a, b)

def jax_max(a, b):
    return jnp.maximum(a, b)

def jax_ifcast(x):
    return jnp.float64(x)

def jax_optbarrier(x):
    return x


JAX_OPCODE_MAP = {
    'fadd': jax_fadd,
    'fsub': jax_fsub,
    'fmul': jax_fmul,
    'fdiv': jax_fdiv,
    'fneg': jax_fneg,
    'flt': jax_flt,
    'fle': jax_fle,
    'fgt': jax_fgt,
    'fge': jax_fge,
    'feq': jax_feq,
    'fne': jax_fne,
    'pow': jax_pow,
    'ln': jax_ln,
    'exp': jax_exp,
    'sqrt': jax_sqrt,
    'sin': jax_sin,
    'cos': jax_cos,
    'tan': jax_tan,
    'atan': jax_atan,
    'atan2': jax_atan2,
    'abs': jax_abs,
    'min': jax_min,
    'max': jax_max,
    'ifcast': jax_ifcast,
    'optbarrier': jax_optbarrier,
}


@dataclass
class OpenVAFToJAX:
    """Translator from OpenVAF MIR to JAX functions.

    This class provides:
    - init_fn: Python function to compute cache values (run once at setup)
    - eval_fn: JAX-traced function for model evaluation (JIT-compilable)
    """

    module: Any
    dae_data: Dict = None
    _init_fn: Callable = None
    _eval_fn: Callable = None
    _cache_size: int = 0

    def __post_init__(self):
        """Initialize data from module."""
        self.dae_data = self.module.get_dae_system()
        self._cache_size = self.module.num_cached_values

    def translate(self) -> Callable:
        """Generate the eval function.

        Returns:
            A callable that takes input array and returns (residuals, jacobian)
        """
        self._build_init_fn()
        self._build_eval_fn()
        return self._create_wrapped_eval()

    def _build_init_fn(self):
        """Build the init function from MIR."""
        # Import our Python MIR interpreter for init
        # (init runs once at setup, doesn't need JAX)
        import sys
        from pathlib import Path

        # Add scripts to path for mir_codegen
        scripts_path = Path(__file__).parent.parent / "scripts"
        if str(scripts_path) not in sys.path:
            sys.path.insert(0, str(scripts_path))

        from mir_codegen import generate_init_function
        self._init_fn = generate_init_function(self.module)

    def _build_eval_fn(self):
        """Build the eval function from MIR for JAX tracing."""
        eval_mir = self.module.get_mir_instructions()
        metadata = self.module.get_codegen_metadata()

        constants = eval_mir['constants']
        params = eval_mir.get('params', [])
        instructions = eval_mir['instructions']
        blocks = eval_mir['blocks']

        eval_param_mapping = metadata['eval_param_mapping']
        cache_info = metadata['cache_info']
        residuals_meta = metadata['residuals']
        jacobian_meta = metadata['jacobian']

        # Build cache-to-param index mapping
        cache_to_param_idx = {}
        for ci in cache_info:
            eval_param = ci['eval_param']
            if eval_param.startswith('v'):
                param_idx = int(eval_param[1:])
                cache_to_param_idx[ci['cache_idx']] = param_idx

        # Build param name to index mapping
        param_name_to_idx = {}
        for name, mir_var in eval_param_mapping.items():
            for i, p in enumerate(params):
                if p == mir_var:
                    param_name_to_idx[name] = i
                    break

        # For simple straight-line code (no complex control flow),
        # we can build a JAX function directly
        # Group instructions by block
        block_instrs = {}
        for instr in instructions:
            block = instr.get('block', 'default')
            if block not in block_instrs:
                block_instrs[block] = []
            block_instrs[block].append(instr)

        def eval_fn(input_array: jnp.ndarray, cache: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Evaluate the device model.

            Args:
                input_array: Array of input parameter values (voltages, params)
                cache: Array of cache values from init

            Returns:
                (residuals, jacobian) arrays
            """
            # Initialize values dict with constants
            values = {k: jnp.float64(v) for k, v in constants.items()}

            # Set input params
            for i, var in enumerate(params):
                if i < len(input_array):
                    values[var] = input_array[i]

            # Set cache values
            for cache_idx, param_idx in cache_to_param_idx.items():
                if param_idx < len(params) and cache_idx < len(cache):
                    values[params[param_idx]] = cache[cache_idx]

            # Execute instructions (simple case: single block, no control flow)
            # For complex models with control flow, we'd need lax.cond/lax.switch
            for block_name in sorted(block_instrs.keys()):
                for instr in block_instrs[block_name]:
                    opcode = instr['opcode']
                    result_var = instr.get('result')
                    operands = instr.get('operands', [])

                    if opcode in ('jmp', 'br', 'phi'):
                        # Skip control flow for now
                        continue

                    if opcode in JAX_OPCODE_MAP:
                        func = JAX_OPCODE_MAP[opcode]
                        op_values = [values.get(op, jnp.float64(0.0)) for op in operands]
                        if result_var:
                            values[result_var] = func(*op_values)
                    elif opcode == 'call':
                        # Handle special function calls
                        if result_var:
                            values[result_var] = jnp.float64(0.0)

            # Extract residuals
            n_residuals = len(residuals_meta)
            resist_residuals = jnp.zeros(n_residuals)
            react_residuals = jnp.zeros(n_residuals)

            for i, r in enumerate(residuals_meta):
                resist_residuals = resist_residuals.at[i].set(
                    values.get(r['resist_var'], jnp.float64(0.0))
                )
                react_residuals = react_residuals.at[i].set(
                    values.get(r['react_var'], jnp.float64(0.0))
                )

            # Extract jacobian
            n_jacobian = len(jacobian_meta)
            resist_jacobian = jnp.zeros(n_jacobian)
            react_jacobian = jnp.zeros(n_jacobian)

            for i, j in enumerate(jacobian_meta):
                resist_jacobian = resist_jacobian.at[i].set(
                    values.get(j['resist_var'], jnp.float64(0.0))
                )
                react_jacobian = react_jacobian.at[i].set(
                    values.get(j['react_var'], jnp.float64(0.0))
                )

            return (resist_residuals, react_residuals), (resist_jacobian, react_jacobian)

        self._eval_fn = eval_fn

    def _create_wrapped_eval(self) -> Callable:
        """Create wrapped eval that handles dict I/O."""
        metadata = self.module.get_codegen_metadata()
        eval_param_mapping = metadata['eval_param_mapping']
        residuals_meta = metadata['residuals']
        jacobian_meta = metadata['jacobian']

        mir_params = self.module.get_mir_instructions().get('params', [])
        param_name_to_idx = {}
        for name, mir_var in eval_param_mapping.items():
            for i, p in enumerate(mir_params):
                if p == mir_var:
                    param_name_to_idx[name] = i
                    break

        init_fn = self._init_fn
        eval_fn = self._eval_fn
        n_params = len(mir_params)
        cache_size = self._cache_size

        def wrapped_eval(inputs: List[float]) -> Tuple[Dict, Dict]:
            """Evaluate the device.

            Args:
                inputs: List of input values in param order

            Returns:
                (residuals_dict, jacobian_dict)
            """
            # Convert to arrays
            input_array = jnp.array(inputs, dtype=jnp.float64)

            # For now, compute cache on each call
            # In a real implementation, cache would be computed once at init
            # and stored in the device instance

            # Build init params from eval inputs
            # Map semantic names to values from eval inputs
            init_params = {}
            init_param_mapping = metadata.get('init_param_mapping', {})

            # First, build a mapping from eval semantic names to input values
            eval_values = {}
            for name, mir_var in eval_param_mapping.items():
                if name in param_name_to_idx:
                    idx = param_name_to_idx[name]
                    if idx < len(inputs):
                        eval_values[name] = inputs[idx]

            # Now map to init params
            for name, mir_var in init_param_mapping.items():
                if name in eval_values:
                    # Direct mapping from eval param
                    init_params[name] = eval_values[name]
                elif name.endswith('_given'):
                    # Check if base param is in eval_values (non-zero means given)
                    base_name = name[:-6]  # Remove '_given' suffix
                    if base_name in eval_values and eval_values[base_name] != 0:
                        init_params[name] = True
                    else:
                        init_params[name] = False
                elif name == 'mfactor' and 'mfactor' in eval_values:
                    init_params[name] = eval_values['mfactor']
                else:
                    init_params[name] = 0.0

            # Compute cache
            cache_list = init_fn(**init_params)
            cache = jnp.array([v if v is not None else 0.0 for v in cache_list], dtype=jnp.float64)

            # Run eval
            (resist_res, react_res), (resist_jac, react_jac) = eval_fn(input_array, cache)

            # Build output dicts
            residuals = {}
            for i, r in enumerate(residuals_meta):
                node_idx = r['residual_idx']
                residuals[node_idx] = {
                    'resist': float(resist_res[i]),
                    'react': float(react_res[i]),
                }

            jacobian = {}
            for i, j in enumerate(jacobian_meta):
                key = (j['row'], j['col'])
                jacobian[key] = {
                    'resist': float(resist_jac[i]),
                    'react': float(react_jac[i]),
                }

            return residuals, jacobian

        return wrapped_eval

    def get_jit_eval(self) -> Callable:
        """Get a JIT-compiled version of the eval function.

        Returns:
            JIT-compiled eval function taking (input_array, cache) arrays
        """
        return jax.jit(self._eval_fn)
