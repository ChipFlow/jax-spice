"""JAX code generator for OpenVAF MIR.

This module generates JAX-traceable functions from OpenVAF MIR.
Unlike pure_callback, the generated code can be JIT-compiled and run on GPU.

Two modes:
1. Straight-line: For models without control flow (resistor, capacitor)
2. CFG-to-conditional: For models with branches (diode, PSP103)

See docs/reference/openvaf/EVAL_FUNCTION_GENERATION.md for details.
"""

from typing import Any, Callable, Dict, List, Tuple, Set
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jax import lax


# Pre-allocated MIR constants (always present, not in MIR dumps)
# See docs/reference/openvaf/OPENVAF_MIR_CONSTANTS.md
PREALLOCATED_CONSTANTS = {
    'v0': None,   # GRAVESTONE - dead value placeholder
    'v1': False,  # FALSE
    'v2': True,   # TRUE
    'v3': 0.0,    # F_ZERO
    'v4': 0,      # ZERO (integer)
    'v5': 1,      # ONE (integer)
    'v6': 1.0,    # F_ONE
    'v7': -1.0,   # F_N_ONE
}


def jax_safe_add(a, b):
    """Addition with safe handling of indeterminate forms.

    For circuit simulation, inf + (-inf) = 0 (common convention).
    This matches native OSDI behavior where infinities from division-by-zero
    often cancel in subsequent additions.
    """
    result = a + b
    # Handle inf + (-inf) = NaN -> return 0
    is_indeterminate = jnp.isnan(result) & jnp.isinf(a) & jnp.isinf(b)
    return jnp.where(is_indeterminate, 0.0, result)


def jax_safe_sub(a, b):
    """Subtraction with safe handling of indeterminate forms.

    For circuit simulation, inf - inf = 0 (common convention).
    """
    result = a - b
    # Handle inf - inf = NaN -> return 0
    is_indeterminate = jnp.isnan(result) & jnp.isinf(a) & jnp.isinf(b)
    return jnp.where(is_indeterminate, 0.0, result)


def jax_safe_mul(a, b):
    """Multiplication with safe handling of indeterminate forms.

    For circuit simulation, 0 * inf = 0 (common convention).
    This handles cases where a zero coefficient should kill an infinite term.
    """
    result = a * b
    # Handle 0 * inf = NaN -> return 0
    is_zero_times_inf = jnp.isnan(result) & ((a == 0) | (b == 0))
    return jnp.where(is_zero_times_inf, 0.0, result)


def jax_safe_div(a, b):
    """Division with safe handling of zero.

    For circuit simulation, 0/0 = 0 (common convention for indeterminate forms).
    This matches native OSDI behavior.
    """
    return jnp.where(b != 0, a / b,
                     jnp.where(a > 0, jnp.inf,
                              jnp.where(a < 0, -jnp.inf, 0.0)))  # 0/0 = 0


def jax_safe_ln(x):
    """Logarithm with safe handling of non-positive."""
    return jnp.log(jnp.maximum(x, 1e-300))


def jax_safe_exp(x):
    """Exponential with overflow protection."""
    return jnp.exp(jnp.clip(x, -700, 700))


def jax_safe_sqrt(x):
    """Square root with safe handling of negative."""
    return jnp.sqrt(jnp.maximum(x, 0.0))


def jax_safe_pow(base, exp):
    """Power with safe handling of edge cases."""
    return jnp.power(jnp.maximum(base, 0.0), exp)


# Map MIR opcodes to JAX operations
JAX_BINARY_OPS = {
    # Float arithmetic - using safe versions to handle indeterminate forms
    'fadd': jax_safe_add,
    'fsub': jax_safe_sub,
    'fmul': jax_safe_mul,
    'fdiv': jax_safe_div,
    # Float comparisons
    'flt': lambda a, b: a < b,
    'fle': lambda a, b: a <= b,
    'fgt': lambda a, b: a > b,
    'fge': lambda a, b: a >= b,
    'feq': lambda a, b: a == b,
    'fne': lambda a, b: a != b,
    # Integer comparisons (used by PSP103)
    'ilt': lambda a, b: a < b,
    'ile': lambda a, b: a <= b,
    'igt': lambda a, b: a > b,
    'ige': lambda a, b: a >= b,
    'ieq': lambda a, b: a == b,
    'ine': lambda a, b: a != b,
    # Math functions
    'pow': jax_safe_pow,
    'atan2': jnp.arctan2,
    'hypot': jnp.hypot,
    'min': jnp.minimum,
    'max': jnp.maximum,
}

JAX_UNARY_OPS = {
    # Negation
    'fneg': lambda a: -a,
    'ineg': lambda a: -a,  # Integer negation (used by PSP103)
    # Boolean/bitwise
    'bnot': lambda a: jnp.logical_not(a),  # Boolean NOT (used by PSP103)
    'inot': lambda a: jnp.logical_not(a),  # Integer bitwise NOT
    # Math functions
    'ln': jax_safe_ln,
    'exp': jax_safe_exp,
    'sqrt': jax_safe_sqrt,
    'log': lambda x: jnp.log10(jnp.maximum(x, 1e-300)),  # Base-10 log
    'abs': jnp.abs,
    'floor': jnp.floor,
    'ceil': jnp.ceil,
    # Trig functions
    'sin': jnp.sin,
    'cos': jnp.cos,
    'tan': jnp.tan,
    'asin': jnp.arcsin,
    'acos': jnp.arccos,
    'atan': jnp.arctan,
    # Hyperbolic functions
    'sinh': jnp.sinh,
    'cosh': jnp.cosh,
    'tanh': jnp.tanh,
    'asinh': jnp.arcsinh,
    'acosh': jnp.arccosh,
    'atanh': jnp.arctanh,
    # Type casts
    'ifcast': lambda x: jnp.float64(x),
    'ficast': lambda x: jnp.int32(x),
    'bfcast': lambda x: jnp.float64(x),
    'bicast': lambda x: jnp.int32(x),
    'ibcast': lambda x: x != 0,
    'fbcast': lambda x: x != 0.0,
    # Optimization barrier
    'optbarrier': lambda x: x,
}


@dataclass
class MIRAnalysis:
    """Analysis results for MIR control flow."""
    has_branches: bool
    has_phi_nodes: bool
    num_branches: int
    num_phi_nodes: int
    varying_dependent_branches: int
    fixed_only_branches: int
    blocks: Dict[str, Dict]
    instructions: List[Dict]
    producers: Dict[str, Dict]  # var -> instruction that produces it
    param_to_idx: Dict[str, int]  # param var -> index
    varying_indices: Set[int]  # indices of varying params (voltage/current)


def analyze_mir(module) -> MIRAnalysis:
    """Analyze MIR for control flow complexity.

    Returns analysis results including whether the model has branches,
    and which branches depend on varying (voltage/current) params.
    """
    eval_mir = module.get_mir_instructions()
    instructions = eval_mir['instructions']
    blocks = eval_mir['blocks']
    params = eval_mir.get('params', [])

    # Get param kinds to identify varying params
    param_kinds = list(module.param_kinds)
    varying_kinds = {'voltage', 'current'}
    varying_indices = {i for i, k in enumerate(param_kinds) if k in varying_kinds}

    # Build producer map
    producers = {}
    for inst in instructions:
        result = inst.get('result')
        if result:
            producers[result] = inst

    # Build param index map
    param_to_idx = {p: i for i, p in enumerate(params)}

    # Count branches and phi nodes
    branches = [i for i in instructions if i['opcode'] == 'br']
    phi_nodes = [i for i in instructions if i['opcode'] == 'phi']

    # Analyze branch dependencies
    def trace_deps(var, visited=None):
        if visited is None:
            visited = set()
        if var in visited:
            return set()
        visited.add(var)
        deps = set()
        if var in param_to_idx:
            deps.add(param_to_idx[var])
        elif var in producers:
            inst = producers[var]
            operands = inst.get('operands', [])
            for phi_op in inst.get('phi_operands', []):
                if phi_op.get('value'):
                    operands.append(phi_op['value'])
            for op in operands:
                if op:
                    deps |= trace_deps(op, visited)
        return deps

    varying_dependent = 0
    fixed_only = 0
    for br in branches:
        deps = trace_deps(br.get('condition'))
        if deps & varying_indices:
            varying_dependent += 1
        else:
            fixed_only += 1

    return MIRAnalysis(
        has_branches=len(branches) > 0,
        has_phi_nodes=len(phi_nodes) > 0,
        num_branches=len(branches),
        num_phi_nodes=len(phi_nodes),
        varying_dependent_branches=varying_dependent,
        fixed_only_branches=fixed_only,
        blocks=blocks,
        instructions=instructions,
        producers=producers,
        param_to_idx=param_to_idx,
        varying_indices=varying_indices,
    )


def generate_straight_line_eval(module) -> Callable:
    """Generate JAX eval function for models without control flow.

    This generates a pure JAX function that can be JIT-compiled.
    Only works for models with no branches (resistor, capacitor).

    Args:
        module: openvaf_py module object

    Returns:
        eval_fn(params_array, cache_array) -> ((res_resist, res_react), (jac_resist, jac_react))
    """
    eval_mir = module.get_mir_instructions()
    metadata = module.get_codegen_metadata()

    constants = eval_mir['constants']
    params = eval_mir.get('params', [])
    instructions = eval_mir['instructions']
    function_decls = eval_mir.get('function_decls', {})

    residuals_meta = metadata['residuals']
    jacobian_meta = metadata['jacobian']
    cache_info = metadata['cache_info']

    n_residuals = len(residuals_meta)
    n_jacobian = len(jacobian_meta)
    n_params = len(params)

    # Build cache index -> param index mapping
    cache_to_param_idx = {}
    for ci in cache_info:
        eval_param = ci['eval_param']
        if eval_param.startswith('v'):
            try:
                param_idx = int(eval_param[1:])
                cache_to_param_idx[ci['cache_idx']] = param_idx
            except ValueError:
                pass

    # Pre-compute which instructions produce which results
    # This allows us to generate code in dependency order
    result_to_inst = {}
    for inst in instructions:
        result = inst.get('result')
        if result:
            result_to_inst[result] = inst

    # Get output variable names
    resist_res_vars = [r['resist_var'] for r in residuals_meta]
    react_res_vars = [r['react_var'] for r in residuals_meta]
    resist_jac_vars = [j['resist_var'] for j in jacobian_meta]
    react_jac_vars = [j['react_var'] for j in jacobian_meta]

    # Collect all needed output vars
    output_vars = set(resist_res_vars + react_res_vars + resist_jac_vars + react_jac_vars)

    def eval_fn(input_array: jnp.ndarray, cache: jnp.ndarray):
        """Evaluate the device model.

        Args:
            input_array: Array of input parameter values
            cache: Array of cache values from init

        Returns:
            ((resist_residuals, react_residuals), (resist_jacobian, react_jacobian))
        """
        # Initialize values dict with pre-allocated constants
        # Convert to JAX-compatible types
        values = {
            'v1': jnp.array(0.0),  # FALSE as 0.0
            'v2': jnp.array(1.0),  # TRUE as 1.0
            'v3': jnp.array(0.0),  # F_ZERO
            'v4': jnp.array(0.0),  # ZERO (as float)
            'v5': jnp.array(1.0),  # ONE (as float)
            'v6': jnp.array(1.0),  # F_ONE
            'v7': jnp.array(-1.0), # F_N_ONE
        }

        # Add MIR constants
        for var, val in constants.items():
            if val is not None:
                values[var] = jnp.array(float(val) if isinstance(val, (int, float)) else val)

        # Map params from input array
        for i, var in enumerate(params):
            if i < input_array.shape[0]:
                values[var] = input_array[i]

        # Override with cache values
        for cache_idx, param_idx in cache_to_param_idx.items():
            if param_idx < len(params) and cache_idx < cache.shape[0]:
                values[params[param_idx]] = cache[cache_idx]

        # Execute instructions in order
        for inst in instructions:
            opcode = inst['opcode']
            result = inst.get('result')
            operands = inst.get('operands', [])

            # Skip control flow (should not exist in straight-line code)
            if opcode in ('jmp', 'br', 'phi'):
                continue

            # Get operand values
            op_vals = []
            for op in operands:
                if op in values:
                    op_vals.append(values[op])
                else:
                    op_vals.append(jnp.array(0.0))

            # Execute operation
            if opcode in JAX_BINARY_OPS and len(op_vals) >= 2:
                val = JAX_BINARY_OPS[opcode](op_vals[0], op_vals[1])
            elif opcode in JAX_UNARY_OPS and len(op_vals) >= 1:
                val = JAX_UNARY_OPS[opcode](op_vals[0])
            elif opcode == 'call':
                # Handle function calls - look up function name from func_ref
                func_ref = inst.get('func_ref', '')
                func_name = function_decls.get(func_ref, {}).get('name', '')

                if 'SimParam' in func_name:
                    # gmin is typically 1e-12 for convergence
                    val = jnp.array(1e-12)
                elif 'TimeDerivative' in func_name:
                    # For DC analysis, time derivatives are 0
                    val = jnp.array(0.0)
                elif 'NodeDerivative' in func_name:
                    # For DC analysis, frequency-dependent terms are 0
                    val = jnp.array(0.0)
                elif 'limexp' in func_name.lower():
                    val = jax_safe_exp(jnp.minimum(op_vals[0], 700.0)) if op_vals else jnp.array(1.0)
                else:
                    # Unknown function - default to 0
                    val = jnp.array(0.0)
            else:
                # Unknown opcode - use first operand or zero
                val = op_vals[0] if op_vals else jnp.array(0.0)

            if result:
                values[result] = val

        # Extract outputs
        def get_val(var):
            return values.get(var, jnp.array(0.0))

        resist_res = jnp.array([get_val(v) for v in resist_res_vars])
        react_res = jnp.array([get_val(v) for v in react_res_vars])
        resist_jac = jnp.array([get_val(v) for v in resist_jac_vars])
        react_jac = jnp.array([get_val(v) for v in react_jac_vars])

        return (resist_res, react_res), (resist_jac, react_jac)

    return eval_fn


def generate_cfg_eval(module) -> Callable:
    """Generate JAX eval function for models with control flow.

    Uses a flat execution approach that converts control flow to data flow:
    - Execute all instructions in topological order
    - Track which blocks are "active" using masks
    - PHI nodes select values based on predecessor masks
    - Branches update the active masks

    This avoids exponential blowup from recursive tracing.

    OPTIMIZATION: Uses array-based storage instead of dict lookups for speed.

    Args:
        module: openvaf_py module object

    Returns:
        eval_fn(params_array, cache_array) -> ((res_resist, res_react), (jac_resist, jac_react))
    """
    eval_mir = module.get_mir_instructions()
    metadata = module.get_codegen_metadata()

    constants = eval_mir['constants']
    params = eval_mir.get('params', [])
    instructions = eval_mir['instructions']
    blocks_info = eval_mir['blocks']
    function_decls = eval_mir.get('function_decls', {})

    residuals_meta = metadata['residuals']
    jacobian_meta = metadata['jacobian']
    cache_info = metadata['cache_info']

    # Build var -> index mapping for array-based storage
    all_vars = set()
    # Pre-allocated constants
    for i in range(8):
        all_vars.add(f'v{i}')
    # MIR constants
    all_vars.update(constants.keys())
    # Params
    all_vars.update(params)
    # Instruction results
    for inst in instructions:
        result = inst.get('result')
        if result:
            all_vars.add(result)
        # Also add any referenced vars
        for op in inst.get('operands', []):
            if op:
                all_vars.add(op)
        for phi_op in inst.get('phi_operands', []):
            val = phi_op.get('value')
            if val:
                all_vars.add(val)

    # Create sorted list for deterministic indexing
    var_list = sorted(all_vars)
    var_to_idx = {v: i for i, v in enumerate(var_list)}
    n_vars = len(var_list)

    # Pre-compute constant values array
    const_vals = [0.0] * n_vars
    # Pre-allocated constants
    const_vals[var_to_idx['v1']] = 0.0  # FALSE
    const_vals[var_to_idx['v2']] = 1.0  # TRUE
    const_vals[var_to_idx['v3']] = 0.0  # F_ZERO
    const_vals[var_to_idx['v4']] = 0.0  # ZERO
    const_vals[var_to_idx['v5']] = 1.0  # ONE
    const_vals[var_to_idx['v6']] = 1.0  # F_ONE
    const_vals[var_to_idx['v7']] = -1.0  # F_N_ONE
    # MIR constants
    for var, val in constants.items():
        if val is not None and var in var_to_idx:
            fval = float(val) if isinstance(val, (int, float)) else val
            # Sanitize NaN constants - replace with 0.0 (safe default for unused constants)
            if isinstance(fval, float) and (fval != fval):  # NaN check
                fval = 0.0
            const_vals[var_to_idx[var]] = fval

    const_array = jnp.array(const_vals)

    # Pre-compute param indices
    param_indices = [var_to_idx[p] for p in params]

    # Build cache index -> var index mapping
    cache_to_var_idx = {}
    for ci in cache_info:
        eval_param = ci['eval_param']
        if eval_param.startswith('v'):
            try:
                param_idx = int(eval_param[1:])
                if param_idx < len(params):
                    cache_to_var_idx[ci['cache_idx']] = var_to_idx[params[param_idx]]
            except ValueError:
                pass

    # Pre-compute output indices
    resist_res_indices = [var_to_idx.get(r['resist_var'], 0) for r in residuals_meta]
    react_res_indices = [var_to_idx.get(r['react_var'], 0) for r in residuals_meta]
    resist_jac_indices = [var_to_idx.get(j['resist_var'], 0) for j in jacobian_meta]
    react_jac_indices = [var_to_idx.get(j['react_var'], 0) for j in jacobian_meta]

    # Pre-process instructions into compact form for fast execution
    # Convert string opcodes/vars to indices
    compiled_instructions = []
    block_to_idx = {name: i for i, name in enumerate(blocks_info.keys())}

    for inst in instructions:
        opcode = inst['opcode']
        result_idx = var_to_idx.get(inst.get('result'), -1)
        block_idx = block_to_idx.get(inst.get('block', 'default'), 0)

        if opcode == 'phi':
            phi_ops = []
            for phi_op in inst.get('phi_operands', []):
                val_idx = var_to_idx.get(phi_op.get('value'), 0)
                pred_idx = block_to_idx.get(phi_op.get('block'), -1)
                phi_ops.append((val_idx, pred_idx))
            compiled_instructions.append(('phi', result_idx, block_idx, phi_ops))
        elif opcode == 'br':
            cond_idx = var_to_idx.get(inst.get('condition'), 0)
            true_idx = block_to_idx.get(inst.get('true_block'), 0)
            false_idx = block_to_idx.get(inst.get('false_block'), 0)
            compiled_instructions.append(('br', block_idx, cond_idx, true_idx, false_idx))
        elif opcode == 'jmp':
            dest_idx = block_to_idx.get(inst.get('destination'), 0)
            compiled_instructions.append(('jmp', block_idx, dest_idx))
        elif opcode == 'call':
            # Track function reference for proper call handling
            op_indices = [var_to_idx.get(op, 0) for op in inst.get('operands', [])]
            func_ref = inst.get('func_ref', '')
            func_name = function_decls.get(func_ref, {}).get('name', '')
            compiled_instructions.append(('call', result_idx, op_indices, func_name))
        else:
            op_indices = [var_to_idx.get(op, 0) for op in inst.get('operands', [])]
            compiled_instructions.append((opcode, result_idx, op_indices))

    n_blocks = len(blocks_info)

    def eval_fn(input_array: jnp.ndarray, cache: jnp.ndarray):
        """Evaluate the device model using flat array-based execution."""
        # Start with constants
        vals = const_array.copy()

        # Load params from input array
        for i, param_idx in enumerate(param_indices):
            if i < input_array.shape[0]:
                vals = vals.at[param_idx].set(input_array[i])

        # Override with cache values
        for cache_idx, var_idx in cache_to_var_idx.items():
            if cache_idx < cache.shape[0]:
                vals = vals.at[var_idx].set(cache[cache_idx])

        # Block reachability tracking for PHI node resolution
        # block_reached[i] = 1.0 if block i was executed, 0.0 otherwise
        # Entry block (block 0) is always reached
        block_reached = jnp.zeros(n_blocks).at[0].set(1.0)

        # Execute compiled instructions
        for inst in compiled_instructions:
            opcode = inst[0]

            if opcode == 'phi':
                _, result_idx, block_idx, phi_ops = inst
                if result_idx < 0 or not phi_ops:
                    continue

                # PHI: select value from the predecessor block that was reached
                # Start with first value as default
                first_val_idx, first_pred_idx = phi_ops[0]
                merged_val = vals[first_val_idx]

                # Select value from the predecessor that was actually reached
                for val_idx, pred_block_idx in phi_ops:
                    if pred_block_idx >= 0:
                        pred_reached = block_reached[pred_block_idx]
                        op_val = vals[val_idx]
                        # If this predecessor was reached, use its value
                        merged_val = jnp.where(pred_reached > 0.5, op_val, merged_val)

                vals = vals.at[result_idx].set(merged_val)

            elif opcode == 'br':
                # BR: propagate reachability based on condition
                _, block_idx, cond_idx, true_idx, false_idx = inst
                this_block_reached = block_reached[block_idx]
                cond_val = vals[cond_idx]

                # true_block is reached if this block was reached AND condition is true
                true_reaches = this_block_reached * cond_val
                block_reached = block_reached.at[true_idx].set(
                    jnp.maximum(block_reached[true_idx], true_reaches)
                )

                # false_block is reached if this block was reached AND condition is false
                false_reaches = this_block_reached * (1.0 - cond_val)
                block_reached = block_reached.at[false_idx].set(
                    jnp.maximum(block_reached[false_idx], false_reaches)
                )

            elif opcode == 'jmp':
                # JMP: propagate reachability unconditionally
                _, block_idx, dest_idx = inst
                this_block_reached = block_reached[block_idx]
                block_reached = block_reached.at[dest_idx].set(
                    jnp.maximum(block_reached[dest_idx], this_block_reached)
                )

            elif opcode == 'call':
                # Call instruction has func_name as 4th element
                _, result_idx, op_indices, func_name = inst
                if result_idx < 0:
                    continue

                op_vals = [vals[i] for i in op_indices]

                # Handle different function types
                if 'SimParam' in func_name:
                    # SimParam returns simulation parameters
                    # gmin is typically 1e-12 for convergence
                    val = jnp.array(1e-12)
                elif 'TimeDerivative' in func_name:
                    # For DC analysis, time derivatives are 0
                    val = jnp.array(0.0)
                elif 'NodeDerivative' in func_name:
                    # For DC analysis, frequency-dependent terms are 0
                    val = jnp.array(0.0)
                elif 'limexp' in func_name.lower():
                    # Safe exponential
                    val = jax_safe_exp(jnp.minimum(op_vals[0], 700.0)) if op_vals else jnp.array(1.0)
                else:
                    # Unknown function - default to 0
                    val = jnp.array(0.0)

                vals = vals.at[result_idx].set(val)

            else:
                # Regular operation
                _, result_idx, op_indices = inst
                if result_idx < 0:
                    continue

                op_vals = [vals[i] for i in op_indices]

                if opcode in JAX_BINARY_OPS and len(op_vals) >= 2:
                    val = JAX_BINARY_OPS[opcode](op_vals[0], op_vals[1])
                elif opcode in JAX_UNARY_OPS and len(op_vals) >= 1:
                    val = JAX_UNARY_OPS[opcode](op_vals[0])
                else:
                    val = op_vals[0] if op_vals else 0.0

                vals = vals.at[result_idx].set(val)

        # Extract outputs
        resist_res = vals[jnp.array(resist_res_indices)]
        react_res = vals[jnp.array(react_res_indices)]
        resist_jac = vals[jnp.array(resist_jac_indices)]
        react_jac = vals[jnp.array(react_jac_indices)]

        return (resist_res, react_res), (resist_jac, react_jac)

    return eval_fn


def generate_lax_loop_eval(module) -> Callable:
    """Generate a lax.fori_loop based interpreter for large models.

    This creates a small XLA graph that loops at runtime, avoiding the
    exponential blowup from unrolling 20k+ instructions during tracing.

    The key insight: we encode all instructions as data arrays and use
    lax.fori_loop to iterate through them. Each loop iteration executes
    one instruction using dynamic indexing and lax.switch for dispatch.

    Args:
        module: openvaf_py module object

    Returns:
        eval_fn(params_array, cache_array) -> ((res_resist, res_react), (jac_resist, jac_react))
    """
    import numpy as np

    eval_mir = module.get_mir_instructions()
    metadata = module.get_codegen_metadata()

    constants = eval_mir['constants']
    params = eval_mir.get('params', [])
    instructions = eval_mir['instructions']
    blocks_info = eval_mir['blocks']

    residuals_meta = metadata['residuals']
    jacobian_meta = metadata['jacobian']
    cache_info = metadata['cache_info']

    # Build var -> index mapping
    all_vars = set()
    for i in range(8):
        all_vars.add(f'v{i}')
    all_vars.update(constants.keys())
    all_vars.update(params)
    for inst in instructions:
        result = inst.get('result')
        if result:
            all_vars.add(result)
        for op in inst.get('operands', []):
            if op:
                all_vars.add(op)
        for phi_op in inst.get('phi_operands', []):
            val = phi_op.get('value')
            if val:
                all_vars.add(val)
    # Also add cache variable names (they may not be in params or instructions)
    for ci in cache_info:
        all_vars.add(ci['eval_param'])

    var_list = sorted(all_vars)
    var_to_idx = {v: i for i, v in enumerate(var_list)}
    n_vars = len(var_list)

    # Build constant array
    const_vals = np.zeros(n_vars, dtype=np.float64)
    const_vals[var_to_idx['v1']] = 0.0
    const_vals[var_to_idx['v2']] = 1.0
    const_vals[var_to_idx['v3']] = 0.0
    const_vals[var_to_idx['v4']] = 0.0
    const_vals[var_to_idx['v5']] = 1.0
    const_vals[var_to_idx['v6']] = 1.0
    const_vals[var_to_idx['v7']] = -1.0
    for var, val in constants.items():
        if val is not None and var in var_to_idx:
            fval = float(val) if isinstance(val, (int, float)) else val
            # Sanitize NaN constants - replace with 0.0 (safe default for unused constants)
            if isinstance(fval, float) and np.isnan(fval):
                fval = 0.0
            const_vals[var_to_idx[var]] = fval
    const_array = jnp.array(const_vals)

    # Param indices
    param_indices_np = np.array([var_to_idx[p] for p in params], dtype=np.int32)
    param_indices = jnp.array(param_indices_np)
    n_params = len(params)

    # Cache mappings - eval_param is a variable name like "v2616" that should be looked up in var_to_idx
    cache_indices_list = []
    cache_var_indices_list = []
    for ci in cache_info:
        eval_param = ci['eval_param']
        if eval_param in var_to_idx:
            cache_indices_list.append(ci['cache_idx'])
            cache_var_indices_list.append(var_to_idx[eval_param])
    cache_indices = jnp.array(cache_indices_list, dtype=jnp.int32)
    cache_var_indices = jnp.array(cache_var_indices_list, dtype=jnp.int32)
    n_cache_mappings = len(cache_indices_list)

    # Output indices
    resist_res_indices = jnp.array([var_to_idx.get(r['resist_var'], 0) for r in residuals_meta], dtype=jnp.int32)
    react_res_indices = jnp.array([var_to_idx.get(r['react_var'], 0) for r in residuals_meta], dtype=jnp.int32)
    resist_jac_indices = jnp.array([var_to_idx.get(j['resist_var'], 0) for j in jacobian_meta], dtype=jnp.int32)
    react_jac_indices = jnp.array([var_to_idx.get(j['react_var'], 0) for j in jacobian_meta], dtype=jnp.int32)

    block_to_idx = {name: i for i, name in enumerate(blocks_info.keys())}
    n_blocks = len(blocks_info)

    # Encode instructions for lax.fori_loop
    # Each instruction encoded as: [opcode, result_idx, op1_idx, op2_idx, block_idx, extra1, extra2, extra3]
    # Fixed 8 values per instruction for uniform indexing
    INST_SIZE = 8
    OPCODE_MAP = {
        # Float arithmetic (0-3)
        'fadd': 0, 'fsub': 1, 'fmul': 2, 'fdiv': 3,
        # Float comparisons (4-9)
        'flt': 4, 'fle': 5, 'fgt': 6, 'fge': 7, 'feq': 8, 'fne': 9,
        # Unary ops (10-18)
        'fneg': 10, 'ln': 11, 'exp': 12, 'sqrt': 13,
        'sin': 14, 'cos': 15, 'tan': 16, 'atan': 17, 'abs': 18,
        # Binary math (19-22)
        'pow': 19, 'min': 20, 'max': 21, 'atan2': 22,
        # Casts and misc (23-25)
        'ifcast': 23, 'optbarrier': 24, 'call': 25,
        # Integer comparisons (26-31) - used by PSP103
        'ilt': 26, 'ile': 27, 'igt': 28, 'ige': 29, 'ieq': 30, 'ine': 31,
        # Integer/boolean ops (32-35) - used by PSP103
        'ineg': 32, 'bnot': 33, 'inot': 34, 'bfcast': 35,
        # More trig (36-39)
        'asin': 36, 'acos': 37, 'sinh': 38, 'cosh': 39,
        # More math (40-45)
        'tanh': 40, 'asinh': 41, 'acosh': 42, 'atanh': 43, 'log': 44, 'hypot': 45,
        # More casts/misc (46-49)
        'ficast': 46, 'bicast': 47, 'ibcast': 48, 'fbcast': 49,
        'floor': 50, 'ceil': 51,
        # Control flow (100-102)
        'phi': 100, 'br': 101, 'jmp': 102, 'noop': 127,
    }

    # For PHI nodes with variable operands, we need a separate array
    # Main instruction array just references into phi_data array
    inst_data = []
    phi_data = []  # Flattened: [n_ops, val1, pred1, val2, pred2, ...]

    for inst in instructions:
        opcode = inst['opcode']
        opcode_int = OPCODE_MAP.get(opcode, 127)  # 127 = noop
        result_idx = var_to_idx.get(inst.get('result'), 0)
        block_idx = block_to_idx.get(inst.get('block', 'default'), 0)

        if opcode == 'phi':
            phi_ops = inst.get('phi_operands', [])
            phi_start = len(phi_data)
            phi_data.append(len(phi_ops))
            for phi_op in phi_ops:
                val_idx = var_to_idx.get(phi_op.get('value'), 0)
                pred_idx = block_to_idx.get(phi_op.get('block'), 0)
                phi_data.extend([val_idx, pred_idx])
            # [opcode, result_idx, phi_start, 0, block_idx, 0, 0, 0]
            inst_data.extend([100, result_idx, phi_start, 0, block_idx, 0, 0, 0])
        elif opcode == 'br':
            cond_idx = var_to_idx.get(inst.get('condition'), 0)
            true_idx = block_to_idx.get(inst.get('true_block'), 0)
            false_idx = block_to_idx.get(inst.get('false_block'), 0)
            # [opcode, 0, cond_idx, 0, block_idx, true_idx, false_idx, 0]
            inst_data.extend([101, 0, cond_idx, 0, block_idx, true_idx, false_idx, 0])
        elif opcode == 'jmp':
            dest_idx = block_to_idx.get(inst.get('destination'), 0)
            # [opcode, 0, 0, 0, block_idx, dest_idx, dest_idx, 0]
            inst_data.extend([102, 0, 0, 0, block_idx, dest_idx, dest_idx, 0])
        else:
            operands = inst.get('operands', [])
            op1_idx = var_to_idx.get(operands[0], 0) if len(operands) > 0 else 0
            op2_idx = var_to_idx.get(operands[1], 0) if len(operands) > 1 else 0
            # [opcode, result_idx, op1_idx, op2_idx, block_idx, 0, 0, 0]
            inst_data.extend([opcode_int, result_idx, op1_idx, op2_idx, block_idx, 0, 0, 0])

    inst_array = jnp.array(inst_data, dtype=jnp.int32)
    phi_array = jnp.array(phi_data if phi_data else [0], dtype=jnp.int32)
    n_instructions = len(instructions)

    def eval_fn(input_array: jnp.ndarray, cache: jnp.ndarray):
        """Evaluate using lax.fori_loop interpreter."""
        # Initialize values from constants
        vals = const_array.copy()

        # Load input params using scatter
        # Use lax.fori_loop for param loading too
        def load_param(i, v):
            idx = param_indices[i]
            val = input_array[i]
            return v.at[idx].set(val)

        vals = lax.fori_loop(0, jnp.minimum(input_array.shape[0], n_params), load_param, vals)

        # Load cache values
        def load_cache(i, v):
            cache_idx = cache_indices[i]
            var_idx = cache_var_indices[i]
            val = cache[cache_idx]
            return v.at[var_idx].set(val)

        if n_cache_mappings > 0:
            vals = lax.fori_loop(0, n_cache_mappings, load_cache, vals)

        # Block reachability tracking for PHI node resolution
        # block_reached[i] = 1.0 if block i was executed, 0.0 otherwise
        # Entry block (block 0) is always reached
        block_reached = jnp.zeros(n_blocks).at[0].set(1.0)

        # Execute instructions using lax.fori_loop
        def exec_instruction(i, state):
            vals, block_reached = state

            # Get instruction data
            base = i * INST_SIZE
            opcode = inst_array[base]
            result_idx = inst_array[base + 1]
            op1_idx = inst_array[base + 2]
            op2_idx = inst_array[base + 3]
            block_idx = inst_array[base + 4]
            extra1 = inst_array[base + 5]  # true_block for BR, dest for JMP
            extra2 = inst_array[base + 6]  # false_block for BR

            a = vals[op1_idx]
            b = vals[op2_idx]

            # Compute result for each opcode type
            # Float binary ops (0-3)
            is_fadd = (opcode == 0)
            is_fsub = (opcode == 1)
            is_fmul = (opcode == 2)
            is_fdiv = (opcode == 3)
            # Float comparisons (4-9)
            is_flt = (opcode == 4)
            is_fle = (opcode == 5)
            is_fgt = (opcode == 6)
            is_fge = (opcode == 7)
            is_feq = (opcode == 8)
            is_fne = (opcode == 9)
            # Binary math (19-22)
            is_pow = (opcode == 19)
            is_min = (opcode == 20)
            is_max = (opcode == 21)
            is_atan2 = (opcode == 22)
            # Integer comparisons (26-31) - used by PSP103
            is_ilt = (opcode == 26)
            is_ile = (opcode == 27)
            is_igt = (opcode == 28)
            is_ige = (opcode == 29)
            is_ieq = (opcode == 30)
            is_ine = (opcode == 31)
            # Hypot (45)
            is_hypot = (opcode == 45)

            # Unary ops (10-18)
            is_fneg = (opcode == 10)
            is_ln = (opcode == 11)
            is_exp = (opcode == 12)
            is_sqrt = (opcode == 13)
            is_sin = (opcode == 14)
            is_cos = (opcode == 15)
            is_tan = (opcode == 16)
            is_atan = (opcode == 17)
            is_abs = (opcode == 18)
            # Casts and misc (23-25)
            is_ifcast = (opcode == 23)
            is_optbarrier = (opcode == 24)
            is_call = (opcode == 25)
            # Integer/boolean ops (32-35) - used by PSP103
            is_ineg = (opcode == 32)
            is_bnot = (opcode == 33)
            is_inot = (opcode == 34)
            is_bfcast = (opcode == 35)
            # More trig (36-39)
            is_asin = (opcode == 36)
            is_acos = (opcode == 37)
            is_sinh = (opcode == 38)
            is_cosh = (opcode == 39)
            # More math (40-44)
            is_tanh = (opcode == 40)
            is_asinh = (opcode == 41)
            is_acosh = (opcode == 42)
            is_atanh = (opcode == 43)
            is_log = (opcode == 44)
            # More casts (46-51)
            is_ficast = (opcode == 46)
            is_floor = (opcode == 50)
            is_ceil = (opcode == 51)

            # Control flow
            is_phi = (opcode == 100)
            is_br = (opcode == 101)
            is_jmp = (opcode == 102)

            # Compute binary result (float and integer comparisons)
            # Using safe arithmetic to handle indeterminate forms (inf-inf, 0*inf, etc)
            binary_result = jnp.where(is_fadd, jax_safe_add(a, b),
                           jnp.where(is_fsub, jax_safe_sub(a, b),
                           jnp.where(is_fmul, jax_safe_mul(a, b),
                           jnp.where(is_fdiv, jax_safe_div(a, b),
                           jnp.where(is_flt, jnp.float64(a < b),
                           jnp.where(is_fle, jnp.float64(a <= b),
                           jnp.where(is_fgt, jnp.float64(a > b),
                           jnp.where(is_fge, jnp.float64(a >= b),
                           jnp.where(is_feq, jnp.float64(a == b),
                           jnp.where(is_fne, jnp.float64(a != b),
                           jnp.where(is_pow, jax_safe_pow(a, b),
                           jnp.where(is_min, jnp.minimum(a, b),
                           jnp.where(is_max, jnp.maximum(a, b),
                           jnp.where(is_atan2, jnp.arctan2(a, b),
                           # Integer comparisons
                           jnp.where(is_ilt, jnp.float64(a < b),
                           jnp.where(is_ile, jnp.float64(a <= b),
                           jnp.where(is_igt, jnp.float64(a > b),
                           jnp.where(is_ige, jnp.float64(a >= b),
                           jnp.where(is_ieq, jnp.float64(a == b),
                           jnp.where(is_ine, jnp.float64(a != b),
                           jnp.where(is_hypot, jnp.hypot(a, b),
                           0.0)))))))))))))))))))))

            # Compute unary result
            unary_result = jnp.where(is_fneg, -a,
                          jnp.where(is_ln, jax_safe_ln(a),
                          jnp.where(is_exp, jax_safe_exp(a),
                          jnp.where(is_sqrt, jax_safe_sqrt(a),
                          jnp.where(is_sin, jnp.sin(a),
                          jnp.where(is_cos, jnp.cos(a),
                          jnp.where(is_tan, jnp.tan(a),
                          jnp.where(is_atan, jnp.arctan(a),
                          jnp.where(is_abs, jnp.abs(a),
                          jnp.where(is_ifcast, jnp.float64(a),
                          jnp.where(is_optbarrier, a,
                          jnp.where(is_call, jax_safe_exp(jnp.minimum(a, 700.0)),
                          # Integer/boolean ops
                          jnp.where(is_ineg, -a,
                          jnp.where(is_bnot, jnp.float64(a == 0.0),
                          jnp.where(is_inot, jnp.float64(a == 0.0),
                          jnp.where(is_bfcast, jnp.float64(a),
                          # More trig
                          jnp.where(is_asin, jnp.arcsin(a),
                          jnp.where(is_acos, jnp.arccos(a),
                          jnp.where(is_sinh, jnp.sinh(a),
                          jnp.where(is_cosh, jnp.cosh(a),
                          jnp.where(is_tanh, jnp.tanh(a),
                          jnp.where(is_asinh, jnp.arcsinh(a),
                          jnp.where(is_acosh, jnp.arccosh(a),
                          jnp.where(is_atanh, jnp.arctanh(a),
                          jnp.where(is_log, jnp.log10(jnp.maximum(a, 1e-300)),
                          jnp.where(is_ficast, jnp.float64(jnp.int32(a)),
                          jnp.where(is_floor, jnp.floor(a),
                          jnp.where(is_ceil, jnp.ceil(a),
                          0.0))))))))))))))))))))))))))))

            # Classify opcode type for result selection
            is_binary = ((opcode <= 3) | ((opcode >= 19) & (opcode <= 22)) |
                        ((opcode >= 26) & (opcode <= 31)) | (opcode == 45))
            is_unary = ((opcode >= 10) & (opcode <= 18)) | (opcode == 23) | (opcode == 24) | (opcode == 25) | \
                       ((opcode >= 32) & (opcode <= 51))

            arith_result = jnp.where(is_binary, binary_result,
                           jnp.where(is_unary, unary_result, 0.0))

            # Handle PHI nodes
            # PHI: select value from the predecessor block that was reached
            # phi_array format: [n_ops, val1, pred1, val2, pred2, ...]
            phi_start = op1_idx  # reused for phi start index
            n_phi_ops = phi_array[phi_start]

            # Get first value as default
            first_val_idx = phi_array[phi_start + 1]
            first_pred_idx = phi_array[phi_start + 2]
            phi_result = vals[first_val_idx]

            # Select value from the predecessor that was actually reached
            # For each operand: if its predecessor was reached, use its value
            def select_phi_op(phi_result, op_num):
                val_idx = phi_array[phi_start + 1 + op_num * 2]
                pred_idx = phi_array[phi_start + 2 + op_num * 2]
                op_val = vals[val_idx]
                pred_reached = block_reached[pred_idx]
                # If this predecessor was reached, use its value
                return jnp.where(pred_reached > 0.5, op_val, phi_result)

            # Apply selection for each PHI operand (up to 4)
            # The last reached predecessor wins
            phi_result = jnp.where(block_reached[first_pred_idx] > 0.5, vals[first_val_idx], phi_result)
            phi_result = jnp.where(n_phi_ops > 1, select_phi_op(phi_result, 1), phi_result)
            phi_result = jnp.where(n_phi_ops > 2, select_phi_op(phi_result, 2), phi_result)
            phi_result = jnp.where(n_phi_ops > 3, select_phi_op(phi_result, 3), phi_result)

            # Handle BR and JMP - update block reachability
            # BR: if this block was reached AND condition is true -> true_block reached
            #     if this block was reached AND condition is false -> false_block reached
            # JMP: if this block was reached -> dest_block reached
            cond_val = vals[op1_idx]  # condition for br
            this_block_reached = block_reached[block_idx]

            # For BR: propagate reachability based on condition
            # For JMP: propagate reachability unconditionally (extra1 == extra2 == dest)
            new_block_reached = block_reached

            # Update true_block reachability (for BR when cond is true, for JMP always)
            true_block_reaches = jnp.where(
                is_br,
                this_block_reached * cond_val,  # BR: reached if this block reached AND cond true
                jnp.where(is_jmp, this_block_reached, 0.0)  # JMP: reached if this block reached
            )
            new_block_reached = jnp.where(
                is_br | is_jmp,
                new_block_reached.at[extra1].set(
                    jnp.maximum(new_block_reached[extra1], true_block_reaches)
                ),
                new_block_reached
            )

            # Update false_block reachability (for BR when cond is false)
            false_block_reaches = jnp.where(
                is_br,
                this_block_reached * (1.0 - cond_val),  # BR: reached if this block reached AND cond false
                0.0  # JMP doesn't have a false target
            )
            new_block_reached = jnp.where(
                is_br,
                new_block_reached.at[extra2].set(
                    jnp.maximum(new_block_reached[extra2], false_block_reaches)
                ),
                new_block_reached
            )

            # Choose final result
            final_val = jnp.where(is_phi, phi_result, arith_result)

            # Update vals only for non-control-flow instructions with valid result
            is_arith = (opcode < 100)
            should_update = is_arith | is_phi
            new_vals = jnp.where(should_update, vals.at[result_idx].set(final_val), vals)

            return (new_vals, new_block_reached)

        vals, _ = lax.fori_loop(0, n_instructions, exec_instruction, (vals, block_reached))

        # Extract outputs
        resist_res = vals[resist_res_indices]
        react_res = vals[react_res_indices]
        resist_jac = vals[resist_jac_indices]
        react_jac = vals[react_jac_indices]

        return (resist_res, react_res), (resist_jac, react_jac)

    return eval_fn


def build_eval_fn(module, force_lax_loop: bool = False) -> Tuple[Callable, Dict]:
    """Build JAX eval function from module, choosing appropriate strategy.

    Strategy selection:
    - straight_line: For simple models with no branches (fastest)
    - cfg_to_conditional: For medium models with branches (<1000 instructions)
    - lax_loop: For large models (>1000 instructions) - uses lax.fori_loop interpreter

    Args:
        module: openvaf_py module object
        force_lax_loop: If True, always use lax_loop mode (fast JIT, slower execution)

    Returns:
        (eval_fn, metadata) where eval_fn is JAX-traceable
    """
    analysis = analyze_mir(module)

    # Choose strategy based on complexity
    # lax_loop for large models (>1000 instructions) to avoid XLA compilation timeout
    n_instructions = len(analysis.instructions)
    use_lax_loop = force_lax_loop or n_instructions > 1000

    metadata = {
        'has_branches': analysis.has_branches,
        'num_branches': analysis.num_branches,
        'num_phi_nodes': analysis.num_phi_nodes,
        'num_instructions': n_instructions,
        'varying_dependent_branches': analysis.varying_dependent_branches,
        'fixed_only_branches': analysis.fixed_only_branches,
        'strategy': 'lax_loop' if use_lax_loop else (
            'straight_line' if not analysis.has_branches else 'cfg_to_conditional'
        ),
    }

    if use_lax_loop:
        # Large model - use lax.fori_loop interpreter for fast JIT
        eval_fn = generate_lax_loop_eval(module)
        return eval_fn, metadata
    elif not analysis.has_branches:
        # Simple case: no control flow
        eval_fn = generate_straight_line_eval(module)
        return eval_fn, metadata
    else:
        # Medium complexity: use CFG-to-conditional transformation
        eval_fn = generate_cfg_eval(module)
        return eval_fn, metadata


# Test
if __name__ == '__main__':
    import openvaf_py
    from pathlib import Path

    REPO_ROOT = Path(__file__).parent.parent
    VACASK = REPO_ROOT / "vendor" / "VACASK" / "devices"

    print("Testing JAX code generator...")
    print()

    # Test resistor (no control flow)
    print("=== Resistor ===")
    modules = openvaf_py.compile_va(str(VACASK / "resistor.va"))
    resistor = modules[0]

    analysis = analyze_mir(resistor)
    print(f"Branches: {analysis.num_branches}, PHI nodes: {analysis.num_phi_nodes}")

    eval_fn, meta = build_eval_fn(resistor)
    print(f"Strategy: {meta['strategy']}")

    # Test with dummy inputs
    import numpy as np
    params = jnp.zeros(10)
    cache = jnp.zeros(5)

    # JIT compile
    jit_eval = jax.jit(eval_fn)
    result = jit_eval(params, cache)
    print(f"Result shapes: res={result[0][0].shape}, jac={result[1][0].shape}")
    print()

    # Test capacitor (no control flow)
    print("=== Capacitor ===")
    modules = openvaf_py.compile_va(str(VACASK / "capacitor.va"))
    capacitor = modules[0]

    analysis = analyze_mir(capacitor)
    print(f"Branches: {analysis.num_branches}, PHI nodes: {analysis.num_phi_nodes}")

    eval_fn, meta = build_eval_fn(capacitor)
    print(f"Strategy: {meta['strategy']}")
    print()

    # Test diode (has control flow)
    print("=== Diode ===")
    modules = openvaf_py.compile_va(str(VACASK / "diode.va"))
    diode = modules[0]

    analysis = analyze_mir(diode)
    print(f"Branches: {analysis.num_branches}, PHI nodes: {analysis.num_phi_nodes}")
    print(f"Varying-dependent: {analysis.varying_dependent_branches}")
    print(f"Fixed-only: {analysis.fixed_only_branches}")

    try:
        eval_fn, meta = build_eval_fn(diode)
        print(f"Strategy: {meta['strategy']}")

        # Test with dummy inputs
        n_params = len(list(diode.param_names))
        n_cache = diode.num_cached_values
        params = jnp.zeros(n_params)
        cache = jnp.zeros(n_cache)

        # Try to JIT compile
        print("Attempting JIT compilation...")
        jit_eval = jax.jit(eval_fn)
        result = jit_eval(params, cache)
        print(f"Result shapes: res={result[0][0].shape}, jac={result[1][0].shape}")
        print("✅ Diode JIT compilation succeeded!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone!")
