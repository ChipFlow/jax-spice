#!/usr/bin/env -S uv run --script
"""MIR-based Python code generator for Verilog-A models.

This module generates Python functions from OpenVAF MIR (Mid-level IR).
It supports generating both init (setup_instance) and eval functions.

Usage:
    import openvaf_py
    from mir_codegen import generate_init_function, generate_eval_function

    modules = openvaf_py.compile_va('model.va')
    module = modules[0]

    init_fn = generate_init_function(module)
    eval_fn = generate_eval_function(module)
"""

import math
from typing import Any, Callable


# MIR opcode implementations
def mir_fadd(a: float, b: float) -> float:
    return a + b

def mir_fsub(a: float, b: float) -> float:
    return a - b

def mir_fmul(a: float, b: float) -> float:
    return a * b

def mir_fdiv(a: float, b: float) -> float:
    return a / b if b != 0 else math.inf if a > 0 else -math.inf if a < 0 else math.nan

def mir_fneg(a: float) -> float:
    return -a

def mir_flt(a: float, b: float) -> bool:
    return a < b

def mir_fle(a: float, b: float) -> bool:
    return a <= b

def mir_fgt(a: float, b: float) -> bool:
    return a > b

def mir_fge(a: float, b: float) -> bool:
    return a >= b

def mir_feq(a: float, b: float) -> bool:
    return a == b

def mir_fne(a: float, b: float) -> bool:
    return a != b

def mir_pow(base: float, exp: float) -> float:
    return math.pow(base, exp)

def mir_ln(x: float) -> float:
    return math.log(x) if x > 0 else -math.inf

def mir_exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf

def mir_sqrt(x: float) -> float:
    return math.sqrt(x) if x >= 0 else math.nan

def mir_sin(x: float) -> float:
    return math.sin(x)

def mir_cos(x: float) -> float:
    return math.cos(x)

def mir_tan(x: float) -> float:
    return math.tan(x)

def mir_atan(x: float) -> float:
    return math.atan(x)

def mir_atan2(y: float, x: float) -> float:
    return math.atan2(y, x)

def mir_abs(x: float) -> float:
    return abs(x)

def mir_min(a: float, b: float) -> float:
    return min(a, b)

def mir_max(a: float, b: float) -> float:
    return max(a, b)

def mir_ifcast(x: int) -> float:
    """Cast integer to float."""
    return float(x)

def mir_optbarrier(x: Any) -> Any:
    """Optimization barrier - just passes through the value."""
    return x


# Map MIR opcodes to Python implementations
OPCODE_MAP = {
    'fadd': mir_fadd,
    'fsub': mir_fsub,
    'fmul': mir_fmul,
    'fdiv': mir_fdiv,
    'fneg': mir_fneg,
    'flt': mir_flt,
    'fle': mir_fle,
    'fgt': mir_fgt,
    'fge': mir_fge,
    'feq': mir_feq,
    'fne': mir_fne,
    'pow': mir_pow,
    'ln': mir_ln,
    'exp': mir_exp,
    'sqrt': mir_sqrt,
    'sin': mir_sin,
    'cos': mir_cos,
    'tan': mir_tan,
    'atan': mir_atan,
    'atan2': mir_atan2,
    'abs': mir_abs,
    'min': mir_min,
    'max': mir_max,
    'ifcast': mir_ifcast,
    'optbarrier': mir_optbarrier,
}


class MIRInterpreter:
    """Interprets MIR instructions to compute results."""

    def __init__(self, constants: dict, params: list):
        """Initialize interpreter.

        Args:
            constants: Dict mapping variable names to constant values
            params: List of parameter variable names (in order)
        """
        self.values = dict(constants)  # Copy constants
        self.param_names = params

    def set_params(self, param_values: list):
        """Set parameter values.

        Args:
            param_values: List of values (same order as params list)
        """
        for name, value in zip(self.param_names, param_values):
            self.values[name] = value

    def get(self, var: str) -> Any:
        """Get variable value."""
        return self.values.get(var)

    def set(self, var: str, value: Any):
        """Set variable value."""
        self.values[var] = value

    def execute_instruction(self, instr: dict) -> Any:
        """Execute a single MIR instruction.

        Args:
            instr: Dict with 'opcode', 'result', 'operands'

        Returns:
            The computed result value
        """
        opcode = instr['opcode']
        operands = instr.get('operands', [])
        result_var = instr.get('result')

        # Resolve operand values
        operand_values = [self.get(op) for op in operands]

        if opcode == 'phi':
            # PHI nodes handled separately in control flow
            return None
        elif opcode == 'jmp' or opcode == 'br':
            # Control flow handled separately
            return None
        elif opcode == 'call':
            # Handle special function calls
            func_name = operands[0] if operands else ''
            if func_name == 'limexp':
                # Limited exponential
                x = operand_values[1] if len(operand_values) > 1 else 0
                return mir_exp(min(x, 700))  # Prevent overflow
            elif func_name == 'simparam':
                # Simulation parameter - return default or lookup
                return operand_values[1] if len(operand_values) > 1 else 0.0
            else:
                return 0.0  # Unknown function
        elif opcode in OPCODE_MAP:
            func = OPCODE_MAP[opcode]
            try:
                result = func(*operand_values)
            except Exception:
                result = math.nan
            if result_var:
                self.set(result_var, result)
            return result
        else:
            # Unknown opcode - treat as identity if single operand
            if operand_values:
                result = operand_values[0]
                if result_var:
                    self.set(result_var, result)
                return result
            return None

    def execute_block(self, instructions: list, blocks: dict,
                      start_block: str = None) -> dict:
        """Execute instructions with control flow.

        Args:
            instructions: List of all instructions
            blocks: Dict mapping block names to {predecessors, successors}
            start_block: Optional starting block name

        Returns:
            Dict of final variable values
        """
        # Group instructions by block
        block_instrs = {}
        for instr in instructions:
            block = instr.get('block', 'default')
            if block not in block_instrs:
                block_instrs[block] = []
            block_instrs[block].append(instr)

        # Find entry block
        if start_block:
            current_block = start_block
        else:
            # Find block with no predecessors
            for name, info in blocks.items():
                if not info.get('predecessors'):
                    current_block = name
                    break
            else:
                current_block = list(blocks.keys())[0] if blocks else 'default'

        prev_block = None
        max_iterations = 10000

        for _ in range(max_iterations):
            if current_block is None:
                break

            instrs = block_instrs.get(current_block, [])
            next_block = None

            for instr in instrs:
                opcode = instr['opcode']

                if opcode == 'phi':
                    # PHI node: select value based on predecessor
                    # Format: phi_operands: [{value: 'v36', block: 'block10'}, ...]
                    phi_operands = instr.get('phi_operands', [])
                    result_var = instr.get('result')
                    for phi_op in phi_operands:
                        if phi_op.get('block') == prev_block:
                            value_var = phi_op.get('value')
                            self.set(result_var, self.get(value_var))
                            break
                    else:
                        # Use first value as default
                        if phi_operands:
                            value_var = phi_operands[0].get('value')
                            self.set(result_var, self.get(value_var))

                elif opcode == 'jmp':
                    # Unconditional jump: destination field
                    next_block = instr.get('destination')

                elif opcode == 'br':
                    # Conditional branch: condition, true_block, false_block
                    cond_var = instr.get('condition')
                    true_block = instr.get('true_block')
                    false_block = instr.get('false_block')
                    cond = self.get(cond_var)
                    next_block = true_block if cond else false_block

                else:
                    self.execute_instruction(instr)

            # Check if we've reached terminal block (no successors)
            block_info = blocks.get(current_block, {})
            if not next_block and not block_info.get('successors'):
                break

            prev_block = current_block
            current_block = next_block

        return self.values


def generate_init_function(module) -> Callable:
    """Generate init function from module MIR.

    Args:
        module: openvaf_py module object

    Returns:
        Callable init function that takes parameter dict and returns cache list
    """
    # Get MIR data
    init_mir = module.get_init_mir_instructions()
    metadata = module.get_codegen_metadata()

    constants = init_mir['constants']
    params = init_mir.get('params', [])
    instructions = init_mir['instructions']
    blocks = init_mir['blocks']
    cache_mapping = init_mir['cache_mapping']
    init_param_mapping = metadata['init_param_mapping']

    # Invert param mapping: semantic name -> MIR var
    # (the metadata provides semantic -> MIR)
    param_to_mir = init_param_mapping

    def init_function(**input_params) -> list:
        """Compute cache values for model.

        Args:
            **input_params: Parameter values by semantic name

        Returns:
            List of cache values
        """
        # Create interpreter
        interp = MIRInterpreter(constants, params)

        # Map input params to MIR variables
        for name, value in input_params.items():
            mir_var = param_to_mir.get(name)
            if mir_var:
                interp.set(mir_var, value)

        # Execute MIR
        interp.execute_block(instructions, blocks)

        # Extract cache values
        cache = []
        for cm in cache_mapping:
            init_val = cm['init_value']
            cache.append(interp.get(init_val))

        return cache

    return init_function


def generate_eval_function(module) -> Callable:
    """Generate eval function from module MIR.

    Args:
        module: openvaf_py module object

    Returns:
        Callable eval function
    """
    # Get MIR data
    eval_mir = module.get_mir_instructions()
    metadata = module.get_codegen_metadata()

    constants = eval_mir['constants']
    params = eval_mir.get('params', [])
    instructions = eval_mir['instructions']
    blocks = eval_mir['blocks']

    eval_param_mapping = metadata['eval_param_mapping']
    cache_info = metadata['cache_info']
    residuals = metadata['residuals']
    jacobian = metadata['jacobian']

    # Build cache param mapping: cache_idx -> eval param var
    cache_to_param = {}
    for ci in cache_info:
        # eval_param might be 'v5' meaning param index 5
        eval_param = ci['eval_param']
        if eval_param.startswith('v'):
            param_idx = int(eval_param[1:])
            if param_idx < len(params):
                cache_to_param[ci['cache_idx']] = params[param_idx]

    def eval_function(cache: list, **input_params) -> dict:
        """Evaluate model.

        Args:
            cache: Cache values from init function
            **input_params: Parameter values (voltages, mfactor, etc.)

        Returns:
            Dict with residuals and jacobian entries
        """
        # Create interpreter
        interp = MIRInterpreter(constants, params)

        # Map input params to MIR variables
        for name, value in input_params.items():
            mir_var = eval_param_mapping.get(name)
            if mir_var:
                interp.set(mir_var, value)

        # Map cache values to MIR variables
        for idx, value in enumerate(cache):
            mir_var = cache_to_param.get(idx)
            if mir_var:
                interp.set(mir_var, value)

        # Execute MIR
        interp.execute_block(instructions, blocks)

        # Extract results
        result = {
            'residuals_resist': [],
            'residuals_react': [],
            'jacobian_resist': [],
            'jacobian_react': [],
        }

        for r in residuals:
            result['residuals_resist'].append(interp.get(r['resist_var']) or 0.0)
            result['residuals_react'].append(interp.get(r['react_var']) or 0.0)

        for j in jacobian:
            result['jacobian_resist'].append(interp.get(j['resist_var']) or 0.0)
            result['jacobian_react'].append(interp.get(j['react_var']) or 0.0)

        return result

    return eval_function


# Test
if __name__ == '__main__':
    import openvaf_py
    from pathlib import Path

    REPO_ROOT = Path(__file__).parent.parent
    VACASK = REPO_ROOT / "vendor" / "VACASK" / "devices"

    print("Testing MIR code generator with capacitor model...")

    modules = openvaf_py.compile_va(str(VACASK / "capacitor.va"))
    cap = modules[0]

    # Generate functions
    init_fn = generate_init_function(cap)
    eval_fn = generate_eval_function(cap)

    # Test init
    print("\n=== INIT TEST ===")
    cache = init_fn(c=1e-9, mfactor=1.0, c_given=True)
    print(f"init(c=1nF, m=1): cache = {cache}")

    cache2 = init_fn(c=2e-9, mfactor=2.0, c_given=True)
    print(f"init(c=2nF, m=2): cache = {cache2}")

    # Test eval
    print("\n=== EVAL TEST ===")
    # For capacitor, we need c, voltage, mfactor
    # (cache values are for jacobian, residual is computed from c directly)
    result = eval_fn(cache, c=1e-9, **{"V(A,B)": 1.0, "mfactor": 1.0})
    print(f"eval(c=1nF, V=1.0, m=1):")
    print(f"  residuals_react: {result['residuals_react']}")
    print(f"  jacobian_react: {result['jacobian_react']}")

    # Compare with expected: Q = mfactor * c * V = 1e-9 * 1.0 = 1e-9
    print(f"  expected residuals: [1e-09, -1e-09]")
    print(f"  expected jacobian: [1e-09, -1e-09, -1e-09, 1e-09]")

    print("\nAll tests completed!")
