"""Python code generation from MIR."""

from .mir_parser import MIRFunction, MIRInstruction, MIRValue
from typing import Dict, Set


def generate_python_eval(mir_func: MIRFunction, param_map: Dict[str, str]) -> str:
    """Generate Python code from MIR evaluation function.

    Args:
        mir_func: Parsed MIR function
        param_map: Maps MIR parameter names (v16, v17) to semantic names (V_br, r)

    Returns:
        Python function source code as string
    """
    # Build function signature
    sig_params = [param_map.get(p.name, p.name) for p in mir_func.params]
    func_def = f"def eval_device({', '.join(sig_params)}):\n"
    func_def += '    """Generated from MIR - evaluates device model."""\n'

    # Generate body - translate each instruction
    body_lines = []

    for block in mir_func.blocks:
        # Add block comment
        body_lines.append(f"    # {block.name}")

        for inst in block.instructions:
            # Skip optbarrier (optimization hints, not needed in Python)
            if inst.opcode == 'optbarrier':
                continue

            # Generate Python statement
            python_stmt = translate_instruction(inst, param_map)
            if python_stmt:
                body_lines.append(f"    {python_stmt}")

    # Add return statement - return all computed values for now
    # TODO: Filter to only residuals and Jacobian entries
    body_lines.append("\n    # Return computed values")
    body_lines.append("    return {")

    # Find all computed values (not parameters)
    computed_vars = set()
    for block in mir_func.blocks:
        for inst in block.instructions:
            if inst.result and inst.opcode != 'optbarrier':
                var_name = param_map.get(inst.result.name, inst.result.name)
                computed_vars.add(var_name)

    for var_name in sorted(computed_vars):
        body_lines.append(f"        '{var_name}': {var_name},")

    body_lines.append("    }")

    return func_def + '\n'.join(body_lines)


def translate_instruction(inst: MIRInstruction, param_map: Dict[str, str]) -> str:
    """Translate a single MIR instruction to Python.

    Args:
        inst: MIR instruction
        param_map: Parameter name mapping

    Returns:
        Python statement as string
    """
    if not inst.result:
        return f"# {inst.opcode}()"  # Void instruction (callback, etc.)

    # Apply semantic mapping to result too
    result = param_map.get(inst.result.name, inst.result.name)
    opcode = inst.opcode
    args = [param_map.get(a.name, a.name) for a in inst.args]

    # Map MIR opcodes to Python operations
    if opcode == 'fdiv':
        return f"{result} = {args[0]} / {args[1]}"
    elif opcode == 'fmul':
        return f"{result} = {args[0]} * {args[1]}"
    elif opcode == 'fadd':
        return f"{result} = {args[0]} + {args[1]}"
    elif opcode == 'fsub':
        return f"{result} = {args[0]} - {args[1]}"
    elif opcode == 'fneg':
        return f"{result} = -{args[0]}"
    elif opcode == 'pow':
        return f"{result} = {args[0]} ** {args[1]}"
    elif opcode == 'sqrt':
        return f"{result} = math.sqrt({args[0]})"
    elif opcode == 'exp':
        return f"{result} = math.exp({args[0]})"
    elif opcode == 'ln':
        return f"{result} = math.log({args[0]})"
    elif opcode == 'sin':
        return f"{result} = math.sin({args[0]})"
    elif opcode == 'cos':
        return f"{result} = math.cos({args[0]})"
    elif opcode == 'tan':
        return f"{result} = math.tan({args[0]})"
    elif opcode == 'phi':
        # PHI nodes for control flow - for now, just use first arg
        # TODO: Handle properly with conditional logic
        return f"{result} = {args[0]}  # PHI node"
    elif opcode == 'optbarrier':
        # Optimization barrier - pass-through in Python
        return f"{result} = {args[0]}  # optbarrier"
    else:
        return f"# Unsupported opcode: {opcode}"


def analyze_mir_semantics(mir_func: MIRFunction) -> Dict[str, str]:
    """Analyze MIR to infer semantic meaning of parameters.

    For the resistor example:
        v16 = V(br)     - branch voltage
        v17 = r         - resistance parameter
        v18 = I         - computed current

    Returns mapping of MIR names to semantic names.
    """
    # This is device-specific. For now, hardcode resistor mapping
    # TODO: Make this generic by analyzing DAE system metadata
    return {
        'v16': 'V_br',      # Branch voltage
        'v17': 'r',         # Resistance
        'v51': 'mfactor',   # Multiplicity factor (typically v51 in MIR)
        'v18': 'I',         # Current = V/R
        'v55': 'I_neg',     # -I for Jacobian
        'v65': 'I_scaled',  # mfactor * I
        'v67': 'dI_neg',    # mfactor * (-I)
    }
