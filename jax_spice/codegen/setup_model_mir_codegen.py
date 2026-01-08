"""Generate setup_model() function from MIR model_param_setup."""

from .mir_parser import MIRFunction
from .control_flow_codegen import generate_python_with_control_flow
from typing import Dict


def generate_setup_model_from_mir(mir_func: MIRFunction, param_map: Dict[str, str], model_name: str = "device") -> str:
    """Generate setup_model() function from model_param_setup MIR.

    The setup_model() function validates parameters and applies defaults.
    It's called once per model type (not per instance).

    Args:
        mir_func: Parsed model_param_setup MIR function
        param_map: Maps MIR SSA names to semantic parameter names
        model_name: Name of the device model (for function naming)

    Returns:
        Python source code for setup_model() function
    """
    # Use the control flow codegen to generate the core logic
    core_code = generate_python_with_control_flow(mir_func, param_map)

    # Wrap it in a setup_model() function with proper interface
    lines = []
    lines.append(f'def setup_model_{model_name}(**params):')
    lines.append(f'    """Setup and validate model parameters for {model_name}."""')
    lines.append('')
    lines.append('    # Extract parameters and given flags')

    # Build parameter extraction code
    # The MIR function signature tells us which parameters are expected
    for param in mir_func.params:
        param_name = param_map.get(param.name) or param.name

        # Distinguish between value and given flag
        if param_name.endswith('_given'):
            base_name = param_name[:-6]  # Remove '_given' suffix
            lines.append(f'    {param_name} = params.get("{base_name}_given", False)')
        else:
            lines.append(f'    {param_name} = params.get("{param_name}", 0.0)')

    lines.append('')
    lines.append('    # Run validation and default logic from MIR')

    # Extract the function body from core_code (skip the def line and final return)
    core_lines = core_code.split('\n')
    in_final_return = False
    for line in core_lines:
        # Skip def line
        if line.strip().startswith('def '):
            continue

        # Detect start of final return statement
        if '# Collect results' in line or (line.strip().startswith('return {') and not in_final_return):
            in_final_return = True
            continue

        # Skip lines in final return
        if in_final_return:
            if line.strip() == '}':
                in_final_return = False
            continue

        # Add the line
        if line.strip():
            lines.append(line)

    lines.append('')
    lines.append('    # Return validated parameters (use *_final values computed by MIR)')
    lines.append('    return {')

    # For each parameter, return its _final version if available, otherwise the input
    for param in mir_func.params:
        param_name = param_map.get(param.name) or param.name
        if not param_name.endswith('_given'):
            # Check if there's a *_final version
            final_name = param_name + '_final'
            if final_name in param_map.values():
                lines.append(f'        "{param_name}": {final_name},')
            else:
                lines.append(f'        "{param_name}": {param_name},')

    lines.append('    }')

    return '\n'.join(lines)
