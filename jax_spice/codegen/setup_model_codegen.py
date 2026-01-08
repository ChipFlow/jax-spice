"""Generate setup_model() function from parameter metadata."""

from typing import Dict, List, Tuple, Optional


def generate_setup_model(
    params: List[Dict],
    model_name: str = "device"
) -> str:
    """Generate setup_model() function for parameter validation.

    Args:
        params: List of parameter definitions with keys:
            - name: Parameter name (e.g., "r")
            - type: "real" or "integer"
            - default: Default value
            - range: Optional tuple (min, max) or None
            - description: Optional description
        model_name: Device model name

    Returns:
        Python function source code
    """
    func_lines = []

    # Build signature - each param has value + given flag
    sig_params = []
    for p in params:
        sig_params.append(f"{p['name']}")
        sig_params.append(f"{p['name']}_given")

    func_lines.append(f"def setup_model({', '.join(sig_params)}):")
    func_lines.append(f'    """Setup and validate model parameters for {model_name}.')
    func_lines.append('    ')
    func_lines.append('    For each parameter:')
    func_lines.append('    - If given, validate range')
    func_lines.append('    - If not given, apply default value')
    func_lines.append('    """')
    func_lines.append('    errors = []')
    func_lines.append('')

    # Generate validation logic for each parameter
    for p in params:
        name = p['name']
        default = p['default']
        param_range = p.get('range')

        func_lines.append(f"    # Parameter: {name}")
        func_lines.append(f"    if {name}_given:")

        # Add range validation if present
        if param_range:
            min_val, max_val = param_range
            min_str = str(min_val) if min_val != float('-inf') else "float('-inf')"
            max_str = str(max_val) if max_val != float('inf') else "float('inf')"

            func_lines.append(f"        if not ({min_str} <= {name} <= {max_str}):")
            func_lines.append(f"            errors.append(f'{name} = {{{name}}} is out of range [{min_str}, {max_str}]')")

        func_lines.append(f"    else:")
        func_lines.append(f"        {name} = {default}  # Apply default")
        func_lines.append('')

    # Check for errors and return
    func_lines.append('    if errors:')
    func_lines.append('        raise ValueError("Parameter validation failed:\\n" + "\\n".join(errors))')
    func_lines.append('')
    func_lines.append('    return {')
    for p in params:
        func_lines.append(f"        '{p['name']}': {p['name']},")
    func_lines.append('    }')

    return '\n'.join(func_lines)


def extract_param_info_from_va(va_source: str) -> List[Dict]:
    """Extract parameter metadata from Verilog-A source.

    Parses lines like:
        parameter real r = 1 from [0:inf];
        parameter integer has_noise = 1;

    Returns list of parameter dicts.
    """
    import re
    params = []

    # Pattern: parameter <type> <name> = <default> [from [min:max]];
    param_pattern = r'parameter\s+(real|integer)\s+(\w+)\s*=\s*([^\s;]+)(?:\s+from\s+\[([^:]+):([^\]]+)\])?'

    for line in va_source.split('\n'):
        match = re.search(param_pattern, line)
        if match:
            param_type, name, default_str, min_str, max_str = match.groups()

            # Parse default value
            if param_type == 'integer':
                default = int(default_str)
            else:
                default = float(default_str)

            # Parse range if present
            param_range = None
            if min_str and max_str:
                min_val = float('-inf') if min_str == '-inf' else float(min_str)
                max_val = float('inf') if max_str == 'inf' else float(max_str)
                param_range = (min_val, max_val)

            params.append({
                'name': name,
                'type': param_type,
                'default': default,
                'range': param_range
            })

    return params
