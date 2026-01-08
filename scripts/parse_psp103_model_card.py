#!/usr/bin/env python3
"""Parse PSP103 model card from VACASK ring oscillator and extract all parameters."""

import re
from pathlib import Path

def parse_model_card(model_file, model_name):
    """Parse SPICE model card and return dict of parameters."""

    with open(model_file) as f:
        content = f.read()

    # Find the model definition
    pattern = rf"^model\s+{model_name}\s+\w+\s*\("
    match = re.search(pattern, content, re.MULTILINE)

    if not match:
        print(f"Model '{model_name}' not found!")
        return {}

    # Extract everything between the opening ( and closing )
    start = match.end()
    depth = 1
    end = start

    for i in range(start, len(content)):
        if content[i] == '(':
            depth += 1
        elif content[i] == ')':
            depth -= 1
            if depth == 0:
                end = i
                break

    model_body = content[start:end]

    # Parse parameters (format: param=value or param=value.value)
    params = {}
    for line in model_body.split('\n'):
        line = line.strip()
        if '=' in line and not line.startswith('//') and not line.startswith('#'):
            # Remove comments
            line = line.split('//')[0].split('#')[0].strip()

            # Parse param=value
            match = re.match(r'(\w+)\s*=\s*([+-]?[\d.eE+-]+)', line)
            if match:
                param_name = match.group(1)
                param_value = match.group(2)

                try:
                    # Convert to float
                    value = float(param_value)
                    params[param_name] = value
                except ValueError:
                    pass

    return params

def main():
    model_file = Path(__file__).parent.parent.parent / "VACASK" / "benchmark" / "ring" / "vacask" / "models.inc"

    if not model_file.exists():
        print(f"Model file not found: {model_file}")
        return 1

    print("="*80)
    print("Parsing PSP103 Model Cards")
    print("="*80)
    print()

    # Parse NMOS
    nmos_params = parse_model_card(model_file, "psp103n")
    print(f"NMOS (psp103n): {len(nmos_params)} parameters")

    # Parse PMOS
    pmos_params = parse_model_card(model_file, "psp103p")
    print(f"PMOS (psp103p): {len(pmos_params)} parameters")
    print()

    # Show first few PMOS parameters
    print("First 10 PMOS parameters:")
    for i, (k, v) in enumerate(list(pmos_params.items())[:10]):
        print(f"  {k} = {v}")
    print()

    # Save to file for use in tests
    import json
    output = {
        'nmos': nmos_params,
        'pmos': pmos_params,
    }

    output_file = Path(__file__).parent / 'psp103_model_params.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved parameters to: {output_file}")
    print()
    print(f"Total NMOS params: {len(nmos_params)}")
    print(f"Total PMOS params: {len(pmos_params)}")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
