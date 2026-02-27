"""Safe expression evaluator for SPICE parameter expressions.

Uses simpleeval to safely evaluate arithmetic expressions without
allowing arbitrary code execution. Supports SPICE SI suffixes and
common math functions.

Example:
    >>> from vajax.utils.safe_eval import safe_eval_expr
    >>> safe_eval_expr("2*wmin + 1u", {"wmin": 1e-6})
    3e-06
"""

import math
import re
from typing import Dict, List, Tuple

from simpleeval import EvalWithCompoundTypes, InvalidExpression

# SPICE SI suffixes (order matters - check longer suffixes first)
SI_SUFFIXES = [
    ("meg", 1e6),
    ("mil", 25.4e-6),
    # Time units (must come before single-letter suffixes)
    ("ms", 1e-3),
    ("us", 1e-6),
    ("ns", 1e-9),
    ("ps", 1e-12),
    ("fs", 1e-15),
    # Voltage/current units
    ("mv", 1e-3),
    ("uv", 1e-6),
    ("nv", 1e-9),
    ("ma", 1e-3),
    ("ua", 1e-6),
    ("na", 1e-9),
    ("pa", 1e-12),
    ("fa", 1e-15),
    # Standard SI prefixes
    ("g", 1e9),
    ("t", 1e12),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
    ("a", 1e-18),
]

# Pre-compiled SI suffix regex patterns (avoids re-compilation on every call)
_COMPILED_SI_PATTERNS: List[Tuple[re.Pattern, str]] = []
for _suffix, _mult in SI_SUFFIXES:
    _pattern = re.compile(
        rf"(\d+\.?\d*|\.\d+)({_suffix})(?![a-zA-Z0-9])",
        re.IGNORECASE,
    )
    _COMPILED_SI_PATTERNS.append((_pattern, rf"\g<1>*{_mult}"))

# Safe math functions available in expressions
SAFE_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "abs": abs,
    "pow": pow,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
}

# Module-level caches for hot-path performance
_SPICE_NUMBER_CACHE: Dict[str, Tuple[float, bool]] = {}
_EVAL_CACHE: Dict[Tuple[str, frozenset], float] = {}


def parse_spice_number(s: str) -> tuple[float, bool]:
    """Parse a SPICE number with optional SI suffix.

    Args:
        s: String like "1u", "100n", "1.5meg", "2.5e-6"

    Returns:
        Tuple of (value, success). If parsing fails, returns (0.0, False).
    """
    cached = _SPICE_NUMBER_CACHE.get(s)
    if cached is not None:
        return cached

    s_norm = s.strip().lower()
    if not s_norm:
        result = (0.0, False)
        _SPICE_NUMBER_CACHE[s] = result
        return result

    # Try direct float parse first
    try:
        result = (float(s_norm), True)
        _SPICE_NUMBER_CACHE[s] = result
        return result
    except ValueError:
        pass

    # Try with SI suffix
    for suffix, mult in SI_SUFFIXES:
        if s_norm.endswith(suffix):
            try:
                result = (float(s_norm[: -len(suffix)]) * mult, True)
                _SPICE_NUMBER_CACHE[s] = result
                return result
            except ValueError:
                pass

    result = (0.0, False)
    _SPICE_NUMBER_CACHE[s] = result
    return result


def _substitute_params(expr: str, params: Dict[str, float]) -> str:
    """Substitute parameter names with their values in an expression.

    Substitutes longer names first to avoid partial replacements
    (e.g., 'wmin' before 'w').
    """
    if not params:
        return expr
    result = expr
    for name, value in sorted(params.items(), key=lambda x: -len(x[0])):
        # Use word boundaries to avoid partial matches
        # But SPICE params can contain underscores, so we do simple replace
        result = result.replace(name, str(value))
    return result


def _expand_si_suffixes(expr: str) -> str:
    """Expand SI suffixes in an expression to their numeric values.

    Converts "1u" to "1e-6", "100n" to "100e-9", etc.
    Uses pre-compiled regex patterns for performance.
    """
    for pattern, replacement in _COMPILED_SI_PATTERNS:
        expr = pattern.sub(replacement, expr)
    return expr


# Create a reusable evaluator instance
_evaluator = EvalWithCompoundTypes(functions=SAFE_FUNCTIONS)


def safe_eval_expr(expr: str, params: Dict[str, float], default: float = 0.0) -> float:
    """Safely evaluate a SPICE parameter expression.

    Supports:
    - Arithmetic operators: +, -, *, /, **, ()
    - Math functions: sin, cos, exp, log, sqrt, abs, etc.
    - SPICE SI suffixes: u, n, p, f, k, meg, etc.
    - Parameter substitution from the params dict

    Args:
        expr: Expression string like "2*wmin + 1u" or "sin(2*pi*f)"
        params: Dict of parameter names to values for substitution
        default: Value to return if evaluation fails

    Returns:
        Evaluated float value, or default if evaluation fails

    Example:
        >>> safe_eval_expr("2*w + l", {"w": 1e-6, "l": 0.5e-6})
        2.5e-06
        >>> safe_eval_expr("1u + 100n", {})
        1.1e-06
    """
    if not isinstance(expr, str):
        try:
            return float(expr)
        except (ValueError, TypeError):
            return default

    expr = expr.strip()
    if not expr:
        return default

    # Try direct SPICE number parse first (most common case)
    val, success = parse_spice_number(expr)
    if success:
        return val

    # Cache lookup: (expr, params) -> result
    # Use frozenset for small param dicts; skip for large ones (expensive to hash)
    cache_key = None
    if len(params) < 20:
        cache_key = (expr, frozenset(params.items()))
        cached = _EVAL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    try:
        # Substitute parameters
        eval_expr = _substitute_params(expr, params)

        # Expand SI suffixes
        eval_expr = _expand_si_suffixes(eval_expr)

        # Evaluate safely
        result = _evaluator.eval(eval_expr)
        result = float(result)

        if cache_key is not None:
            _EVAL_CACHE[cache_key] = result

        return result

    except (InvalidExpression, ValueError, TypeError, KeyError, SyntaxError):
        return default
    except Exception:
        # Catch any other unexpected errors from simpleeval
        return default
