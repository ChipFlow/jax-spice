"""JAX code generator - backward compatibility stub.

This module has been archived. The MIR interpreter approach is being replaced
by a DAE-aware code generator (jax_emit.py).

For now, this stub re-exports from archive for backward compatibility.
"""

# Re-export from archive for backward compatibility
from archive.jax_codegen_mir_interpreter import (
    PREALLOCATED_CONSTANTS,
    jax_safe_add,
    jax_safe_sub,
    jax_safe_mul,
    jax_safe_div,
    jax_safe_ln,
    jax_safe_exp,
    jax_safe_sqrt,
    jax_safe_pow,
    JAX_BINARY_OPS,
    JAX_UNARY_OPS,
    MIRAnalysis,
    analyze_mir,
    generate_straight_line_eval,
    generate_cfg_eval,
    generate_lax_loop_eval,
    build_eval_fn,
)

__all__ = [
    'PREALLOCATED_CONSTANTS',
    'jax_safe_add',
    'jax_safe_sub',
    'jax_safe_mul',
    'jax_safe_div',
    'jax_safe_ln',
    'jax_safe_exp',
    'jax_safe_sqrt',
    'jax_safe_pow',
    'JAX_BINARY_OPS',
    'JAX_UNARY_OPS',
    'MIRAnalysis',
    'analyze_mir',
    'generate_straight_line_eval',
    'generate_cfg_eval',
    'generate_lax_loop_eval',
    'build_eval_fn',
]
