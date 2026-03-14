"""AST transformer: rewrite generated JAX Python source to MLX.

This module has been moved to openvaf_jax.mlx_transform.
This file re-exports for backward compatibility.
"""

from openvaf_jax.mlx_transform import (  # noqa: F401
    JaxToMlxTransformer,
    get_mlx_exec_namespace,
    jax_to_mlx,
    mlx_select,
    mlx_while_loop,
)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    print(jax_to_mlx(source))
