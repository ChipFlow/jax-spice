"""Function caching utilities for OpenVAF to JAX/MLX translation.

This module provides caching for:
- Compiled Python functions (by code hash)
- vmapped+jit'd functions (by code hash + in_axes)
- MLX-transformed functions (by code hash + "mlx" suffix)
- Persistent code cache (stores generated Python code to disk)
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union, cast, overload

logger = logging.getLogger("vajax.openvaf")

# Default persistent cache directory
_PERSISTENT_CACHE_DIR: Optional[Path] = None

# Module-level cache for exec'd functions (keyed by code hash)
# This allows JAX to reuse JIT-compiled functions across translator instances
_exec_fn_cache: Dict[str, Callable] = {}

# Module-level cache for vmapped+jit'd functions (keyed by (code_hash, in_axes))
# This avoids repeated JIT compilation for the same function with same vmap axes
_vmapped_jit_cache: Dict[Tuple[str, Tuple], Callable] = {}

# Module-level cache for MLX exec'd functions (keyed by code hash + "mlx")
_mlx_exec_fn_cache: Dict[str, Callable] = {}

# Module-level cache for MLX vmapped+compiled functions
_mlx_vmapped_cache: Dict[Tuple[str, Tuple], Callable] = {}

# Lazy MLX availability check
_mlx_available: Optional[bool] = None


def is_mlx_available() -> bool:
    """Check if MLX is available (cached after first check)."""
    global _mlx_available
    if _mlx_available is None:
        try:
            import mlx.core  # noqa: F401
            _mlx_available = True
        except ImportError:
            _mlx_available = False
    return _mlx_available


@overload
def exec_with_cache(code: str, fn_name: str, return_hash: Literal[False] = ...) -> Callable: ...


@overload
def exec_with_cache(
    code: str, fn_name: str, return_hash: Literal[True]
) -> Tuple[Callable, str]: ...


def exec_with_cache(
    code: str, fn_name: str, return_hash: bool = False
) -> Union[Callable, Tuple[Callable, str]]:
    """Execute code and cache the resulting function by code hash.

    This dramatically speeds up repeated compilations (e.g., re-parsing the same
    circuit) by reusing previously exec'd functions. JAX can then reuse its
    JIT-compiled versions.

    Args:
        code: Python source code to execute
        fn_name: Name of function to extract from executed code
        return_hash: If True, return (function, code_hash) tuple

    Returns:
        If return_hash=False: Just the function
        If return_hash=True: Tuple of (function, code_hash)
    """
    import jax.numpy as jnp
    from jax import lax

    code_hash = hashlib.sha256(code.encode()).hexdigest()

    if code_hash in _exec_fn_cache:
        logger.debug(f"    {fn_name}: using cached function (hash={code_hash[:8]})")
        fn = _exec_fn_cache[code_hash]
        return (fn, code_hash) if return_hash else fn

    local_ns = {"jnp": jnp, "lax": lax}
    exec(code, local_ns)
    fn = cast(Callable, local_ns[fn_name])

    _exec_fn_cache[code_hash] = fn
    logger.debug(f"    {fn_name}: cached new function (hash={code_hash[:8]})")
    return (fn, code_hash) if return_hash else fn


def get_vmapped_jit(code_hash: str, fn: Callable, in_axes: Tuple) -> Callable:
    """Get a cached vmapped+jit'd version of a function.

    This caches the entire jax.jit(jax.vmap(fn, in_axes=in_axes)) result,
    avoiding repeated JIT compilation for the same function.

    Args:
        code_hash: Hash of the function's source code
        fn: The function to vmap and jit
        in_axes: vmap in_axes specification

    Returns:
        vmapped and jit'd function
    """
    import jax

    cache_key = (code_hash, in_axes)

    if cache_key in _vmapped_jit_cache:
        logger.debug(f"    vmapped_jit: using cached (hash={code_hash[:8]}, in_axes={in_axes})")
        return _vmapped_jit_cache[cache_key]

    vmapped_jit_fn = jax.jit(jax.vmap(fn, in_axes=in_axes))
    _vmapped_jit_cache[cache_key] = vmapped_jit_fn
    logger.debug(f"    vmapped_jit: cached new (hash={code_hash[:8]}, in_axes={in_axes})")
    return vmapped_jit_fn


def exec_with_cache_mlx(
    jax_code: str, fn_name: str
) -> Optional[Callable]:
    """Transform JAX code to MLX and execute, caching by code hash.

    Args:
        jax_code: Python source code using jnp.*, jax.*, lax.* APIs
        fn_name: Name of function to extract from executed code

    Returns:
        The MLX function, or None if MLX is not available
    """
    if not is_mlx_available():
        return None

    from openvaf_jax.mlx_transform import get_mlx_exec_namespace, jax_to_mlx

    code_hash = hashlib.sha256(jax_code.encode()).hexdigest()
    mlx_key = code_hash + "_mlx"

    if mlx_key in _mlx_exec_fn_cache:
        logger.debug(f"    {fn_name}: using cached MLX function (hash={code_hash[:8]})")
        return _mlx_exec_fn_cache[mlx_key]

    mlx_code = jax_to_mlx(jax_code)
    ns = get_mlx_exec_namespace()
    exec(mlx_code, ns)
    fn = cast(Callable, ns[fn_name])

    _mlx_exec_fn_cache[mlx_key] = fn
    logger.debug(f"    {fn_name}: cached new MLX function (hash={code_hash[:8]})")
    return fn


def get_vmapped_compiled_mlx(
    code_hash: str, fn: Callable, in_axes: Tuple
) -> Optional[Callable]:
    """Get a cached mx.vmap'd + mx.compile'd version of a function.

    Args:
        code_hash: Hash of the function's source code
        fn: The function to vmap and compile
        in_axes: vmap in_axes specification (same format as JAX)

    Returns:
        vmapped and compiled MLX function, or None if MLX unavailable
    """
    if not is_mlx_available():
        return None

    import mlx.core as mx

    cache_key = (code_hash + "_mlx", in_axes)

    if cache_key in _mlx_vmapped_cache:
        logger.debug(f"    mlx_vmapped: using cached (hash={code_hash[:8]}, in_axes={in_axes})")
        return _mlx_vmapped_cache[cache_key]

    # mx.vmap can't handle None in in_axes, so we wrap the function
    # to remove the limit_funcs arg (always None for MLX)
    def wrapped(shared, device, sc, dc, sp, ls):
        return fn(shared, device, sc, dc, sp, ls, None)

    # Filter in_axes to remove the last one (limit_funcs=None)
    mlx_in_axes = in_axes[:-1] if len(in_axes) > 6 else in_axes

    vmapped_fn = mx.compile(mx.vmap(wrapped, in_axes=mlx_in_axes))
    _mlx_vmapped_cache[cache_key] = vmapped_fn
    logger.debug(f"    mlx_vmapped: cached new (hash={code_hash[:8]}, in_axes={in_axes})")
    return vmapped_fn


def clear_cache():
    """Clear all function caches (useful for testing or memory management)."""
    global _exec_fn_cache, _vmapped_jit_cache, _mlx_exec_fn_cache, _mlx_vmapped_cache
    _exec_fn_cache.clear()
    _vmapped_jit_cache.clear()
    _mlx_exec_fn_cache.clear()
    _mlx_vmapped_cache.clear()


def cache_stats() -> Dict[str, int]:
    """Get cache statistics.

    Returns:
        Dict with counts for JAX and MLX caches
    """
    return {
        "exec_fn_count": len(_exec_fn_cache),
        "vmapped_jit_count": len(_vmapped_jit_cache),
        "mlx_exec_fn_count": len(_mlx_exec_fn_cache),
        "mlx_vmapped_count": len(_mlx_vmapped_cache),
    }


# =============================================================================
# Persistent Code Cache
# =============================================================================


def get_persistent_cache_dir() -> Path:
    """Get the persistent cache directory, creating it if needed.

    Uses VAJAX_CACHE_DIR env var, or ~/.cache/vajax/openvaf by default.

    Returns:
        Path to the cache directory
    """
    global _PERSISTENT_CACHE_DIR

    if _PERSISTENT_CACHE_DIR is not None:
        return _PERSISTENT_CACHE_DIR

    cache_dir = os.environ.get("VAJAX_CACHE_DIR")
    if cache_dir:
        _PERSISTENT_CACHE_DIR = Path(cache_dir) / "openvaf"
    else:
        _PERSISTENT_CACHE_DIR = Path.home() / ".cache" / "vajax" / "openvaf"

    _PERSISTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _PERSISTENT_CACHE_DIR


def compute_va_hash(va_path: Path) -> str:
    """Compute hash of a VA file for cache keying.

    Args:
        va_path: Path to the Verilog-A file

    Returns:
        SHA256 hash of the file content (first 16 chars)
    """
    content = va_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def get_model_cache_path(model_type: str, va_hash: str) -> Path:
    """Get the cache directory path for a specific model.

    Args:
        model_type: Model type name (e.g., 'psp103')
        va_hash: Hash of the VA file content

    Returns:
        Path to the model's cache directory
    """
    cache_dir = get_persistent_cache_dir()
    return cache_dir / f"{model_type}_{va_hash}"


def load_cached_model(model_type: str, va_path: Path) -> Optional[Dict[str, Any]]:
    """Load a cached model if available and valid.

    Checks if:
    1. Cache directory exists for this model + VA hash
    2. Metadata file exists and is readable
    3. Generated code files exist

    Args:
        model_type: Model type name (e.g., 'psp103')
        va_path: Path to the Verilog-A source file

    Returns:
        Cached model dict with 'metadata', 'init_code', 'eval_code', or None if not cached
    """
    va_hash = compute_va_hash(va_path)
    cache_path = get_model_cache_path(model_type, va_hash)

    metadata_file = cache_path / "metadata.json"
    init_code_file = cache_path / "init_fn.py"
    eval_code_file = cache_path / "eval_fn.py"

    # Check if all required files exist
    if not all(f.exists() for f in [metadata_file, init_code_file, eval_code_file]):
        logger.debug(f"  {model_type}: no valid cache found")
        return None

    try:
        # Load metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Load generated code
        init_code = init_code_file.read_text()
        eval_code = eval_code_file.read_text()

        logger.info(f"  {model_type}: loaded from persistent cache (hash={va_hash})")
        return {
            "metadata": metadata,
            "init_code": init_code,
            "eval_code": eval_code,
            "va_hash": va_hash,
        }
    except Exception as e:
        logger.warning(f"  {model_type}: failed to load cache: {e}")
        return None


def save_model_cache(
    model_type: str,
    va_path: Path,
    metadata: Dict[str, Any],
    init_code: str,
    eval_code: str,
) -> bool:
    """Save model to persistent cache.

    Args:
        model_type: Model type name (e.g., 'psp103')
        va_path: Path to the Verilog-A source file
        metadata: Model metadata (param names, nodes, cache size, etc.)
        init_code: Generated Python code for init function
        eval_code: Generated Python code for eval function

    Returns:
        True if cache was saved successfully
    """
    va_hash = compute_va_hash(va_path)
    cache_path = get_model_cache_path(model_type, va_hash)

    try:
        # Create cache directory
        cache_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata_file = cache_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save generated code
        init_code_file = cache_path / "init_fn.py"
        init_code_file.write_text(init_code)

        eval_code_file = cache_path / "eval_fn.py"
        eval_code_file.write_text(eval_code)

        logger.info(f"  {model_type}: saved to persistent cache (hash={va_hash})")
        return True
    except Exception as e:
        logger.warning(f"  {model_type}: failed to save cache: {e}")
        return False


def clear_persistent_cache():
    """Clear the entire persistent cache directory."""
    import shutil

    cache_dir = get_persistent_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Cleared persistent OpenVAF cache")


def cleanup_persistent_cache(max_age_days: int = 30) -> dict:
    """Remove persistent cache entries older than max_age_days.

    Uses the most recent file modification time within each model's cache
    directory to determine age.

    Args:
        max_age_days: Maximum age in days before a cache entry is removed.

    Returns:
        Dict with cleanup stats: {"removed": [...], "kept": [...]}
    """
    import shutil
    import time

    cache_dir = get_persistent_cache_dir()
    if not cache_dir.exists():
        return {"removed": [], "kept": []}

    cutoff = time.time() - (max_age_days * 86400)
    removed = []
    kept = []

    for entry in cache_dir.iterdir():
        if not entry.is_dir():
            continue
        # Find newest file in the cache entry
        newest_mtime = 0.0
        for f in entry.rglob("*"):
            if f.is_file():
                newest_mtime = max(newest_mtime, f.stat().st_mtime)
        if newest_mtime == 0.0:
            newest_mtime = entry.stat().st_mtime

        if newest_mtime < cutoff:
            shutil.rmtree(entry)
            removed.append(entry.name)
            logger.info(f"Removed stale cache entry: {entry.name}")
        else:
            kept.append(entry.name)

    if removed:
        logger.info(f"Cache cleanup: removed {len(removed)}, kept {len(kept)}")
    return {"removed": removed, "kept": kept}
