"""MLX backend for VAJAX circuit simulation on Apple Silicon.

This module provides MLX equivalents of the JAX-based simulation pipeline,
enabling GPU-accelerated circuit simulation on Apple Silicon via Metal.

Key design decisions:
- Reuses existing JAX-path setup logic (NOI masks, vsource incidence, etc.)
  by computing them in NumPy/JAX at factory time and converting to MLX arrays
- Dense-only: MLX has no sparse matrix support
- CPU solve: mx.linalg.solve only works on CPU stream
- Float32: Metal doesn't support float64
- Python while loop: MLX doesn't have lax.while_loop equivalent

The build_system and NR solver share the same algorithmic structure as the
JAX path (mna_builder.py, solver_factories.py) but use MLX array operations.
Setup/configuration code is shared via numpy computation at factory time.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# =============================================================================
# Array conversion utilities
# =============================================================================


def _to_mx(arr, dtype=None) -> "mx.array":
    """Convert JAX/numpy array to MLX, defaulting to float32."""
    if dtype is None:
        dtype = mx.float32
    return mx.array(np.asarray(arr), dtype=dtype)


def _to_mx_int(arr) -> "mx.array":
    """Convert JAX/numpy integer array to MLX int32."""
    return mx.array(np.asarray(arr), dtype=mx.int32)


# =============================================================================
# Shared setup: compute masks and incidence using numpy (framework-agnostic)
# =============================================================================
# These functions compute values at factory time using numpy. The results
# are then converted to MLX arrays. This avoids duplicating the mask
# computation logic from solver_factories.py._compute_noi_masks.


def compute_noi_masks_np(
    noi_indices: Optional[np.ndarray],
    internal_device_indices: Optional[np.ndarray],
    n_unknowns: int,
    n_vsources: int,
) -> Dict[str, Optional[np.ndarray]]:
    """Compute NOI and internal device masks in numpy (shared by JAX and MLX).

    Same algorithm as solver_factories._compute_noi_masks but returns numpy
    arrays that can be converted to any framework.

    Returns dict with:
        noi_res_idx: NOI residual indices (or None)
        residual_mask: Boolean mask for delta convergence (augmented)
        residual_conv_mask: Boolean mask for residual convergence (augmented)
    """
    result: Dict[str, Optional[np.ndarray]] = {
        "noi_res_idx": None,
        "residual_mask": None,
        "residual_conv_mask": None,
    }

    # Internal device node mask (VACASK-style: skip in residual convergence)
    if internal_device_indices is not None and len(internal_device_indices) > 0:
        conv_mask = np.ones(n_unknowns, dtype=bool)
        conv_mask[np.asarray(internal_device_indices) - 1] = False
        result["residual_conv_mask"] = np.concatenate([
            conv_mask, np.ones(n_vsources, dtype=bool)
        ])

    # NOI node mask
    if noi_indices is not None and len(noi_indices) > 0:
        noi_res_idx = np.asarray(noi_indices) - 1
        result["noi_res_idx"] = noi_res_idx

        mask = np.ones(n_unknowns, dtype=bool)
        mask[noi_res_idx] = False
        result["residual_mask"] = np.concatenate([
            mask, np.ones(n_vsources, dtype=bool)
        ])

        # If no conv_mask was set, fall back to NOI mask
        if result["residual_conv_mask"] is None:
            result["residual_conv_mask"] = result["residual_mask"]

    return result


def compute_vsource_incidence_np(
    vsource_node_p: np.ndarray,
    vsource_node_n: np.ndarray,
    n_unknowns: int,
    n_vsources: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute vsource incidence COO in numpy (shared computation).

    Same algorithm as mna.build_vsource_incidence_coo but returns numpy arrays.
    """
    if n_vsources == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    valid_p = vsource_node_p > 0
    valid_n = vsource_node_n > 0
    branch_idx = np.arange(n_vsources, dtype=np.int32)

    # B block
    rows = [
        np.where(valid_p, vsource_node_p - 1, 0).astype(np.int32),
        np.where(valid_n, vsource_node_n - 1, 0).astype(np.int32),
    ]
    cols = [
        np.where(valid_p, n_unknowns + branch_idx, 0).astype(np.int32),
        np.where(valid_n, n_unknowns + branch_idx, 0).astype(np.int32),
    ]
    vals = [
        np.where(valid_p, 1.0, 0.0).astype(np.float32),
        np.where(valid_n, -1.0, 0.0).astype(np.float32),
    ]

    # B^T block
    rows += [
        np.where(valid_p, n_unknowns + branch_idx, 0).astype(np.int32),
        np.where(valid_n, n_unknowns + branch_idx, 0).astype(np.int32),
    ]
    cols += [
        np.where(valid_p, vsource_node_p - 1, 0).astype(np.int32),
        np.where(valid_n, vsource_node_n - 1, 0).astype(np.int32),
    ]
    vals += [
        np.where(valid_p, 1.0, 0.0).astype(np.float32),
        np.where(valid_n, -1.0, 0.0).astype(np.float32),
    ]

    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


# =============================================================================
# Shared convergence logic (pure Python, works with any array backend)
# =============================================================================


def check_nr_convergence(
    residual_converged: bool,
    delta_converged: bool,
    iteration: int,
    limit_state_out_np: Optional[np.ndarray],
    limit_state_np: Optional[np.ndarray],
    total_limit_states: int,
    reltol: float,
    vntol: float,
) -> bool:
    """Check NR convergence (shared logic for JAX and MLX paths).

    VACASK-style AND convergence: both delta and residual must converge.
    When device limiting is active, limit states must also settle.

    Args are expected as Python scalars/numpy arrays (both backends
    can convert their arrays to numpy for this check).
    """
    converged = residual_converged and delta_converged

    if total_limit_states > 0 and limit_state_out_np is not None and limit_state_np is not None:
        limit_delta = float(np.max(np.abs(limit_state_out_np - limit_state_np)))
        limit_ref = max(float(np.max(np.abs(limit_state_np))) * reltol, vntol)
        limit_settled = limit_delta < limit_ref
        converged = converged and limit_settled and (iteration >= 1)

    return converged


# =============================================================================
# MLX-specific array operations for build_system
# =============================================================================


def _stamp_vector_mlx(vec, indices, values):
    """Stamp values into vector at indices, masking invalid (-1)."""
    valid = indices >= 0
    pos = mx.where(valid, indices, mx.array(0, dtype=mx.int32))
    vals = mx.where(valid, values, mx.array(0.0, dtype=mx.float32))
    vals = mx.where(mx.isnan(vals), mx.array(0.0, dtype=mx.float32), vals)
    return vec.at[pos].add(vals)


def _assemble_coo_vector_mlx(all_idx, all_val, size):
    """Assemble dense vector from COO data via .at[].add()."""
    vec = mx.zeros(size, dtype=mx.float32)
    return vec.at[all_idx].add(all_val)


def _assemble_coo_max_abs_mlx(all_idx, all_val, size):
    """Max absolute value per index from COO data."""
    vec = mx.zeros(size, dtype=mx.float32)
    return vec.at[all_idx].maximum(mx.abs(all_val))


def _assemble_dense_jacobian_mlx(j_rows, j_cols, j_vals, n_aug, n_unk, n_vs, min_diag, gshunt):
    """Assemble dense Jacobian via flat indexing + .at[].add()."""
    flat_idx = j_rows * n_aug + j_cols
    J_flat = mx.zeros(n_aug * n_aug, dtype=mx.float32)
    J_flat = J_flat.at[flat_idx].add(j_vals)
    J = J_flat.reshape(n_aug, n_aug)

    diag_reg = mx.concatenate([
        mx.full(n_unk, min_diag + gshunt, dtype=mx.float32),
        mx.zeros(n_vs, dtype=mx.float32),
    ])
    return J + mx.diag(diag_reg)


# =============================================================================
# MLX Build System Function
# =============================================================================


def make_mlx_build_system_fn(
    source_device_data: Dict[str, Any],
    static_inputs_cache: Dict[str, Tuple],
    compiled_models: Dict[str, Dict[str, Any]],
    gmin: float,
    n_unknowns: int,
) -> Tuple[Callable, Dict[str, "mx.array"], int]:
    """Create MLX build_system function for dense MNA.

    Same algorithmic structure as mna_builder.make_mna_build_system_fn (COO path)
    but uses MLX arrays. Setup computation uses numpy (shared with JAX path).

    Returns:
        Tuple of (build_system_fn, device_arrays_mlx, total_limit_states)
    """
    n_vsources = len(source_device_data.get("vsource", {}).get("names", []))
    n_augmented = n_unknowns + n_vsources
    model_types = list(static_inputs_cache.keys())
    min_diag_reg = gmin

    # Convert static metadata to MLX (factory-time conversion)
    static_metadata_mlx: Dict[str, Tuple] = {}
    device_arrays_mlx: Dict[str, "mx.array"] = {}
    split_eval_info: Dict[str, Dict[str, Any]] = {}

    for model_type in model_types:
        voltage_indices, stamp_indices, voltage_node1, voltage_node2, cache, _ = (
            static_inputs_cache[model_type]
        )
        stamp_indices_mlx = {k: _to_mx_int(v) for k, v in stamp_indices.items()}
        static_metadata_mlx[model_type] = (
            voltage_indices,
            stamp_indices_mlx,
            _to_mx_int(voltage_node1),
            _to_mx_int(voltage_node2),
        )

        compiled = compiled_models.get(model_type, {})
        n_devices = compiled["device_params"].shape[0]
        num_limit_states = compiled.get("num_limit_states", 0)

        # Get MLX eval fn
        eval_meta = compiled.get("eval_meta", {})
        mlx_eval_fn = eval_meta.get("mlx_eval_fn")
        assert mlx_eval_fn is not None, (
            f"No MLX eval function for {model_type}. "
            "Ensure translate_eval() was called with MLX available."
        )

        # Wrap for vmap (remove None limit_funcs arg)
        def _make_wrapped(fn):
            def wrapped(shared, device, sc, dc, sp, ls):
                return fn(shared, device, sc, dc, sp, ls, None)
            return wrapped

        vmapped_fn = mx.vmap(
            _make_wrapped(mlx_eval_fn),
            in_axes=(None, 0, None, 0, None, 0),
        )

        default_sp = np.asarray(compiled.get(
            "default_simparams", np.array([0.0, 1.0, 1e-12], dtype=np.float32)
        ))

        split_eval_info[model_type] = {
            "vmapped_eval_mlx": vmapped_fn,
            "shared_params": _to_mx(compiled["shared_params"]),
            "device_params": _to_mx(compiled["device_params"]),
            "voltage_positions": np.asarray(compiled["voltage_positions_in_varying"]).tolist(),
            "shared_cache": _to_mx(compiled["shared_cache"]),
            "default_simparams": mx.array(default_sp, dtype=mx.float32),
            "simparam_indices": compiled.get("simparam_indices", {}),
            "uses_analysis": compiled.get("uses_analysis", False),
            "uses_simparam_gmin": compiled.get("uses_simparam_gmin", False),
            "use_device_limiting": compiled.get("use_device_limiting", False),
            "num_limit_states": num_limit_states,
            "n_devices": n_devices,
        }
        device_arrays_mlx[model_type] = _to_mx(compiled["device_cache"])

    # Vsource setup (numpy computation shared with JAX path)
    vs_node_p_np = np.asarray(source_device_data.get("vsource", {}).get("node_p", []), dtype=np.int32)
    vs_node_n_np = np.asarray(source_device_data.get("vsource", {}).get("node_n", []), dtype=np.int32)
    vs_node_p = _to_mx_int(vs_node_p_np) if n_vsources > 0 else mx.zeros(0, dtype=mx.int32)
    vs_node_n = _to_mx_int(vs_node_n_np) if n_vsources > 0 else mx.zeros(0, dtype=mx.int32)

    # Pre-compute vsource incidence COO using shared numpy logic
    vs_inc_rows_np, vs_inc_cols_np, vs_inc_vals_np = compute_vsource_incidence_np(
        vs_node_p_np, vs_node_n_np, n_unknowns, n_vsources
    )
    vs_inc_rows = _to_mx_int(vs_inc_rows_np)
    vs_inc_cols = _to_mx_int(vs_inc_cols_np)
    vs_inc_vals = _to_mx(vs_inc_vals_np)

    # Isource data
    isource_f_indices = None
    if "isource" in source_device_data:
        isource_f_indices = _to_mx_int(source_device_data["isource"]["f_indices"])

    # Limit state offsets (same algorithm as mna_builder.py)
    total_limit_states = 0
    limit_state_offsets: Dict[str, Tuple[int, int, int]] = {}
    for model_type in model_types:
        info = split_eval_info[model_type]
        if info.get("use_device_limiting", False) and info.get("num_limit_states", 0) > 0:
            n_dev = info["n_devices"]
            n_lim = info["num_limit_states"]
            limit_state_offsets[model_type] = (total_limit_states, n_dev, n_lim)
            total_limit_states += n_dev * n_lim

    def _prepare_simparams(split_info, integ_c0, gmin_arg, nr_iteration):
        """Build simparams array. Same algorithm as mna_builder._prepare_simparams."""
        simparams = mx.array(split_info["default_simparams"])  # copy
        si = split_info.get("simparam_indices", {})

        analysis_val = mx.where(mx.array(integ_c0) > 0, mx.array(2.0), mx.array(0.0))
        iniLim_val = mx.where(mx.array(nr_iteration) == 0, mx.array(1.0), mx.array(0.0))

        if "$analysis_type" in si:
            simparams[si["$analysis_type"]] = analysis_val
        if "gmin" in si:
            simparams[si["gmin"]] = mx.array(gmin_arg, dtype=mx.float32)
        if "iniLim" in si:
            simparams[si["iniLim"]] = iniLim_val
        if "iteration" in si:
            simparams[si["iteration"]] = mx.array(float(nr_iteration), dtype=mx.float32)
        return simparams

    def build_system_mlx(
        X, vsource_vals, isource_vals, Q_prev, integ_c0,
        device_arrays_arg, gmin_arg=1e-12, gshunt=0.0,
        integ_c1=0.0, integ_d1=0.0, dQdt_prev=None,
        integ_c2=0.0, Q_prev2=None, limit_state_in=None, nr_iteration=1,
    ):
        """Build Jacobian J and residual f (MLX, dense COO path).

        Same algorithm as mna_builder._build_system_coo but with MLX arrays.
        Returns: (J, f, Q, I_vsource, limit_state_out, max_res_contrib)
        """
        n_total = n_unknowns + 1
        V = X[:n_total]
        I_branch = X[n_total:] if n_vsources > 0 else mx.zeros(0, dtype=mx.float32)
        limit_state_out = mx.zeros(max(total_limit_states, 1), dtype=mx.float32)

        # Accumulators for COO parts
        res_parts_idx, res_parts_val = [], []
        react_parts_idx, react_parts_val = [], []
        j_parts_rows, j_parts_cols, j_parts_vals = [], [], []
        lim_res_parts_idx, lim_res_parts_val = [], []
        lim_react_parts_idx, lim_react_parts_val = [], []

        # Current sources
        if isource_f_indices is not None and isource_vals.size > 0:
            f_vals = isource_vals[:, None] * mx.array([[1.0, -1.0]])
            f_idx = isource_f_indices.reshape(-1)
            f_val = f_vals.reshape(-1)
            valid = f_idx >= 0
            res_parts_idx.append(mx.where(valid, f_idx, mx.array(0, dtype=mx.int32)))
            res_parts_val.append(mx.where(valid, f_val, mx.array(0.0, dtype=mx.float32)))

        # OpenVAF devices (same loop structure as mna_builder._build_system_coo)
        for model_type in model_types:
            _, stamp_mlx, vn1, vn2 = static_metadata_mlx[model_type]
            cache = device_arrays_arg[model_type]
            voltage_updates = V[vn1] - V[vn2]

            si = split_eval_info[model_type]
            dp = si["device_params"]
            vp = si["voltage_positions"]
            dp_updated = mx.array(dp)  # copy
            dp_updated[:, vp] = voltage_updates

            if si["uses_analysis"]:
                atv = mx.where(mx.array(integ_c0) > 0, mx.array(2.0), mx.array(0.0))
                n_d = dp_updated.shape[0]
                dp_updated[:, -2] = mx.full((n_d,), atv.item(), dtype=mx.float32)
                dp_updated[:, -1] = mx.full((n_d,), gmin_arg, dtype=mx.float32)
            elif si["uses_simparam_gmin"]:
                n_d = dp_updated.shape[0]
                dp_updated[:, -1] = mx.full((n_d,), gmin_arg, dtype=mx.float32)

            simparams = _prepare_simparams(si, integ_c0, gmin_arg, nr_iteration)
            n_dev = si["n_devices"]
            n_lim = max(1, si.get("num_limit_states", 0))

            use_lim = si.get("use_device_limiting", False) and model_type in limit_state_offsets
            if use_lim and limit_state_in is not None:
                off, _, nl = limit_state_offsets[model_type]
                mls_in = limit_state_in[off:off + n_dev * nl].reshape(n_dev, nl)
            else:
                mls_in = mx.zeros((n_dev, n_lim), dtype=mx.float32)

            (rr, rc, jr, jc, lrr, lrc, _, _, lso) = si["vmapped_eval_mlx"](
                si["shared_params"], dp_updated, si["shared_cache"],
                cache, simparams, mls_in,
            )

            if use_lim:
                off, _, nl = limit_state_offsets[model_type]
                limit_state_out[off:off + n_dev * nl] = lso.reshape(-1)

            ri = stamp_mlx["res_indices"].reshape(-1)
            jri = stamp_mlx["jac_row_indices"].reshape(-1)
            jci = stamp_mlx["jac_col_indices"].reshape(-1)

            valid_r = ri >= 0
            res_parts_idx.append(mx.where(valid_r, ri, mx.array(0, dtype=mx.int32)))
            res_parts_val.append(mx.where(valid_r, rr.reshape(-1), mx.array(0.0, dtype=mx.float32)))
            react_parts_idx.append(mx.where(valid_r, ri, mx.array(0, dtype=mx.int32)))
            react_parts_val.append(mx.where(valid_r, rc.reshape(-1), mx.array(0.0, dtype=mx.float32)))

            valid_j = (jri >= 0) & (jci >= 0)
            cj = jr.reshape(-1) + integ_c0 * jc.reshape(-1)
            cj = mx.where(mx.isnan(cj), mx.array(0.0, dtype=mx.float32), cj)
            j_parts_rows.append(mx.where(valid_j, jri, mx.array(0, dtype=mx.int32)))
            j_parts_cols.append(mx.where(valid_j, jci, mx.array(0, dtype=mx.int32)))
            j_parts_vals.append(mx.where(valid_j, cj, mx.array(0.0, dtype=mx.float32)))

            lim_res_parts_idx.append(mx.where(valid_r, ri, mx.array(0, dtype=mx.int32)))
            lim_res_parts_val.append(mx.where(valid_r, lrr.reshape(-1), mx.array(0.0, dtype=mx.float32)))
            lim_react_parts_idx.append(mx.where(valid_r, ri, mx.array(0, dtype=mx.int32)))
            lim_react_parts_val.append(mx.where(valid_r, lrc.reshape(-1), mx.array(0.0, dtype=mx.float32)))

        # --- Assemble vectors ---
        def _cat_and_assemble(idx_parts, val_parts, size):
            if not idx_parts:
                return mx.zeros(size, dtype=mx.float32)
            return _assemble_coo_vector_mlx(mx.concatenate(idx_parts), mx.concatenate(val_parts), size)

        f_resist = _cat_and_assemble(res_parts_idx, res_parts_val, n_unknowns)
        Q = _cat_and_assemble(react_parts_idx, react_parts_val, n_unknowns)
        lim_rhs_resist = _cat_and_assemble(lim_res_parts_idx, lim_res_parts_val, n_unknowns)
        lim_rhs_react = _cat_and_assemble(lim_react_parts_idx, lim_react_parts_val, n_unknowns)

        max_res_contrib = (
            _assemble_coo_max_abs_mlx(mx.concatenate(res_parts_idx), mx.concatenate(res_parts_val), n_unknowns)
            if res_parts_idx else mx.zeros(n_unknowns, dtype=mx.float32)
        )

        # --- Build residual (same formula as mna_builder._build_residual) ---
        f_resist = f_resist - lim_rhs_resist
        # I_vsource from KCL
        if vs_node_p.size > 0:
            p_mna = vs_node_p - 1
            valid_p = vs_node_p > 0
            I_vsource_kcl = -mx.where(valid_p, f_resist[p_mna], mx.array(0.0, dtype=mx.float32))
        else:
            I_vsource_kcl = mx.zeros(0, dtype=mx.float32)

        _dQdt_prev = dQdt_prev if dQdt_prev is not None else mx.zeros(n_unknowns, dtype=mx.float32)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else mx.zeros(n_unknowns, dtype=mx.float32)

        # Branch current KCL contribution
        if n_vsources > 0:
            bp_mna = vs_node_p - 1
            bn_mna = vs_node_n - 1
            bp_valid = vs_node_p > 0
            bn_valid = vs_node_n > 0
            br_idx = mx.concatenate([
                mx.where(bp_valid, bp_mna, mx.array(0, dtype=mx.int32)),
                mx.where(bn_valid, bn_mna, mx.array(0, dtype=mx.int32)),
            ])
            br_val = mx.concatenate([
                mx.where(bp_valid, I_branch, mx.array(0.0, dtype=mx.float32)),
                mx.where(bn_valid, -I_branch, mx.array(0.0, dtype=mx.float32)),
            ])
            f_resist = f_resist + _assemble_coo_vector_mlx(br_idx, br_val, n_unknowns)

        # Transient residual combination (same formula as mna.combine_transient_residual)
        effective_shunt = min_diag_reg + gshunt
        f_node = (
            f_resist - mx.zeros_like(f_resist)  # lim_rhs_resist already subtracted
            + integ_c0 * (Q - lim_rhs_react)
            + integ_c1 * Q_prev
            + integ_d1 * _dQdt_prev
            + integ_c2 * _Q_prev2
            + effective_shunt * V[1:]
        )

        # Vsource equations: V_p - V_n - E = 0
        if n_vsources > 0:
            f_branch = V[vs_node_p] - V[vs_node_n] - vsource_vals
        else:
            f_branch = mx.zeros(0, dtype=mx.float32)
        f_augmented = mx.concatenate([f_node, f_branch])

        # --- Assemble Jacobian ---
        if j_parts_rows:
            cat_jr = mx.concatenate(j_parts_rows)
            cat_jc = mx.concatenate(j_parts_cols)
            cat_jv = mx.concatenate(j_parts_vals)
        else:
            cat_jr = mx.zeros(0, dtype=mx.int32)
            cat_jc = mx.zeros(0, dtype=mx.int32)
            cat_jv = mx.zeros(0, dtype=mx.float32)

        if n_vsources > 0:
            cat_jr = mx.concatenate([cat_jr, vs_inc_rows])
            cat_jc = mx.concatenate([cat_jc, vs_inc_cols])
            cat_jv = mx.concatenate([cat_jv, vs_inc_vals])

        J = _assemble_dense_jacobian_mlx(
            cat_jr, cat_jc, cat_jv, n_augmented, n_unknowns, n_vsources,
            min_diag_reg, gshunt,
        )

        return J, f_augmented, Q, I_vsource_kcl, limit_state_out, max_res_contrib

    return build_system_mlx, device_arrays_mlx, total_limit_states


# =============================================================================
# MLX NR Solver
# =============================================================================


def make_mlx_nr_solver(
    build_system_fn: Callable,
    n_nodes: int,
    n_vsources: int,
    noi_indices: Optional[np.ndarray] = None,
    internal_device_indices: Optional[np.ndarray] = None,
    max_iterations: int = 100,
    abstol: float = 1e-12,
    total_limit_states: int = 0,
    vntol: float = 1e-6,
    reltol: float = 1e-3,
    nr_damping: float = 1.0,
    max_step: float = 1e30,
) -> Callable:
    """Create an MLX NR solver with Python while loop.

    Same convergence algorithm as solver_factories._make_nr_solver_common
    but uses Python while loop + MLX arrays instead of lax.while_loop + JAX.

    Setup uses shared numpy computation (compute_noi_masks_np).
    Convergence check uses shared logic (check_nr_convergence).
    """
    n_unknowns = n_nodes - 1
    n_total = n_nodes

    # Compute masks using shared numpy logic
    masks = compute_noi_masks_np(noi_indices, internal_device_indices, n_unknowns, n_vsources)
    noi_res_idx_np = masks["noi_res_idx"]
    residual_mask_mx = _to_mx(masks["residual_mask"]) if masks["residual_mask"] is not None else None
    residual_conv_mask_mx = _to_mx(masks["residual_conv_mask"]) if masks["residual_conv_mask"] is not None else None

    # Per-unknown absolute tolerance
    delta_abs_tol = mx.concatenate([
        mx.full(n_unknowns, vntol, dtype=mx.float32),
        mx.full(n_vsources, abstol, dtype=mx.float32),
    ])

    def enforce_noi(J, f):
        """Enforce NOI constraints on dense Jacobian."""
        if noi_res_idx_np is not None:
            for idx in noi_res_idx_np.tolist():
                J[idx, :] = mx.zeros(J.shape[1], dtype=mx.float32)
                J[:, idx] = mx.zeros(J.shape[0], dtype=mx.float32)
                J[idx, idx] = mx.array(1.0, dtype=mx.float32)
                f[idx] = mx.array(0.0, dtype=mx.float32)
        return J, f

    def nr_solve(
        X_init, vsource_vals, isource_vals, Q_prev, integ_c0,
        device_arrays_arg, gmin=1e-12, gshunt=0.0,
        integ_c1=0.0, integ_d1=0.0, dQdt_prev=None,
        integ_c2=0.0, Q_prev2=None, limit_state_in=None, res_tol_floor=None,
    ):
        """Newton-Raphson solver (Python while loop, MLX arrays).

        Same convergence criteria as _make_nr_solver_common.
        """
        X = X_init
        _dQdt_prev = dQdt_prev if dQdt_prev is not None else mx.zeros(n_unknowns, dtype=mx.float32)
        _Q_prev2 = Q_prev2 if Q_prev2 is not None else mx.zeros(n_unknowns, dtype=mx.float32)
        limit_state = (
            limit_state_in if limit_state_in is not None
            else mx.zeros(max(total_limit_states, 1), dtype=mx.float32)
        )
        _res_tol_floor = (
            res_tol_floor if res_tol_floor is not None
            else mx.full(n_unknowns, abstol, dtype=mx.float32)
        )

        converged = False
        max_f = float("inf")
        Q_final = mx.zeros(n_unknowns, dtype=mx.float32)
        final_iter = 0

        for iteration in range(max_iterations):
            J, f, Q_cur, _, limit_state_out, max_res_contrib = build_system_fn(
                X, vsource_vals, isource_vals, Q_prev, integ_c0,
                device_arrays_arg, gmin, gshunt,
                integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2,
                limit_state, iteration,
            )

            # Residual tolerance (VACASK-style, same formula as solver_factories)
            res_tol_nodes = mx.maximum(max_res_contrib * reltol, _res_tol_floor)
            res_tol = mx.concatenate([res_tol_nodes, mx.full(n_vsources, vntol, dtype=mx.float32)])

            f_check = f
            if residual_conv_mask_mx is not None:
                f_check = mx.where(residual_conv_mask_mx, f, mx.array(0.0, dtype=mx.float32))

            mx.eval(f_check, res_tol)
            max_f = float(mx.max(mx.abs(f_check)).item())
            residual_converged = bool(mx.all(mx.abs(f_check) < res_tol).item())

            # Enforce NOI and solve
            J, f_solve = enforce_noi(J, f)
            mx.eval(J, f_solve)

            reg = 1e-14 * mx.eye(J.shape[0], dtype=mx.float32)
            delta = mx.linalg.solve(J + reg, -f_solve, stream=mx.cpu)
            mx.eval(delta)

            # Delta convergence (same formula as solver_factories)
            conv_delta = mx.concatenate([delta[:n_unknowns] * nr_damping, delta[n_unknowns:]])
            X_ref = mx.concatenate([X[1:n_total], X[n_total:]])
            tol = mx.maximum(mx.abs(X_ref) * reltol, delta_abs_tol)
            if residual_mask_mx is not None:
                conv_delta = mx.where(residual_mask_mx, conv_delta, mx.array(0.0, dtype=mx.float32))
            delta_converged = (iteration == 0) or bool(mx.all(mx.abs(conv_delta) < tol).item())

            # Update solution (same step limiting as solver_factories)
            V_delta = delta[:n_unknowns]
            max_V_delta = float(mx.max(mx.abs(V_delta)).item())
            V_scale = min(1.0, max_step / max(max_V_delta, 1e-30))
            V_damped = V_delta * V_scale * nr_damping
            X = X.at[1:n_total].add(V_damped)
            X = X.at[n_total:].add(delta[n_unknowns:])

            if noi_res_idx_np is not None:
                noi_circuit_indices = noi_res_idx_np + 1  # back to circuit indices
                for noi_idx in noi_circuit_indices.tolist():
                    X[noi_idx] = mx.array(0.0, dtype=mx.float32)

            mx.eval(X, limit_state_out)

            # Convergence check using shared logic
            converged = check_nr_convergence(
                residual_converged, delta_converged, iteration,
                np.asarray(limit_state_out) if total_limit_states > 0 else None,
                np.asarray(limit_state) if total_limit_states > 0 else None,
                total_limit_states, reltol, vntol,
            )

            limit_state = limit_state_out
            Q_final = Q_cur
            final_iter = iteration + 1

            if converged:
                break

        # Recompute Q and I_vsource from converged solution
        _, _, Q_final, I_vsource, _, max_res_contrib_final = build_system_fn(
            X, vsource_vals, isource_vals, Q_prev, integ_c0,
            device_arrays_arg, gmin, gshunt,
            integ_c1, integ_d1, _dQdt_prev, integ_c2, _Q_prev2,
            limit_state, final_iter,
        )

        dQdt_final = integ_c0 * Q_final + integ_c1 * Q_prev + integ_d1 * _dQdt_prev + integ_c2 * _Q_prev2
        mx.eval(X, Q_final, dQdt_final, I_vsource, limit_state, max_res_contrib_final)

        return (X, final_iter, converged, max_f, Q_final, dQdt_final,
                I_vsource, limit_state, max_res_contrib_final)

    return nr_solve


# =============================================================================
# MLX DC Operating Point
# =============================================================================


def mlx_dc_operating_point(
    n_nodes: int,
    node_names: Dict[str, int],
    devices: List[Dict],
    nr_solve: Callable,
    device_arrays: Dict[str, "mx.array"],
    vsource_dc_vals: "mx.array",
    isource_dc_vals: "mx.array",
    vdd_value: float,
    device_internal_nodes: Optional[Dict[str, Dict[str, int]]] = None,
) -> "mx.array":
    """Compute DC operating point using MLX backend.

    Uses shared numpy logic for voltage initialization (same algorithm as
    dc_operating_point.initialize_dc_voltages).
    """
    logger.info("Computing DC operating point (MLX backend)...")

    # Initialize voltages using numpy (same algorithm as dc_operating_point.py)
    mid_rail = vdd_value / 2.0
    V_np = np.full(n_nodes, mid_rail, dtype=np.float32)
    V_np[0] = 0.0

    for name, idx in node_names.items():
        name_lower = name.lower()
        if "vdd" in name_lower or "vcc" in name_lower:
            V_np[idx] = vdd_value
        elif name_lower in ("gnd", "vss", "0"):
            V_np[idx] = 0.0

    for dev in devices:
        if dev["model"] == "vsource":
            nodes = dev.get("nodes", [])
            if len(nodes) >= 2:
                p_node, n_node = nodes[0], nodes[1]
                dc_val = float(dev["params"].get("dc", 0.0))
                if n_node == 0 and p_node > 0:
                    V_np[p_node] = dc_val
                elif p_node > 0:
                    V_np[p_node] = V_np[n_node] + dc_val

    noi_indices_list: List[int] = []
    if device_internal_nodes:
        for dev_name, internal_nodes in device_internal_nodes.items():
            if "node4" in internal_nodes:
                noi_idx = internal_nodes["node4"]
                V_np[noi_idx] = 0.0
                noi_indices_list.append(noi_idx)

    V = mx.array(V_np, dtype=mx.float32)
    n_unknowns = n_nodes - 1
    Q_prev = mx.zeros(n_unknowns, dtype=mx.float32)

    # Direct NR (no homotopy for MLX path — sufficient for well-initialized circuits)
    V_new, nr_iters, is_converged, max_f, _, _, _, _, _ = nr_solve(
        V, vsource_dc_vals, isource_dc_vals, Q_prev, 0.0, device_arrays,
    )

    V = V_new
    if is_converged:
        logger.info(f"  DC converged ({nr_iters} iters, residual={max_f:.2e})")
    else:
        logger.warning(f"  DC did not converge ({nr_iters} iters, residual={max_f:.2e})")

    for noi_idx in noi_indices_list:
        V[noi_idx] = mx.array(0.0, dtype=mx.float32)
    mx.eval(V)

    n_external = min(len(node_names), 5)
    logger.info(f"  DC solution ({n_external} nodes):")
    for i in range(n_external):
        name = next((n for n, idx in node_names.items() if idx == i), str(i))
        logger.info(f"    Node {name} (idx {i}): {float(V[i]):.6f}V")

    return V
