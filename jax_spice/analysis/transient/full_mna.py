"""Full MNA transient analysis strategy with adaptive timestep.

This strategy uses true Modified Nodal Analysis (MNA) with branch currents
as explicit unknowns, instead of the high-G (G=1e12) voltage source
approximation used in other strategies.

Benefits:
- More accurate current extraction (no numerical noise from G=1e12)
- Smoother dI/dt transitions matching VACASK reference
- Better conditioned matrices for ill-conditioned circuits

The augmented system has structure:

    ┌───────────────┐   ┌───┐   ┌───────┐
    │  G + c0*C   B │   │ V │   │ f_node│
    │               │ × │   │ = │       │
    │    B^T      0 │   │ J │   │ E - V │
    └───────────────┘   └───┘   └───────┘

Where:
- G = device conductance matrix (n×n)
- C = device capacitance matrix (n×n)
- B = incidence matrix mapping currents to nodes (n×m)
- V = node voltages (n×1)
- J = branch currents (m×1) - these are the primary unknowns for vsources
"""

import time as time_module
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

import numpy as np

from jax_spice._logging import logger
from jax_spice.analysis.integration import IntegrationMethod, compute_coefficients
from jax_spice.analysis.solver_factories import (
    make_dense_full_mna_solver,
    make_sparse_full_mna_solver,
    make_umfpack_full_mna_solver,
)
from jax_spice.analysis.umfpack_solver import is_umfpack_available

from .base import TransientSetup, TransientStrategy


class FullMNAStrategy(TransientStrategy):
    """Transient analysis using full MNA with explicit branch currents.

    This strategy uses the true MNA formulation where voltage source
    currents are explicit unknowns in the solution vector. This provides:

    - More accurate current extraction (branch currents are primary unknowns)
    - Smoother dI/dt during fast transitions
    - Better agreement with reference simulators (VACASK)

    Example:
        runner = CircuitEngine(sim_path)
        runner.parse()

        strategy = FullMNAStrategy(runner, use_sparse=False)

        # Warmup (JIT compilation)
        _ = strategy.run(t_stop=1e-9, dt=1e-12)

        # Timed run
        times, voltages, stats = strategy.run(t_stop=1e-6, dt=1e-9)
        print(f"I_VDD: {stats['currents']['vdd']}")  # Direct branch current
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_full_mna_solver: Optional[Callable] = None
        self._cached_full_mna_key: Optional[Tuple] = None

    def ensure_setup(self) -> TransientSetup:
        """Ensure transient setup is initialized with full MNA data.

        Extends base setup with branch current information for full MNA.
        """
        # Get base setup first (this populates runner caches)
        base_setup = super().ensure_setup()

        # Add full MNA branch data
        from jax_spice.analysis.mna import MNABranchData

        branch_data = MNABranchData.from_devices(self.runner.devices, self.runner.node_names)

        # Augment setup with full MNA info
        self._setup = TransientSetup(
            n_total=base_setup.n_total,
            n_unknowns=base_setup.n_unknowns,
            n_external=base_setup.n_external,
            device_internal_nodes=base_setup.device_internal_nodes,
            source_fn=base_setup.source_fn,
            source_device_data=base_setup.source_device_data,
            openvaf_by_type=base_setup.openvaf_by_type,
            vmapped_fns=base_setup.vmapped_fns,
            static_inputs_cache=base_setup.static_inputs_cache,
            # Full MNA fields
            n_branches=branch_data.n_branches,
            n_augmented=base_setup.n_unknowns + branch_data.n_branches,
            use_full_mna=True,
            branch_data=branch_data,
            branch_node_p=jnp.array(branch_data.node_p, dtype=jnp.int32) if branch_data.node_p else None,
            branch_node_n=jnp.array(branch_data.node_n, dtype=jnp.int32) if branch_data.node_n else None,
        )
        return self._setup

    def _ensure_full_mna_solver(self, setup: TransientSetup) -> Callable:
        """Ensure full MNA solver is created and cached."""
        n_nodes = setup.n_unknowns + 1
        n_vsources = setup.n_branches

        cache_key = (n_nodes, n_vsources, self.use_dense)

        if self._cached_full_mna_solver is not None and self._cached_full_mna_key == cache_key:
            return self._cached_full_mna_solver

        # Create full MNA build_system function
        build_system_fn, device_arrays = self.runner._make_full_mna_build_system_fn(
            setup.source_device_data,
            setup.vmapped_fns,
            setup.static_inputs_cache,
            setup.n_unknowns,
            use_dense=self.use_dense,
        )

        # Store device arrays for solver
        self._device_arrays_full_mna = device_arrays

        # JIT compile build_system
        build_system_jit = jax.jit(build_system_fn)

        # Collect NOI node indices
        noi_indices = []
        if setup.device_internal_nodes:
            for dev_name, internal_nodes in setup.device_internal_nodes.items():
                if 'node4' in internal_nodes:  # NOI is node4 in PSP103
                    noi_indices.append(internal_nodes['node4'])
        noi_indices = jnp.array(noi_indices, dtype=jnp.int32) if noi_indices else None

        # Create full MNA solver
        # Note: For full MNA, residuals are in Amperes (node) and Volts (branch)
        # Use tighter tolerance than high-G version (which scales by 1e12)
        # abstol=1e-6 means 1µA current error and 1µV voltage error
        if self.use_dense:
            nr_solve = make_dense_full_mna_solver(
                build_system_jit, n_nodes, n_vsources,
                noi_indices=noi_indices,
                max_iterations=100, abstol=1e-6, max_step=1.0
            )
        else:
            # Sparse path: compute COO→CSR mapping from trial run
            n_augmented = setup.n_unknowns + n_vsources

            # Trial run to get sparse structure
            X_trial = jnp.zeros(n_nodes + n_vsources, dtype=jnp.float64)
            vsource_trial = jnp.zeros(n_vsources, dtype=jnp.float64) if n_vsources > 0 else jnp.zeros(0, dtype=jnp.float64)
            isource_trial = jnp.zeros(0, dtype=jnp.float64)
            Q_trial = jnp.zeros(setup.n_unknowns, dtype=jnp.float64)

            J_bcoo_trial, _, _, _ = build_system_fn(
                X_trial, vsource_trial, isource_trial, Q_trial, 0.0,
                device_arrays, 1e-12, 0.0, 0.0, 0.0, None, 0.0, None
            )

            # Extract COO indices
            coo_rows = np.array(J_bcoo_trial.indices[:, 0])
            coo_cols = np.array(J_bcoo_trial.indices[:, 1])
            n_coo = len(coo_rows)

            # Compute linear indices for duplicate detection
            linear_idx = coo_rows * n_augmented + coo_cols

            # Sort COO by linear index (groups duplicates together)
            coo_sort_perm = np.argsort(linear_idx)
            sorted_linear = linear_idx[coo_sort_perm]
            sorted_rows = coo_rows[coo_sort_perm]

            # Find unique entries
            unique_linear, coo_to_unique = np.unique(sorted_linear, return_inverse=True)
            nse = len(unique_linear)

            # Convert sorted unique linear indices back to row/col
            unique_rows = unique_linear // n_augmented
            unique_cols = unique_linear % n_augmented

            # Build CSR indptr and indices
            csr_indptr = np.zeros(n_augmented + 1, dtype=np.int32)
            for row in unique_rows:
                csr_indptr[row + 1] += 1
            csr_indptr = np.cumsum(csr_indptr).astype(np.int32)
            csr_indices = unique_cols.astype(np.int32)

            # Segment IDs for summing duplicates
            csr_segment_ids = coo_to_unique.astype(np.int32)

            logger.info(f"Full MNA sparse: {n_coo} COO -> {nse} CSR entries")

            # Use UMFPACK if available (better performance with cached symbolic factorization)
            if is_umfpack_available():
                nr_solve = make_umfpack_full_mna_solver(
                    build_system_jit, n_nodes, n_vsources, nse,
                    bcsr_indptr=jnp.array(csr_indptr, dtype=jnp.int32),
                    bcsr_indices=jnp.array(csr_indices, dtype=jnp.int32),
                    noi_indices=noi_indices,
                    max_iterations=100, abstol=1e-6, max_step=1.0,
                    coo_sort_perm=jnp.array(coo_sort_perm, dtype=jnp.int32),
                    csr_segment_ids=jnp.array(csr_segment_ids, dtype=jnp.int32),
                )
            else:
                # Fallback to JAX's spsolve
                nr_solve = make_sparse_full_mna_solver(
                    build_system_jit, n_nodes, n_vsources, nse,
                    noi_indices=noi_indices,
                    max_iterations=100, abstol=1e-6, max_step=1.0,
                    coo_sort_perm=jnp.array(coo_sort_perm, dtype=jnp.int32),
                    csr_segment_ids=jnp.array(csr_segment_ids, dtype=jnp.int32),
                    bcsr_indices=jnp.array(csr_indices, dtype=jnp.int32),
                    bcsr_indptr=jnp.array(csr_indptr, dtype=jnp.int32),
                )

        self._cached_full_mna_solver = nr_solve
        self._cached_full_mna_key = cache_key
        logger.info(f"Created full MNA solver: V({n_nodes}) + I({n_vsources})")

        return nr_solve

    def _init_mid_rail(self, setup: TransientSetup, n_total: int) -> jax.Array:
        """Initialize voltage vector with mid-rail values.

        Provides a good starting point for DC convergence.

        Args:
            setup: TransientSetup with device info
            n_total: Total number of nodes

        Returns:
            V0: Initial voltage vector
        """
        vdd_value = self.runner._get_vdd_value()
        mid_rail = vdd_value / 2.0
        V = jnp.full(n_total, mid_rail, dtype=jnp.float64)
        V = V.at[0].set(0.0)  # Ground is always 0

        # Set VDD/GND nodes
        for name, idx in self.runner.node_names.items():
            name_lower = name.lower()
            if 'vdd' in name_lower or 'vcc' in name_lower:
                V = V.at[idx].set(vdd_value)
            elif name_lower in ('gnd', 'vss', '0'):
                V = V.at[idx].set(0.0)

        # Initialize voltage source nodes to target values
        for dev in self.runner.devices:
            if dev['model'] == 'vsource':
                nodes = dev.get('nodes', [])
                if len(nodes) >= 2:
                    p_node, n_node = nodes[0], nodes[1]
                    dc_val = float(dev['params'].get('dc', 0.0))
                    if n_node == 0 and p_node > 0:
                        V = V.at[p_node].set(dc_val)

        # Initialize NOI nodes to 0V
        if setup.device_internal_nodes:
            for dev_name, internal_nodes in setup.device_internal_nodes.items():
                if 'node4' in internal_nodes:
                    noi_idx = internal_nodes['node4']
                    V = V.at[noi_idx].set(0.0)

        return V

    def run(self, t_stop: float, dt: float,
            max_steps: int = 10000) -> Tuple[jax.Array, Dict[str, jax.Array], Dict]:
        """Run transient analysis with full MNA.

        Args:
            t_stop: Simulation stop time in seconds
            dt: Time step in seconds
            max_steps: Maximum number of time steps

        Returns:
            Tuple of (times, voltages, stats) where:
            - times: JAX array of time points
            - voltages: Dict mapping node name to voltage array
            - stats: Dict with statistics including 'currents' dict
        """
        # Ensure setup is ready
        setup = self.ensure_setup()
        nr_solve = self._ensure_full_mna_solver(setup)

        n_total = setup.n_total
        n_unknowns = setup.n_unknowns
        n_vsources = setup.n_branches
        n_external = setup.n_external
        source_fn = setup.source_fn
        source_device_data = setup.source_device_data

        # Compute number of timesteps
        num_timesteps = self._compute_num_timesteps(t_stop, dt)
        if num_timesteps > max_steps:
            num_timesteps = max_steps
            dt = t_stop / (max_steps - 1) if max_steps > 1 else t_stop
            logger.info(f"{self.name}: Limiting to {max_steps} steps, dt={dt:.2e}s")

        # Get integration method
        # NOTE: Full MNA with trapezoidal integration can exhibit numerical oscillation
        # (2*dt period) due to the algebraic branch equations. Force backward Euler
        # for stability unless a more sophisticated stabilization is implemented.
        tran_method = IntegrationMethod.BACKWARD_EULER
        integ_coeffs = compute_coefficients(tran_method, dt)
        logger.info(f"{self.name}: Using integration method: {tran_method.value} (forced for MNA stability)")

        # Initialize voltage vector with mid-rail values (good starting point for DC convergence)
        icmode = self.runner.analysis_params.get('icmode', 'op')
        V0 = self._init_mid_rail(setup, n_total)
        logger.info(f"{self.name}: Initialized with mid-rail values, icmode={icmode}")

        # Initialize augmented solution vector: [V; I_branch]
        X = jnp.zeros(n_total + n_vsources, dtype=jnp.float64)
        X = X.at[:n_total].set(V0)  # Use DC operating point for voltages

        # Initialize branch currents to 0 (will converge in first iteration)
        # Could estimate from DC, but 0 is safe starting point

        # Initialize charge state from DC operating point
        Q_prev = jnp.zeros(n_unknowns, dtype=jnp.float64)
        dQdt_prev = jnp.zeros(n_unknowns, dtype=jnp.float64) if integ_coeffs.needs_dqdt_history else None
        Q_prev2 = jnp.zeros(n_unknowns, dtype=jnp.float64) if integ_coeffs.history_depth >= 2 else None

        # Get device arrays
        device_arrays = self._device_arrays_full_mna

        # Result storage
        times_list = []
        voltages_dict = {i: [] for i in range(n_external)}
        currents_dict = {name: [] for name in setup.branch_data.name_to_idx.keys()} if setup.branch_data else {}

        total_nr_iters = 0
        non_converged_steps = []

        logger.info(f"{self.name}: Starting simulation ({num_timesteps} timesteps, "
                   f"{n_total} nodes, {n_vsources} vsources, full MNA)")
        t_start = time_module.perf_counter()

        # Compute DC operating point using Full MNA solver (c0=0) if icmode=='op'
        source_values = source_fn(0.0)
        vsource_vals_init, isource_vals_init = self._build_source_arrays(source_values)

        if icmode == 'op':
            # Solve DC operating point - this ensures consistent initial state
            X_dc, _, dc_converged, dc_residual, Q_dc, _, I_vsource_dc = nr_solve(
                X, vsource_vals_init, isource_vals_init, Q_prev, 0.0, device_arrays,  # c0=0 for DC
                1e-12, 0.0, integ_coeffs.c1, integ_coeffs.d1, dQdt_prev,
                integ_coeffs.c2, Q_prev2
            )

            if dc_converged:
                X = X_dc  # Use Full MNA DC solution as starting point
                Q_prev = Q_dc  # Update charge state
                logger.info(f"{self.name}: Full MNA DC converged, V[1]={float(X[1]):.4f}V")
            else:
                logger.warning(f"{self.name}: Full MNA DC did not converge (residual={dc_residual:.2e})")
        else:
            # UIC mode - use mid-rail initialization without DC solve
            logger.info(f"{self.name}: UIC mode - skipping DC solve, using initial conditions")
            I_vsource_dc = jnp.zeros(n_vsources, dtype=jnp.float64)

        # Record initial state at t=0
        times_list.append(0.0)
        for i in range(n_external):
            voltages_dict[i].append(float(X[i]))
        if setup.branch_data:
            for name, idx in setup.branch_data.name_to_idx.items():
                currents_dict[name].append(float(I_vsource_dc[idx]))

        for step_idx in range(1, num_timesteps):  # Start from step 1, not 0
            t = step_idx * dt

            # Evaluate sources at time t
            source_values = source_fn(t)
            vsource_vals, isource_vals = self._build_source_arrays(source_values)

            # Solve with full MNA
            X_new, iterations, converged, max_f, Q, dQdt, I_vsource = nr_solve(
                X, vsource_vals, isource_vals, Q_prev, integ_coeffs.c0, device_arrays,
                1e-12, 0.0,  # gmin, gshunt
                integ_coeffs.c1, integ_coeffs.d1, dQdt_prev,
                integ_coeffs.c2, Q_prev2
            )

            # Extract Python values for tracking
            nr_iters = int(iterations)
            is_converged = bool(converged)
            residual = float(max_f)

            X = X_new
            # Update charge history
            if integ_coeffs.history_depth >= 2:
                Q_prev2 = Q_prev
            Q_prev = Q
            dQdt_prev = dQdt if integ_coeffs.needs_dqdt_history else None
            total_nr_iters += nr_iters

            if not is_converged:
                non_converged_steps.append((t, residual))

            # Record state
            times_list.append(t)
            for i in range(n_external):
                voltages_dict[i].append(float(X[i]))

            # Record branch currents (directly from solution!)
            if setup.branch_data:
                for name, idx in setup.branch_data.name_to_idx.items():
                    currents_dict[name].append(float(I_vsource[idx]))

        wall_time = time_module.perf_counter() - t_start

        # Build results
        times = jnp.array(times_list)

        # Build voltage arrays with string names
        idx_to_voltage = {i: jnp.array(v) for i, v in voltages_dict.items()}
        voltages: Dict[str, jax.Array] = {}
        for name, idx in self.runner.node_names.items():
            if idx > 0 and idx < n_external:
                voltages[name] = idx_to_voltage[idx]

        # Build current arrays
        currents = {name: jnp.array(vals) for name, vals in currents_dict.items()}

        stats = {
            'total_timesteps': len(times_list),
            'total_nr_iterations': total_nr_iters,
            'avg_nr_iterations': total_nr_iters / max(len(times_list), 1),
            'non_converged_count': len(non_converged_steps),
            'non_converged_steps': non_converged_steps,
            'convergence_rate': 1.0 - len(non_converged_steps) / max(len(times_list), 1),
            'wall_time': wall_time,
            'time_per_step_ms': wall_time / len(times_list) * 1000 if times_list else 0,
            'strategy': 'full_mna',
            'solver': 'sparse' if self.use_sparse else 'dense',
            'integration_method': tran_method.value,
            'currents': currents,  # Branch currents directly from solution
        }

        logger.info(f"{self.name}: Completed {len(times_list)} steps in {wall_time:.3f}s "
                   f"({stats['time_per_step_ms']:.2f}ms/step, "
                   f"{total_nr_iters} NR iters, "
                   f"{len(non_converged_steps)} non-converged)")

        return times, voltages, stats
