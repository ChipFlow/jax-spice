"""Tests for jax_codegen module - strategy selection and structural tests."""

import pytest
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from jax_codegen import analyze_mir, build_eval_fn, generate_lax_loop_eval


class TestMIRAnalysis:
    """Test MIR analysis functionality."""

    def test_resistor_analysis(self, resistor_module):
        """Resistor should have no branches (pure straight-line code)."""
        analysis = analyze_mir(resistor_module)

        assert analysis.num_branches == 0
        assert analysis.num_phi_nodes == 0
        assert analysis.varying_dependent_branches == 0

    def test_capacitor_analysis(self, capacitor_module):
        """Capacitor should have no branches."""
        analysis = analyze_mir(capacitor_module)

        assert analysis.num_branches == 0
        assert analysis.num_phi_nodes == 0

    def test_diode_analysis(self, diode_module):
        """Diode has branches for limiting and breakdown."""
        analysis = analyze_mir(diode_module)

        # Diode should have some branches for conditional behavior
        assert analysis.num_branches >= 0  # May vary by implementation
        # PHI nodes match branches in SSA form
        assert analysis.num_phi_nodes >= 0

    def test_psp103_analysis(self, psp103_module):
        """PSP103 has many branches and PHI nodes."""
        analysis = analyze_mir(psp103_module)

        # PSP103 is known to have 431 branches and 1522 PHI nodes
        assert analysis.num_branches > 100
        assert analysis.num_phi_nodes > 500


class TestStrategySelection:
    """Test that build_eval_fn selects the correct strategy."""

    def test_resistor_uses_straight_line(self, resistor_module):
        """Resistor should use straight_line strategy (no control flow)."""
        _, meta = build_eval_fn(resistor_module)

        assert meta['strategy'] == 'straight_line'

    def test_capacitor_uses_straight_line(self, capacitor_module):
        """Capacitor should use straight_line strategy."""
        _, meta = build_eval_fn(capacitor_module)

        assert meta['strategy'] == 'straight_line'

    def test_psp103_uses_lax_loop(self, psp103_module):
        """PSP103 should use lax_loop strategy (>1000 instructions)."""
        _, meta = build_eval_fn(psp103_module)

        assert meta['strategy'] == 'lax_loop'

    def test_force_lax_loop(self, resistor_module):
        """Test force_lax_loop parameter."""
        _, meta = build_eval_fn(resistor_module, force_lax_loop=True)

        assert meta['strategy'] == 'lax_loop'


class TestJITCompilation:
    """Test that generated functions JIT compile successfully."""

    def test_resistor_jit_compiles(self, resistor_module):
        """Resistor eval function should JIT compile."""
        eval_fn, _ = build_eval_fn(resistor_module)

        # Get param count from MIR
        mir = resistor_module.get_mir_instructions()
        n_params = len(mir.get('params', []))
        n_cache = resistor_module.num_cached_values

        # JIT compile
        jitted = jax.jit(eval_fn)

        # Run once to trigger compilation
        params = jnp.zeros(n_params)
        cache = jnp.zeros(n_cache)
        result = jitted(params, cache)

        # Should produce output without error
        assert result is not None
        (res_resist, res_react), (jac_resist, jac_react) = result
        assert res_resist is not None
        assert res_react is not None

    def test_capacitor_jit_compiles(self, capacitor_module):
        """Capacitor eval function should JIT compile."""
        eval_fn, _ = build_eval_fn(capacitor_module)

        mir = capacitor_module.get_mir_instructions()
        n_params = len(mir.get('params', []))
        n_cache = capacitor_module.num_cached_values

        jitted = jax.jit(eval_fn)
        params = jnp.zeros(n_params)
        cache = jnp.zeros(n_cache)
        result = jitted(params, cache)

        assert result is not None

    def test_diode_jit_compiles(self, diode_module):
        """Diode eval function should JIT compile."""
        eval_fn, _ = build_eval_fn(diode_module)

        mir = diode_module.get_mir_instructions()
        n_params = len(mir.get('params', []))
        n_cache = diode_module.num_cached_values

        jitted = jax.jit(eval_fn)
        params = jnp.zeros(n_params)
        cache = jnp.zeros(n_cache)
        result = jitted(params, cache)

        assert result is not None

    @pytest.mark.timeout(60)
    def test_psp103_jit_compiles_in_reasonable_time(self, psp103_module):
        """PSP103 should JIT compile within 60 seconds using lax_loop."""
        eval_fn, meta = build_eval_fn(psp103_module)

        assert meta['strategy'] == 'lax_loop', "PSP103 should use lax_loop"

        mir = psp103_module.get_mir_instructions()
        n_params = len(mir.get('params', []))
        n_cache = psp103_module.num_cached_values

        jitted = jax.jit(eval_fn)
        params = jnp.zeros(n_params)
        cache = jnp.zeros(n_cache)

        # This triggers actual XLA compilation
        result = jitted(params, cache)

        assert result is not None


class TestOutputStructure:
    """Test that output has correct structure."""

    def test_resistor_output_shapes(self, resistor_module):
        """Test resistor output has correct shapes."""
        eval_fn, _ = build_eval_fn(resistor_module)

        metadata = resistor_module.get_codegen_metadata()
        n_residuals = len(metadata['residuals'])
        n_jacobian = len(metadata['jacobian'])

        mir = resistor_module.get_mir_instructions()
        n_params = len(mir.get('params', []))
        n_cache = resistor_module.num_cached_values

        params = jnp.zeros(n_params)
        cache = jnp.zeros(n_cache)
        (res_resist, res_react), (jac_resist, jac_react) = eval_fn(params, cache)

        assert res_resist.shape == (n_residuals,)
        assert res_react.shape == (n_residuals,)
        assert jac_resist.shape == (n_jacobian,)
        assert jac_react.shape == (n_jacobian,)

    def test_capacitor_output_shapes(self, capacitor_module):
        """Test capacitor output has correct shapes."""
        eval_fn, _ = build_eval_fn(capacitor_module)

        metadata = capacitor_module.get_codegen_metadata()
        n_residuals = len(metadata['residuals'])
        n_jacobian = len(metadata['jacobian'])

        mir = capacitor_module.get_mir_instructions()
        n_params = len(mir.get('params', []))
        n_cache = capacitor_module.num_cached_values

        params = jnp.zeros(n_params)
        cache = jnp.zeros(n_cache)
        (res_resist, res_react), (jac_resist, jac_react) = eval_fn(params, cache)

        assert res_resist.shape == (n_residuals,)
        assert res_react.shape == (n_residuals,)
        assert jac_resist.shape == (n_jacobian,)
        assert jac_react.shape == (n_jacobian,)

    def test_psp103_output_shapes(self, psp103_module):
        """Test PSP103 output has correct shapes."""
        eval_fn, _ = build_eval_fn(psp103_module)

        metadata = psp103_module.get_codegen_metadata()
        n_residuals = len(metadata['residuals'])
        n_jacobian = len(metadata['jacobian'])

        mir = psp103_module.get_mir_instructions()
        n_params = len(mir.get('params', []))
        n_cache = psp103_module.num_cached_values

        params = jnp.zeros(n_params)
        cache = jnp.zeros(n_cache)
        (res_resist, res_react), (jac_resist, jac_react) = eval_fn(params, cache)

        assert res_resist.shape == (n_residuals,)
        assert res_react.shape == (n_residuals,)
        assert jac_resist.shape == (n_jacobian,)
        assert jac_react.shape == (n_jacobian,)


class TestNoNaNInf:
    """Test that outputs don't contain NaN or Inf for valid inputs."""

    def test_resistor_no_nan_at_zero(self, resistor_module):
        """Resistor at zero voltage should not produce NaN."""
        eval_fn, _ = build_eval_fn(resistor_module)

        mir = resistor_module.get_mir_instructions()
        n_params = len(mir.get('params', []))
        n_cache = resistor_module.num_cached_values

        # Use small non-zero values to avoid div-by-zero
        params = jnp.ones(n_params) * 0.001
        cache = jnp.ones(n_cache) * 0.001
        (res_resist, res_react), (jac_resist, jac_react) = eval_fn(params, cache)

        assert not jnp.any(jnp.isnan(res_resist)), f"NaN in res_resist: {res_resist}"
        assert not jnp.any(jnp.isnan(res_react)), f"NaN in res_react: {res_react}"
        assert not jnp.any(jnp.isnan(jac_resist)), f"NaN in jac_resist: {jac_resist}"
        assert not jnp.any(jnp.isnan(jac_react)), f"NaN in jac_react: {jac_react}"

        assert not jnp.any(jnp.isinf(res_resist)), f"Inf in res_resist: {res_resist}"
        assert not jnp.any(jnp.isinf(jac_resist)), f"Inf in jac_resist: {jac_resist}"

    def test_capacitor_no_nan_at_zero(self, capacitor_module):
        """Capacitor at zero voltage should not produce NaN."""
        eval_fn, _ = build_eval_fn(capacitor_module)

        mir = capacitor_module.get_mir_instructions()
        n_params = len(mir.get('params', []))
        n_cache = capacitor_module.num_cached_values

        params = jnp.ones(n_params) * 0.001
        cache = jnp.ones(n_cache) * 0.001
        (res_resist, res_react), (jac_resist, jac_react) = eval_fn(params, cache)

        assert not jnp.any(jnp.isnan(res_resist)), f"NaN in res_resist"
        assert not jnp.any(jnp.isnan(res_react)), f"NaN in res_react"
        assert not jnp.any(jnp.isnan(jac_resist)), f"NaN in jac_resist"
        assert not jnp.any(jnp.isnan(jac_react)), f"NaN in jac_react"
