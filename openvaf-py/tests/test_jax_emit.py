"""Tests for JAX emitter."""

import pytest
import jax
import jax.numpy as jnp
import openvaf_py
from pathlib import Path

# Enable float64 for numerical accuracy in tests
jax.config.update('jax_enable_x64', True)

REPO_ROOT = Path(__file__).parent.parent.parent
VACASK = REPO_ROOT / "vendor" / "VACASK" / "devices"
OPENVAF_TESTS = REPO_ROOT / "vendor" / "OpenVAF" / "integration_tests"


@pytest.fixture(scope="module")
def resistor_module():
    """Load resistor VA model."""
    modules = openvaf_py.compile_va(str(VACASK / "resistor.va"))
    return modules[0]


@pytest.fixture(scope="module")
def diode_module():
    """Load diode VA model."""
    modules = openvaf_py.compile_va(str(VACASK / "diode.va"))
    return modules[0]


@pytest.fixture(scope="module")
def psp103_module():
    """Load PSP103 VA model."""
    modules = openvaf_py.compile_va(str(OPENVAF_TESTS / "PSP103" / "psp103.va"))
    return modules[0]


class TestEmitContext:
    """Tests for EmitContext building."""

    def test_resistor_context(self, resistor_module):
        """Test context building for resistor."""
        from jax_emit import _build_emit_context

        ctx = _build_emit_context(resistor_module)

        # Should have reasonable number of variables
        assert ctx.n_vars > 10
        assert ctx.n_blocks >= 1
        assert ctx.n_params > 0

        # Output indices should be populated
        assert len(ctx.resist_res_indices) > 0
        assert len(ctx.resist_jac_indices) > 0

    def test_diode_context(self, diode_module):
        """Test context building for diode with control flow."""
        from jax_emit import _build_emit_context

        ctx = _build_emit_context(diode_module)

        # Diode has multiple blocks due to control flow
        assert ctx.n_blocks > 1
        assert ctx.n_vars > 50


class TestCompiledInstructions:
    """Tests for instruction compilation."""

    def test_resistor_compile(self, resistor_module):
        """Test instruction compilation for resistor."""
        from jax_emit import _build_emit_context, _compile_instructions

        ctx = _build_emit_context(resistor_module)
        compiled = _compile_instructions(resistor_module, ctx)

        assert len(compiled) > 0

        # Resistor should not have phi or branch instructions
        opcodes = [inst.opcode for inst in compiled]
        assert 'phi' not in opcodes
        assert 'br' not in opcodes

    def test_diode_compile(self, diode_module):
        """Test instruction compilation for diode with control flow."""
        from jax_emit import _build_emit_context, _compile_instructions

        ctx = _build_emit_context(diode_module)
        compiled = _compile_instructions(diode_module, ctx)

        # Diode should have phi nodes and branches
        opcodes = [inst.opcode for inst in compiled]
        assert 'phi' in opcodes
        assert 'br' in opcodes

        # Check phi instructions have operands
        phi_insts = [inst for inst in compiled if inst.opcode == 'phi']
        assert len(phi_insts) > 0
        for phi in phi_insts:
            assert len(phi.phi_operands) > 0

        # Check branch instructions have info
        br_insts = [inst for inst in compiled if inst.opcode == 'br']
        assert len(br_insts) > 0
        for br in br_insts:
            assert br.branch_info is not None


class TestEmitEval:
    """Tests for emit_eval function."""

    def test_resistor_emit(self, resistor_module):
        """Test emit_eval for resistor produces callable."""
        from jax_emit import emit_eval

        eval_fn = emit_eval(resistor_module)
        assert callable(eval_fn)

        # Should be JIT-compilable
        jit_fn = jax.jit(eval_fn)

        n_params = len(list(resistor_module.param_names))
        n_cache = resistor_module.num_cached_values
        params = jnp.zeros(n_params)
        cache = jnp.zeros(max(n_cache, 1))

        result = jit_fn(params, cache)

        # Should return ((resist_res, react_res), (resist_jac, react_jac))
        assert len(result) == 2
        (resist_res, react_res), (resist_jac, react_jac) = result
        assert resist_res.shape[0] > 0
        assert resist_jac.shape[0] > 0

    def test_diode_emit(self, diode_module):
        """Test emit_eval for diode with control flow."""
        from jax_emit import emit_eval

        eval_fn = emit_eval(diode_module)
        jit_fn = jax.jit(eval_fn)

        n_params = len(list(diode_module.param_names))
        n_cache = diode_module.num_cached_values
        params = jnp.zeros(n_params)
        cache = jnp.zeros(max(n_cache, 1))

        result = jit_fn(params, cache)

        (resist_res, react_res), (resist_jac, react_jac) = result
        assert resist_res.shape[0] > 0
        assert resist_jac.shape[0] > 0

        # Results should be finite for zero inputs
        assert jnp.all(jnp.isfinite(resist_res))
        assert jnp.all(jnp.isfinite(resist_jac))

    def test_diode_nonzero_voltage(self, diode_module):
        """Test diode with non-zero voltage to exercise branches."""
        from jax_emit import emit_eval

        eval_fn = emit_eval(diode_module)
        jit_fn = jax.jit(eval_fn)

        # Get param info
        param_kinds = list(diode_module.param_kinds)
        n_params = len(param_kinds)
        n_cache = diode_module.num_cached_values

        # Set up params with a small forward bias
        params = jnp.zeros(n_params)

        # Find voltage parameters and set them
        for i, kind in enumerate(param_kinds):
            if kind == 'voltage':
                params = params.at[i].set(0.3)  # 0.3V forward bias
                break

        cache = jnp.zeros(max(n_cache, 1))

        result = jit_fn(params, cache)
        (resist_res, react_res), (resist_jac, react_jac) = result

        # Just verify it runs without error and produces finite results
        # (Non-zero results require proper cache initialization from init phase)
        assert jnp.all(jnp.isfinite(resist_res))
        assert jnp.all(jnp.isfinite(resist_jac))


class TestEmitEvalLaxLoop:
    """Tests for lax_loop based emission for large models."""

    def test_psp103_emit(self, psp103_module):
        """Test emit for PSP103 (large model)."""
        from jax_emit import build_eval_fn

        eval_fn, meta = build_eval_fn(psp103_module)

        # Should use lax_loop strategy for large model
        assert meta['strategy'] == 'lax_loop'
        assert meta['n_instructions'] > 10000

        # Should be JIT-compilable
        jit_fn = jax.jit(eval_fn)

        n_params = len(list(psp103_module.param_names))
        n_cache = psp103_module.num_cached_values
        params = jnp.zeros(n_params)
        cache = jnp.zeros(n_cache)

        result = jit_fn(params, cache)

        (resist_res, react_res), (resist_jac, react_jac) = result

        # PSP103 should produce 13 residuals and 56 jacobian entries
        assert resist_res.shape[0] == 13
        assert resist_jac.shape[0] == 56

    def test_force_lax_loop_on_small_model(self, resistor_module):
        """Test forcing lax_loop on small model."""
        from jax_emit import build_eval_fn

        eval_fn, meta = build_eval_fn(resistor_module, force_lax_loop=True)

        assert meta['strategy'] == 'lax_loop'

        jit_fn = jax.jit(eval_fn)

        n_params = len(list(resistor_module.param_names))
        n_cache = resistor_module.num_cached_values
        params = jnp.zeros(n_params)
        cache = jnp.zeros(max(n_cache, 1))

        result = jit_fn(params, cache)
        (resist_res, react_res), (resist_jac, react_jac) = result

        assert resist_res.shape[0] > 0


class TestBuildEvalFn:
    """Tests for the build_eval_fn dispatcher."""

    def test_strategy_selection_small(self, resistor_module):
        """Test strategy selection for small model."""
        from jax_emit import build_eval_fn

        _, meta = build_eval_fn(resistor_module)
        assert meta['strategy'] == 'branchless'

    def test_strategy_selection_medium(self, diode_module):
        """Test strategy selection for medium model."""
        from jax_emit import build_eval_fn

        _, meta = build_eval_fn(diode_module)
        assert meta['strategy'] == 'branchless'

    def test_strategy_selection_large(self, psp103_module):
        """Test strategy selection for large model."""
        from jax_emit import build_eval_fn

        _, meta = build_eval_fn(psp103_module)
        assert meta['strategy'] == 'lax_loop'


class TestEmitModelParamSetup:
    """Tests for emit_model_param_setup."""

    def test_resistor_model_param_setup(self, resistor_module):
        """Test model param setup for resistor."""
        from jax_emit import emit_model_param_setup

        setup_fn = emit_model_param_setup(resistor_module)
        jit_setup = jax.jit(setup_fn)

        # r=1000, has_noise=0, both given
        param_values = jnp.array([1000.0, 0.0])
        param_given = jnp.array([1.0, 1.0])

        result = jit_setup(param_values, param_given)
        assert result.shape[0] == 2
        assert jnp.all(jnp.isfinite(result))


class TestEmitInit:
    """Tests for emit_init."""

    def test_resistor_init(self, resistor_module):
        """Test init for resistor."""
        from jax_emit import emit_init

        init_fn = emit_init(resistor_module)
        jit_init = jax.jit(init_fn)

        # Init params: r, has_noise, temperature, mfactor, r_given
        init_params = jnp.array([1000.0, 0.0, 300.0, 1.0, 1.0])
        cache = jit_init(init_params)

        assert cache.shape[0] == resistor_module.num_cached_values
        assert jnp.all(jnp.isfinite(cache))

        # For R=1000, G=1/R=0.001 should be in cache
        assert jnp.any(jnp.abs(cache - 0.001) < 1e-10)

    def test_diode_init(self, diode_module):
        """Test init for diode."""
        from jax_emit import emit_init

        init_fn = emit_init(diode_module)
        jit_init = jax.jit(init_fn)

        n_init = len(list(diode_module.init_param_names))
        init_params = jnp.zeros(n_init)

        # Set temperature and mfactor
        init_kinds = list(diode_module.init_param_kinds)
        for i, kind in enumerate(init_kinds):
            if kind == 'temperature':
                init_params = init_params.at[i].set(300.0)
            elif kind == 'sysfun':
                init_params = init_params.at[i].set(1.0)

        cache = jit_init(init_params)
        assert cache.shape[0] == diode_module.num_cached_values
        # Some cache values may be inf/-inf for edge cases, but most should be finite
        finite_count = jnp.sum(jnp.isfinite(cache))
        assert finite_count > cache.shape[0] // 2


class TestFullPipeline:
    """Tests for the full setup -> eval pipeline."""

    def test_resistor_pipeline(self, resistor_module):
        """Test resistor: init -> eval."""
        from jax_emit import emit_init, build_eval_fn

        init_fn = emit_init(resistor_module)
        eval_fn, _ = build_eval_fn(resistor_module)

        jit_init = jax.jit(init_fn)
        jit_eval = jax.jit(eval_fn)

        # Init with R=1000
        init_params = jnp.array([1000.0, 0.0, 300.0, 1.0, 1.0])
        cache = jit_init(init_params)

        # Eval with V=1V
        eval_params = jnp.array([1.0, 1000.0, 0.0, 0.0, 300.0, 1.0])
        (resist_res, _), (resist_jac, _) = jit_eval(eval_params, cache)

        # Check Ohm's law: I = V/R = 1/1000 = 0.001
        assert jnp.abs(resist_res[0] - 0.001) < 1e-10

    def test_diode_pipeline(self, diode_module):
        """Test diode: init -> eval."""
        from jax_emit import emit_init, build_eval_fn

        init_fn = emit_init(diode_module)
        eval_fn, _ = build_eval_fn(diode_module)

        jit_init = jax.jit(init_fn)
        jit_eval = jax.jit(eval_fn)

        # Init
        n_init = len(list(diode_module.init_param_names))
        init_params = jnp.zeros(n_init)
        init_kinds = list(diode_module.init_param_kinds)
        for i, kind in enumerate(init_kinds):
            if kind == 'temperature':
                init_params = init_params.at[i].set(300.0)
            elif kind == 'sysfun':
                init_params = init_params.at[i].set(1.0)

        cache = jit_init(init_params)

        # Eval with forward bias
        n_eval = len(list(diode_module.param_names))
        eval_params = jnp.zeros(n_eval)
        eval_kinds = list(diode_module.param_kinds)
        for i, kind in enumerate(eval_kinds):
            if kind == 'voltage':
                eval_params = eval_params.at[i].set(0.6)
            elif kind == 'temperature':
                eval_params = eval_params.at[i].set(300.0)
            elif kind == 'sysfun':
                eval_params = eval_params.at[i].set(1.0)

        (resist_res, _), (resist_jac, _) = jit_eval(eval_params, cache)

        # Should produce some current
        assert jnp.all(jnp.isfinite(resist_res))
        assert jnp.all(jnp.isfinite(resist_jac))


class TestSafeMath:
    """Tests for safe math operations."""

    def test_safe_div_zero(self):
        """Test safe_div handles 0/0."""
        from jax_emit import safe_div

        result = safe_div(jnp.array(0.0), jnp.array(0.0))
        assert result == 0.0

    def test_safe_div_inf(self):
        """Test safe_div handles 1/0."""
        from jax_emit import safe_div

        result = safe_div(jnp.array(1.0), jnp.array(0.0))
        assert jnp.isinf(result)
        assert result > 0

    def test_safe_add_indeterminate(self):
        """Test safe_add handles inf + (-inf)."""
        from jax_emit import safe_add

        result = safe_add(jnp.array(jnp.inf), jnp.array(-jnp.inf))
        assert result == 0.0

    def test_safe_mul_zero_inf(self):
        """Test safe_mul handles 0 * inf."""
        from jax_emit import safe_mul

        result = safe_mul(jnp.array(0.0), jnp.array(jnp.inf))
        assert result == 0.0

    def test_safe_ln_negative(self):
        """Test safe_ln handles negative input."""
        from jax_emit import safe_ln

        result = safe_ln(jnp.array(-1.0))
        # Should not produce NaN
        assert not jnp.isnan(result)

    def test_safe_sqrt_negative(self):
        """Test safe_sqrt handles negative input."""
        from jax_emit import safe_sqrt

        result = safe_sqrt(jnp.array(-1.0))
        assert result == 0.0
