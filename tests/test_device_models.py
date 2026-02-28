"""Smoke tests for VACASK device model compilation and evaluation.

Verifies that each Verilog-A device model in vendor/VACASK/devices/ can be:
1. Compiled with OpenVAF
2. Translated to JAX init/eval functions via openvaf_jax
3. Evaluated with default parameters without errors
"""

import os

os.environ["JAX_ENABLE_X64"] = "true"

from pathlib import Path

import jax.numpy as jnp
import openvaf_py
import pytest

import openvaf_jax
from vajax import build_simparams

PROJECT_ROOT = Path(__file__).parent.parent
VACASK_DEVICES = PROJECT_ROOT / "vendor" / "VACASK" / "devices"


def get_va_models():
    """Discover all .va files in VACASK devices directory."""
    if not VACASK_DEVICES.exists():
        return []
    return sorted(VACASK_DEVICES.glob("*.va"))


VA_MODELS = get_va_models()


@pytest.mark.parametrize(
    "va_path",
    VA_MODELS,
    ids=[p.stem for p in VA_MODELS],
)
class TestDeviceModelCompilation:
    """Test that each device model compiles and translates successfully."""

    def test_compile_and_init(self, va_path):
        """Model compiles with OpenVAF and init function runs."""
        modules = openvaf_py.compile_va(str(va_path))
        assert len(modules) >= 1, f"Expected at least 1 module from {va_path.name}"

        translator = openvaf_jax.OpenVAFToJAX(modules[0])
        init_fn, init_meta = translator.translate_init(params={}, temperature=300.0)

        assert callable(init_fn)
        assert "param_names" in init_meta
        assert "param_kinds" in init_meta
        assert "init_inputs" in init_meta

        # Run init function
        init_inputs = jnp.array(init_meta["init_inputs"])
        cache, collapse = init_fn(init_inputs)
        assert cache is not None

    def test_translate_eval(self, va_path):
        """Model translates to JAX eval function and runs without errors."""
        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        # Run init to get cache
        init_fn, init_meta = translator.translate_init(params={}, temperature=300.0)
        init_inputs = jnp.array(init_meta["init_inputs"])
        cache, _ = init_fn(init_inputs)

        # Translate eval
        eval_fn, eval_meta = translator.translate_eval(params={}, temperature=300.0)
        assert callable(eval_fn)
        assert "param_names" in eval_meta
        assert "param_kinds" in eval_meta
        assert "shared_inputs" in eval_meta
        assert "voltage_indices" in eval_meta
        assert "node_names" in eval_meta

        # Build inputs following the eval_fn signature:
        # (shared_params, device_params, shared_cache, device_cache,
        #  simparams, limit_state_in, limit_funcs)
        shared_inputs = jnp.array(eval_meta["shared_inputs"])

        # Build voltage inputs with small values
        voltage_inputs = jnp.full(len(eval_meta["voltage_indices"]), 0.1)

        simparams = jnp.array(build_simparams(eval_meta))
        shared_cache = jnp.array([])
        limit_state_in = jnp.array([])
        limit_funcs = {}

        result = eval_fn(
            shared_inputs,
            voltage_inputs,
            shared_cache,
            cache,
            simparams,
            limit_state_in,
            limit_funcs,
        )
        assert result is not None
        # result[0] is residuals, result[1] is reactive residuals
        assert len(result) >= 2


class TestResistorEval:
    """Verify resistor produces expected current with full init+eval flow."""

    def test_ohms_law(self):
        va_path = VACASK_DEVICES / "resistor.va"
        if not va_path.exists():
            pytest.skip("resistor.va not found")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        # R=1k
        init_fn, init_meta = translator.translate_init(params={"R": 1000.0}, temperature=300.0)
        init_inputs = jnp.array(init_meta["init_inputs"])
        cache, _ = init_fn(init_inputs)

        eval_fn, eval_meta = translator.translate_eval(params={"R": 1000.0}, temperature=300.0)

        shared_inputs = jnp.array(eval_meta["shared_inputs"])
        # 1V across resistor terminals
        voltage_inputs = jnp.array([1.0])
        simparams = jnp.array(build_simparams(eval_meta))

        result = eval_fn(
            shared_inputs,
            voltage_inputs,
            jnp.array([]),
            cache,
            simparams,
            jnp.array([]),
            {},
        )
        res_resist = result[0]
        assert len(res_resist) >= 2
        # Should have non-zero residual for V=1V, R=1k
        assert jnp.any(jnp.abs(res_resist) > 1e-12)


class TestDiodeEval:
    """Verify diode model evaluates under forward bias."""

    def test_forward_bias(self):
        va_path = VACASK_DEVICES / "diode.va"
        if not va_path.exists():
            pytest.skip("diode.va not found")

        modules = openvaf_py.compile_va(str(va_path))
        translator = openvaf_jax.OpenVAFToJAX(modules[0])

        init_fn, init_meta = translator.translate_init(params={}, temperature=300.0)
        init_inputs = jnp.array(init_meta["init_inputs"])
        cache, _ = init_fn(init_inputs)

        eval_fn, eval_meta = translator.translate_eval(params={}, temperature=300.0)

        shared_inputs = jnp.array(eval_meta["shared_inputs"])
        # 0.7V forward bias
        n_voltages = len(eval_meta["voltage_indices"])
        voltage_inputs = jnp.zeros(n_voltages)
        voltage_inputs = voltage_inputs.at[0].set(0.7)  # Forward bias

        simparams = jnp.array(build_simparams(eval_meta))

        result = eval_fn(
            shared_inputs,
            voltage_inputs,
            jnp.array([]),
            cache,
            simparams,
            jnp.array([]),
            {},
        )
        res_resist = result[0]
        # Diode at 0.7V forward bias should have significant current
        assert jnp.any(jnp.abs(res_resist) > 1e-6), "Diode should conduct at 0.7V forward bias"
