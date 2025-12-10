"""Tests comparing JAX function output against MIR interpreter

This module validates that the JAX-compiled models produce the same
results as the reference MIR interpreter.
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple

from conftest import (
    INTEGRATION_PATH, LOCAL_MODELS_PATH, INTEGRATION_MODELS, assert_allclose,
    CompiledModel, build_param_dict
)


class TestJaxVsInterpreter:
    """Compare JAX function output against MIR interpreter for all models"""

    @pytest.mark.parametrize("model_name,model_path", INTEGRATION_MODELS)
    def test_model_compiles(self, compile_model, model_name, model_path):
        """Model compiles to JAX without error"""
        model = compile_model(INTEGRATION_PATH / model_path)
        assert model.module is not None
        assert model.jax_fn is not None
        assert model.name, f"{model_name} has no module name"

    # Simple models that work with the current JAX translator
    SIMPLE_MODELS = ['resistor', 'diode', 'isrc', 'vccs', 'cccs']

    @pytest.mark.parametrize("model_name,model_path", [
        m for m in INTEGRATION_MODELS if m[0] in ['resistor', 'diode', 'isrc', 'vccs', 'cccs']
    ])
    def test_simple_model_produces_valid_output(self, compile_model, model_name, model_path):
        """Simple JAX function produces non-NaN outputs"""
        model = compile_model(INTEGRATION_PATH / model_path)
        inputs = model.build_default_inputs()

        residuals, jacobian = model.jax_fn(inputs)

        # Check residuals are valid
        assert residuals is not None, f"{model_name} returned None residuals"
        for node, res in residuals.items():
            resist = float(res['resist'])
            react = float(res['react'])
            assert not np.isnan(resist), f"{model_name} NaN resist at {node}"
            assert not np.isnan(react), f"{model_name} NaN react at {node}"

    @pytest.mark.parametrize("model_name,model_path", [
        m for m in INTEGRATION_MODELS if m[0] not in ['resistor', 'diode', 'isrc', 'vccs', 'cccs']
    ])
    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_complex_model_produces_valid_output(self, compile_model, model_name, model_path):
        """Complex JAX function produces non-NaN outputs"""
        model = compile_model(INTEGRATION_PATH / model_path)
        inputs = model.build_default_inputs()

        residuals, jacobian = model.jax_fn(inputs)

        # Check residuals are valid
        assert residuals is not None, f"{model_name} returned None residuals"
        for node, res in residuals.items():
            resist = float(res['resist'])
            react = float(res['react'])
            assert not np.isnan(resist), f"{model_name} NaN resist at {node}"
            assert not np.isnan(react), f"{model_name} NaN react at {node}"


class TestResistorJaxInterpreter:
    """Detailed comparison for resistor model"""

    @pytest.mark.parametrize("voltage,resistance,temperature,tnom,zeta,mfactor", [
        (1.0, 1000.0, 300.0, 300.0, 0.0, 1.0),
        (5.0, 1000.0, 300.0, 300.0, 0.0, 1.0),
        (1.0, 470.0, 300.0, 300.0, 0.0, 1.0),
        (1.0, 1000.0, 350.0, 300.0, 1.0, 1.0),
        (1.0, 1000.0, 350.0, 300.0, 2.0, 1.0),
        (1.0, 1000.0, 300.0, 300.0, 0.0, 2.0),
        (1.0, 1000.0, 250.0, 300.0, 1.0, 1.0),
        (0.1, 100.0, 273.15, 300.0, 1.5, 0.5),
    ])
    def test_resistor_residuals_match(
        self, resistor_model: CompiledModel,
        voltage, resistance, temperature, tnom, zeta, mfactor
    ):
        """Compare JAX vs interpreter residuals for resistor"""
        # Build inputs for JAX
        jax_inputs = [voltage, voltage, resistance, temperature, tnom, zeta, resistance, mfactor]

        # Build params for interpreter
        interp_params = {
            'V(A,B)': voltage,
            'vres': voltage,
            'R': resistance,
            '$temperature': temperature,
            'tnom': tnom,
            'zeta': zeta,
            'res': resistance,
            'mfactor': mfactor,
        }

        # Run both
        jax_residuals, jax_jacobian = resistor_model.jax_fn(jax_inputs)
        interp_residuals, interp_jacobian = resistor_model.module.run_init_eval(interp_params)

        # Compare residual
        jax_I = float(jax_residuals['sim_node0']['resist'])
        interp_I = interp_residuals[0][0]  # First node, resist component

        assert_allclose(
            jax_I, interp_I,
            rtol=1e-6, atol=1e-15,
            err_msg=f"Resistor residual mismatch: V={voltage}, R={resistance}"
        )

    @pytest.mark.parametrize("voltage,resistance,temperature,tnom,zeta,mfactor", [
        (1.0, 1000.0, 300.0, 300.0, 0.0, 1.0),
        (5.0, 100.0, 300.0, 300.0, 0.0, 1.0),
        (1.0, 1000.0, 350.0, 300.0, 1.0, 1.0),
    ])
    def test_resistor_jacobian_match(
        self, resistor_model: CompiledModel,
        voltage, resistance, temperature, tnom, zeta, mfactor
    ):
        """Compare JAX vs interpreter jacobian for resistor"""
        jax_inputs = [voltage, voltage, resistance, temperature, tnom, zeta, resistance, mfactor]

        interp_params = {
            'V(A,B)': voltage,
            'vres': voltage,
            'R': resistance,
            '$temperature': temperature,
            'tnom': tnom,
            'zeta': zeta,
            'res': resistance,
            'mfactor': mfactor,
        }

        jax_residuals, jax_jacobian = resistor_model.jax_fn(jax_inputs)
        interp_residuals, interp_jacobian = resistor_model.module.run_init_eval(interp_params)

        # Compare jacobian entry (0,0)
        jax_jac_00 = float(jax_jacobian[('sim_node0', 'sim_node0')]['resist'])
        interp_jac_00 = interp_jacobian[0][2]  # row=0, col=0, resist component

        assert_allclose(
            jax_jac_00, interp_jac_00,
            rtol=1e-5, atol=1e-12,
            err_msg=f"Resistor jacobian mismatch: R={resistance}"
        )


class TestModelNodeCounts:
    """Verify models have reasonable node counts"""

    @pytest.mark.parametrize("model_name,model_path", INTEGRATION_MODELS)
    def test_has_nodes(self, compile_model, model_name, model_path):
        """Model has at least 2 nodes"""
        model = compile_model(INTEGRATION_PATH / model_path)
        assert len(model.nodes) >= 2, f"{model_name} should have at least 2 nodes"

    @pytest.mark.parametrize("model_name,model_path", [
        m for m in INTEGRATION_MODELS if m[0] in ['resistor', 'isrc', 'juncap']
    ])
    def test_two_terminal_devices(self, compile_model, model_name, model_path):
        """Two-terminal devices have 2 nodes"""
        model = compile_model(INTEGRATION_PATH / model_path)
        assert len(model.nodes) == 2, f"{model_name} should be a two-terminal device"

    @pytest.mark.parametrize("model_name,model_path", [
        m for m in INTEGRATION_MODELS if m[0] in ['vccs', 'cccs']
    ])
    def test_four_terminal_devices(self, compile_model, model_name, model_path):
        """Controlled sources have at least 4 terminals"""
        model = compile_model(INTEGRATION_PATH / model_path)
        assert len(model.nodes) >= 4, f"{model_name} should have at least 4 terminals"


class TestModelComplexity:
    """Test that complex models compile and produce outputs"""

    @pytest.mark.parametrize("model_name,model_path", [
        m for m in INTEGRATION_MODELS
        if m[0] in ('bsim4', 'psp103', 'hisim2', 'hicum', 'mextram')
    ])
    def test_complex_model_compiles(self, compile_model, model_name, model_path):
        """Complex model compiles to JAX"""
        model = compile_model(INTEGRATION_PATH / model_path)
        assert model.jax_fn is not None
        assert model.module.num_jacobian > 0

    @pytest.mark.parametrize("model_name,model_path", [
        m for m in INTEGRATION_MODELS
        if m[0] in ('bsim4', 'psp103', 'hisim2', 'hicum', 'mextram')
    ])
    @pytest.mark.xfail(reason="JAX translator has init variable issues for complex models")
    def test_complex_model_outputs(self, compile_model, model_name, model_path):
        """Complex model produces finite outputs"""
        model = compile_model(INTEGRATION_PATH / model_path)
        inputs = model.build_default_inputs()

        residuals, jacobian = model.jax_fn(inputs)

        # Check at least one output is finite
        has_finite = False
        for node, res in residuals.items():
            if np.isfinite(float(res['resist'])):
                has_finite = True
                break

        assert has_finite, f"{model_name} produced no finite outputs"


class TestMosfetLevel1JaxInterpreter:
    """Detailed comparison for simple MOSFET level-1 model

    This tests the multi-way PHI handling (cutoff/linear/saturation regions)
    which requires nested jnp.where calls in JAX.
    """

    @pytest.mark.parametrize("vd,vg,vs,vb,expected_region", [
        # Cutoff: Vgs < Vth (0.4V)
        (1.2, 0.0, 0.0, 0.0, "cutoff"),
        (0.5, 0.3, 0.0, 0.0, "cutoff"),
        # Saturation: Vgs > Vth and Vds > Vov
        (1.2, 1.2, 0.0, 0.0, "saturation"),
        (0.8, 1.0, 0.0, 0.0, "saturation"),
        (1.5, 1.2, 0.0, 0.0, "saturation"),
        # Linear (triode): Vgs > Vth and Vds < Vov
        (0.2, 1.2, 0.0, 0.0, "linear"),
        (0.3, 1.0, 0.0, 0.0, "linear"),
    ])
    def test_mosfet_regions_match_interpreter(
        self, mosfet_level1_model: CompiledModel,
        vd, vg, vs, vb, expected_region
    ):
        """Compare JAX vs interpreter residuals in different MOSFET regions"""
        # The model computes hidden states internally now, but we need to
        # pre-compute them for the interpreter
        vgs = vg - vs
        vds = vd - vs
        vbs = vb - vs
        vth = 0.4
        vov = vgs - vth
        kp = 200e-6
        w = 1e-6
        l = 0.2e-6
        beta = kp * w / l
        lam = 0.01
        gdsmin = 1e-9

        # Build JAX inputs (hidden states will be computed internally)
        jax_inputs = [
            0,      # PMOS (0=NMOS)
            1.0,    # sign
            vs,     # V(s)
            vg,     # V(g)
            0.0,    # Vgs (computed)
            vd,     # V(d)
            0.0,    # Vds (computed)
            vb,     # V(b)
            0.0,    # Vbs (computed)
            kp,     # KP
            w,      # W
            l,      # L
            0.0,    # beta (computed)
            vth,    # VTH0
            0.0,    # Vov (computed)
            gdsmin, # GDSMIN
            0.0,    # Ids (computed)
            lam,    # LAMBDA
            0.0,    # I(d,s)
            1.0,    # mfactor
        ]

        # Build interpreter params (need pre-computed hidden states)
        interp_params = {
            'PMOS': 0,
            'sign': 1.0,
            'V(s)': vs,
            'V(g)': vg,
            'Vgs': vgs,
            'V(d)': vd,
            'Vds': vds,
            'V(b)': vb,
            'Vbs': vbs,
            'KP': kp,
            'W': w,
            'L': l,
            'beta': beta,
            'VTH0': vth,
            'Vov': vov,
            'GDSMIN': gdsmin,
            'Ids': 0.0,  # Will be computed
            'LAMBDA': lam,
            'I(d,s)': 0.0,
            'mfactor': 1.0,
        }

        # Run both
        jax_residuals, jax_jacobian = mosfet_level1_model.jax_fn(jax_inputs)
        interp_residuals, interp_jacobian = mosfet_level1_model.module.run_init_eval(interp_params)

        # Compare drain residual (node 0)
        jax_I_drain = float(jax_residuals['sim_node0']['resist'])
        interp_I_drain = interp_residuals[0][0]

        assert_allclose(
            jax_I_drain, interp_I_drain,
            rtol=1e-6, atol=1e-15,
            err_msg=f"MOSFET drain current mismatch in {expected_region} region: "
                    f"Vd={vd}, Vg={vg}, Vs={vs}"
        )

        # Compare source residual (node 2)
        jax_I_source = float(jax_residuals['sim_node2']['resist'])
        interp_I_source = interp_residuals[2][0]

        assert_allclose(
            jax_I_source, interp_I_source,
            rtol=1e-6, atol=1e-15,
            err_msg=f"MOSFET source current mismatch in {expected_region} region"
        )

        # Verify currents are opposite (KCL)
        assert_allclose(
            jax_I_drain, -jax_I_source,
            rtol=1e-10, atol=1e-15,
            err_msg="MOSFET KCL violated: I_drain != -I_source"
        )

    @pytest.mark.parametrize("vd,vg,vs,vb,expected_region", [
        (1.2, 1.2, 0.0, 0.0, "saturation"),
        (0.2, 1.2, 0.0, 0.0, "linear"),
        (1.2, 0.0, 0.0, 0.0, "cutoff"),
    ])
    def test_mosfet_jacobian_matches_interpreter(
        self, mosfet_level1_model: CompiledModel,
        vd, vg, vs, vb, expected_region
    ):
        """Compare JAX vs interpreter jacobian in different MOSFET regions"""
        vgs = vg - vs
        vds = vd - vs
        vbs = vb - vs
        vth = 0.4
        vov = vgs - vth
        kp = 200e-6
        w = 1e-6
        l = 0.2e-6
        beta = kp * w / l
        lam = 0.01
        gdsmin = 1e-9

        jax_inputs = [
            0, 1.0, vs, vg, 0.0, vd, 0.0, vb, 0.0,
            kp, w, l, 0.0, vth, 0.0, gdsmin, 0.0, lam, 0.0, 1.0
        ]

        interp_params = {
            'PMOS': 0, 'sign': 1.0, 'V(s)': vs, 'V(g)': vg, 'Vgs': vgs,
            'V(d)': vd, 'Vds': vds, 'V(b)': vb, 'Vbs': vbs, 'KP': kp,
            'W': w, 'L': l, 'beta': beta, 'VTH0': vth, 'Vov': vov,
            'GDSMIN': gdsmin, 'Ids': 0.0, 'LAMBDA': lam, 'I(d,s)': 0.0,
            'mfactor': 1.0
        }

        jax_residuals, jax_jacobian = mosfet_level1_model.jax_fn(jax_inputs)
        interp_residuals, interp_jacobian = mosfet_level1_model.module.run_init_eval(interp_params)

        # Compare all 6 Jacobian entries
        # Interpreter format: (row, col, resist, react)
        interp_jac_dict = {(r, c): (resist, react) for r, c, resist, react in interp_jacobian}

        for key, jax_val in jax_jacobian.items():
            row_name, col_name = key
            row_idx = int(row_name.replace('sim_node', ''))
            col_idx = int(col_name.replace('sim_node', ''))

            if (row_idx, col_idx) in interp_jac_dict:
                interp_resist, _ = interp_jac_dict[(row_idx, col_idx)]
                jax_resist = float(jax_val['resist'])

                assert_allclose(
                    jax_resist, interp_resist,
                    rtol=1e-5, atol=1e-12,
                    err_msg=f"MOSFET Jacobian[{row_idx},{col_idx}] mismatch in {expected_region}"
                )

    def test_mosfet_diagonal_is_positive(self, mosfet_level1_model: CompiledModel):
        """Verify MOSFET Jacobian diagonal is positive (critical for Newton stability)

        This is the key property that makes SPICE converge - positive diagonal
        conductances ensure Newton-Raphson steps in the right direction.
        """
        # Test in saturation region
        jax_inputs = [
            0, 1.0, 0.0, 1.2, 0.0, 1.2, 0.0, 0.0, 0.0,
            200e-6, 1e-6, 0.2e-6, 0.0, 0.4, 0.0, 1e-9, 0.0, 0.01, 0.0, 1.0
        ]

        _, jax_jacobian = mosfet_level1_model.jax_fn(jax_inputs)

        # Check diagonal entries are positive
        j_dd = float(jax_jacobian[('sim_node0', 'sim_node0')]['resist'])
        j_ss = float(jax_jacobian[('sim_node2', 'sim_node2')]['resist'])

        assert j_dd > 0, f"Drain diagonal should be positive: J[d,d]={j_dd}"
        assert j_ss > 0, f"Source diagonal should be positive: J[s,s]={j_ss}"

        # Also verify gds (output conductance) is at least GDSMIN
        gdsmin = 1e-9
        assert j_dd >= gdsmin, f"Drain diagonal should be >= GDSMIN: J[d,d]={j_dd}"
