"""Tests comparing JAX output against analytical formulas

This mirrors OpenVAF's integration test approach: use specific physical
constants and verify the computed residuals/Jacobians match analytical formulas.

These tests use the same methodology as:
  openvaf/openvaf/tests/integration.rs
  openvaf/openvaf/tests/mock_sim/mod.rs
"""

import pytest
import numpy as np
from math import exp, sqrt, log, pow

from conftest import (
    INTEGRATION_PATH, LOCAL_MODELS_PATH, assert_allclose,
    CompiledModel
)


# Physical constants (same as OpenVAF tests)
KB = 1.3806488e-23      # Boltzmann constant [J/K]
Q = 1.602176565e-19     # Elementary charge [C]
T_ROOM = 300.0          # Room temperature [K]
VT = KB * T_ROOM / Q    # Thermal voltage at 300K (~25.85mV)


class TestResistorAnalytical:
    """Test resistor model against Ohm's law"""

    @pytest.fixture
    def resistor(self, compile_model):
        return compile_model(INTEGRATION_PATH / "RESISTOR/resistor.va")

    @pytest.mark.parametrize("voltage,resistance", [
        (1.0, 1000.0),
        (0.5, 500.0),
        (-1.0, 1000.0),
        (0.01, 100.0),
    ])
    def test_ohms_law_current(self, resistor, voltage, resistance):
        """I = V/R"""
        # JAX inputs: [V(A,B), vres, R, $temperature, tnom, zeta, res, mfactor]
        jax_inputs = [voltage, voltage, resistance, T_ROOM, T_ROOM, 0.0, resistance, 1.0]

        jax_residuals, _ = resistor.jax_fn(jax_inputs)

        expected_current = voltage / resistance
        actual_current = float(jax_residuals['sim_node0']['resist'])

        assert_allclose(actual_current, expected_current, rtol=1e-6,
                       err_msg=f"V={voltage}, R={resistance}")

    @pytest.mark.parametrize("voltage,resistance", [
        (1.0, 1000.0),
        (5.0, 100.0),
    ])
    def test_ohms_law_conductance(self, resistor, voltage, resistance):
        """dI/dV = 1/R (Jacobian)"""
        jax_inputs = [voltage, voltage, resistance, T_ROOM, T_ROOM, 0.0, resistance, 1.0]

        _, jax_jacobian = resistor.jax_fn(jax_inputs)

        expected_conductance = 1.0 / resistance
        actual_conductance = float(jax_jacobian[('sim_node0', 'sim_node0')]['resist'])

        assert_allclose(actual_conductance, expected_conductance, rtol=1e-6,
                       err_msg=f"R={resistance}")

    @pytest.mark.parametrize("temperature,zeta", [
        (T_ROOM, 0.0),      # No temp dependence
        (350.0, 1.0),       # Linear: R * (T/Tnom)^1
        (400.0, 2.0),       # Quadratic: R * (T/Tnom)^2
    ])
    def test_temperature_scaling(self, resistor, temperature, zeta):
        """R_eff = R * (T/Tnom)^zeta"""
        voltage = 1.0
        R_nominal = 1000.0
        tnom = T_ROOM

        # Effective resistance at temperature
        R_eff = R_nominal * pow(temperature / tnom, zeta)
        expected_current = voltage / R_eff

        jax_inputs = [voltage, voltage, R_nominal, temperature, tnom, zeta, R_eff, 1.0]

        jax_residuals, _ = resistor.jax_fn(jax_inputs)
        actual_current = float(jax_residuals['sim_node0']['resist'])

        assert_allclose(actual_current, expected_current, rtol=1e-6,
                       err_msg=f"T={temperature}, zeta={zeta}")


class TestMosfetLevel1Analytical:
    """Test simple MOSFET model against Shichman-Hodges equations"""

    @pytest.fixture
    def mosfet(self, compile_model):
        return compile_model(LOCAL_MODELS_PATH / "mosfet_level1.va")

    # Model parameters
    VTH0 = 0.4      # Threshold voltage [V]
    KP = 200e-6     # Transconductance [A/V^2]
    LAMBDA = 0.01   # Channel length modulation [1/V]
    W = 1e-6        # Width [m]
    L = 0.2e-6      # Length [m]
    GDSMIN = 1e-9   # Min output conductance [S]

    @property
    def beta(self):
        return self.KP * self.W / self.L  # = 1e-3 A/V^2

    def _build_inputs(self, vd, vg, vs=0.0, vb=0.0, pmos=0):
        """Build JAX input array for MOSFET"""
        return [
            pmos,           # PMOS flag
            1.0 if pmos == 0 else -1.0,  # sign
            vs,             # V(s)
            vg,             # V(g)
            0.0,            # Vgs (computed)
            vd,             # V(d)
            0.0,            # Vds (computed)
            vb,             # V(b)
            0.0,            # Vbs (computed)
            self.KP,        # KP
            self.W,         # W
            self.L,         # L
            0.0,            # beta (computed)
            self.VTH0,      # VTH0
            0.0,            # Vov (computed)
            self.GDSMIN,    # GDSMIN
            0.0,            # Ids (computed)
            self.LAMBDA,    # LAMBDA
            0.0,            # I(d,s)
            1.0,            # mfactor
        ]

    def _calc_ids_analytical(self, vgs, vds):
        """Calculate Ids using Shichman-Hodges equations"""
        vov = vgs - self.VTH0

        if vov <= 0:
            # Cutoff
            ids = self.GDSMIN * vds
        elif vds < vov:
            # Linear region
            ids = self.beta * (vov * vds - 0.5 * vds**2) * (1 + self.LAMBDA * vds)
        else:
            # Saturation region
            ids = 0.5 * self.beta * vov**2 * (1 + self.LAMBDA * vds)

        # Add leakage
        ids += self.GDSMIN * vds
        return ids

    def _calc_gds_analytical(self, vgs, vds):
        """Calculate output conductance dIds/dVds"""
        vov = vgs - self.VTH0

        if vov <= 0:
            # Cutoff
            gds = self.GDSMIN
        elif vds < vov:
            # Linear region
            # Ids = beta * (Vov * Vds - 0.5 * Vds^2) * (1 + lambda * Vds)
            # dIds/dVds = beta * (Vov - Vds) * (1 + lambda*Vds) + beta * (Vov*Vds - 0.5*Vds^2) * lambda
            term1 = self.beta * (vov - vds) * (1 + self.LAMBDA * vds)
            term2 = self.beta * (vov * vds - 0.5 * vds**2) * self.LAMBDA
            gds = term1 + term2
        else:
            # Saturation region
            # Ids = 0.5 * beta * Vov^2 * (1 + lambda * Vds)
            # dIds/dVds = 0.5 * beta * Vov^2 * lambda
            gds = 0.5 * self.beta * vov**2 * self.LAMBDA

        # Add leakage conductance
        gds += self.GDSMIN
        return gds

    def _calc_gm_analytical(self, vgs, vds):
        """Calculate transconductance dIds/dVgs"""
        vov = vgs - self.VTH0

        if vov <= 0:
            return 0.0
        elif vds < vov:
            # Linear region: dIds/dVgs = beta * Vds * (1 + lambda*Vds)
            return self.beta * vds * (1 + self.LAMBDA * vds)
        else:
            # Saturation: dIds/dVgs = beta * Vov * (1 + lambda*Vds)
            return self.beta * vov * (1 + self.LAMBDA * vds)

    @pytest.mark.parametrize("vd,vg,expected_region", [
        (1.2, 0.0, "cutoff"),
        (1.2, 0.3, "cutoff"),
        (1.2, 1.2, "saturation"),
        (0.8, 1.0, "saturation"),
        (0.2, 1.2, "linear"),
        (0.3, 1.0, "linear"),
    ])
    def test_drain_current(self, mosfet, vd, vg, expected_region):
        """Ids matches Shichman-Hodges model"""
        vgs = vg
        vds = vd

        jax_inputs = self._build_inputs(vd, vg)
        jax_residuals, _ = mosfet.jax_fn(jax_inputs)

        expected_ids = self._calc_ids_analytical(vgs, vds)
        actual_ids = float(jax_residuals['sim_node0']['resist'])

        assert_allclose(actual_ids, expected_ids, rtol=1e-6,
                       err_msg=f"{expected_region}: Vgs={vgs}, Vds={vds}")

    @pytest.mark.parametrize("vd,vg,expected_region", [
        (1.2, 0.0, "cutoff"),
        (1.2, 1.2, "saturation"),
        (0.2, 1.2, "linear"),
    ])
    def test_output_conductance(self, mosfet, vd, vg, expected_region):
        """gds = dIds/dVds matches analytical derivative"""
        vgs = vg
        vds = vd

        jax_inputs = self._build_inputs(vd, vg)
        _, jax_jacobian = mosfet.jax_fn(jax_inputs)

        expected_gds = self._calc_gds_analytical(vgs, vds)
        # J[drain,drain] = dI_drain/dV_drain = gds
        actual_gds = float(jax_jacobian[('sim_node0', 'sim_node0')]['resist'])

        assert_allclose(actual_gds, expected_gds, rtol=1e-5,
                       err_msg=f"{expected_region}: gds")

    @pytest.mark.parametrize("vd,vg,expected_region", [
        (1.2, 1.2, "saturation"),
        (0.2, 1.2, "linear"),
    ])
    def test_transconductance(self, mosfet, vd, vg, expected_region):
        """gm = dIds/dVgs matches analytical derivative"""
        vgs = vg
        vds = vd

        jax_inputs = self._build_inputs(vd, vg)
        _, jax_jacobian = mosfet.jax_fn(jax_inputs)

        expected_gm = self._calc_gm_analytical(vgs, vds)
        # J[drain,gate] = dI_drain/dV_gate = gm
        actual_gm = float(jax_jacobian[('sim_node0', 'sim_node1')]['resist'])

        assert_allclose(actual_gm, expected_gm, rtol=1e-5,
                       err_msg=f"{expected_region}: gm")

    def test_kcl_conservation(self, mosfet):
        """Current into drain equals current out of source"""
        jax_inputs = self._build_inputs(vd=1.2, vg=1.2)
        jax_residuals, _ = mosfet.jax_fn(jax_inputs)

        i_drain = float(jax_residuals['sim_node0']['resist'])
        i_source = float(jax_residuals['sim_node2']['resist'])

        # KCL: I_drain + I_source = 0
        assert_allclose(i_drain, -i_source, rtol=1e-10,
                       err_msg="KCL violated")

    def test_positive_diagonal_jacobian(self, mosfet):
        """Jacobian diagonal entries are positive (critical for NR stability)"""
        jax_inputs = self._build_inputs(vd=1.2, vg=1.2)
        _, jax_jacobian = mosfet.jax_fn(jax_inputs)

        j_dd = float(jax_jacobian[('sim_node0', 'sim_node0')]['resist'])
        j_ss = float(jax_jacobian[('sim_node2', 'sim_node2')]['resist'])

        assert j_dd > 0, f"J[d,d] must be positive: {j_dd}"
        assert j_ss > 0, f"J[s,s] must be positive: {j_ss}"
        assert j_dd >= self.GDSMIN, f"J[d,d] must be >= GDSMIN"


class TestCurrentSourceAnalytical:
    """Test current source model"""

    @pytest.fixture
    def isrc(self, compile_model):
        return compile_model(INTEGRATION_PATH / "CURRENT_SOURCE/current_source.va")

    def test_current_source_compiles(self, isrc):
        """Current source compiles and produces output"""
        jax_inputs = isrc.build_default_inputs()
        jax_residuals, jax_jacobian = isrc.jax_fn(jax_inputs)

        assert isinstance(jax_residuals, dict)
        assert len(jax_residuals) >= 2
