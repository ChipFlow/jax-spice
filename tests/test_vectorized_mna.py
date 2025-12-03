"""Test vectorized MNA implementation

Tests the GPU-friendly vectorized device evaluation and Jacobian assembly.
Compares vectorized results against scalar implementation to ensure correctness.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

jax_spice_path = Path(__file__).parent.parent
if str(jax_spice_path) not in sys.path:
    sys.path.insert(0, str(jax_spice_path))

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from jax_spice.analysis.mna import MNASystem, DeviceInfo, DeviceType, VectorizedDeviceGroup
from jax_spice.analysis.context import AnalysisContext
from jax_spice.devices.base import DeviceStamps


def make_resistor_eval():
    """Create a resistor evaluation function"""
    def resistor_eval(voltages, params, context):
        Vp = voltages.get('p', 0.0)
        Vn = voltages.get('n', 0.0)
        V = Vp - Vn
        R_val = params.get('R', params.get('r', 1000.0))
        G = 1.0 / max(float(R_val), 1e-12)
        I = G * V
        return DeviceStamps(
            currents={'p': I, 'n': -I},
            conductances={
                ('p', 'p'): G, ('p', 'n'): -G,
                ('n', 'p'): -G, ('n', 'n'): G
            }
        )
    return resistor_eval


def make_vsource_eval(V_dc):
    """Create a voltage source evaluation function"""
    def vsource_eval(voltages, params, context):
        Vp = voltages.get('p', 0.0)
        Vn = voltages.get('n', 0.0)
        V_actual = Vp - Vn
        V_target = params.get('v', params.get('dc', V_dc))
        G_BIG = 1e12
        I = G_BIG * (V_actual - V_target)
        return DeviceStamps(
            currents={'p': I, 'n': -I},
            conductances={
                ('p', 'p'): G_BIG, ('p', 'n'): -G_BIG,
                ('n', 'p'): -G_BIG, ('n', 'n'): G_BIG
            }
        )
    return vsource_eval


class TestVectorizedBatchFunctions:
    """Test the individual batch functions"""

    def test_resistor_batch(self):
        """Test resistor_batch produces correct I = V/R"""
        from jax_spice.devices.resistor import resistor_batch

        V_batch = jnp.array([[1.0, 0.0], [2.0, 1.0], [1.5, 0.5]])
        R_batch = jnp.array([1000.0, 2000.0, 500.0])

        I_batch, G_batch = resistor_batch(V_batch, R_batch)

        expected_I = (V_batch[:, 0] - V_batch[:, 1]) / R_batch
        expected_G = 1.0 / R_batch

        assert jnp.allclose(I_batch, expected_I), f"I mismatch: {I_batch} vs {expected_I}"
        assert jnp.allclose(G_batch, expected_G), f"G mismatch: {G_batch} vs {expected_G}"

    def test_vsource_batch(self):
        """Test vsource_batch produces correct enforcement current"""
        from jax_spice.devices.vsource import vsource_batch

        V_batch = jnp.array([[0.8, 0.0], [1.5, 0.0], [1.0, 0.0]])
        V_target = jnp.array([1.0, 1.2, 1.0])

        I_batch, G_batch = vsource_batch(V_batch, V_target)

        G_BIG = 1e12
        expected_I = G_BIG * ((V_batch[:, 0] - V_batch[:, 1]) - V_target)

        assert jnp.allclose(I_batch, expected_I), f"I mismatch: {I_batch} vs {expected_I}"
        assert jnp.all(G_batch == G_BIG), f"G should all be {G_BIG}"

    def test_resistor_batch_minimum_resistance(self):
        """Test that tiny resistances are clamped for numerical stability"""
        from jax_spice.devices.resistor import resistor_batch

        V_batch = jnp.array([[1.0, 0.0]])
        R_batch = jnp.array([1e-15])  # Very small resistance

        I_batch, G_batch = resistor_batch(V_batch, R_batch)

        # Should be clamped to 1e-12 minimum
        assert G_batch[0] == 1.0 / 1e-12, "Should use minimum resistance"


class TestDeviceGrouping:
    """Test device grouping logic"""

    def test_group_by_type(self):
        """Test that devices are correctly grouped by type"""
        nodes = {'gnd': 0, 'vdd': 1, 'mid': 2}
        system = MNASystem(num_nodes=3, node_names=nodes)

        # Add mixed devices
        system.devices.append(DeviceInfo(
            name='V1', model_name='vsource', terminals=['p', 'n'],
            node_indices=[1, 0], params={'v': 1.0, 'dc': 1.0},
            eval_fn=make_vsource_eval(1.0)
        ))
        system.devices.append(DeviceInfo(
            name='R1', model_name='resistor', terminals=['p', 'n'],
            node_indices=[1, 2], params={'R': 1000.0, 'r': 1000.0},
            eval_fn=make_resistor_eval()
        ))
        system.devices.append(DeviceInfo(
            name='R2', model_name='resistor', terminals=['p', 'n'],
            node_indices=[2, 0], params={'R': 2000.0, 'r': 2000.0},
            eval_fn=make_resistor_eval()
        ))

        system.build_device_groups()

        assert len(system.device_groups) == 2, "Should have 2 groups"

        vsource_groups = [g for g in system.device_groups if g.device_type == DeviceType.VSOURCE]
        resistor_groups = [g for g in system.device_groups if g.device_type == DeviceType.RESISTOR]

        assert len(vsource_groups) == 1, "Should have 1 VSOURCE group"
        assert len(resistor_groups) == 1, "Should have 1 RESISTOR group"

        assert vsource_groups[0].n_devices == 1
        assert resistor_groups[0].n_devices == 2

    def test_node_indices_shape(self):
        """Test that node_indices array has correct shape"""
        nodes = {'gnd': 0, 'a': 1, 'b': 2, 'c': 3}
        system = MNASystem(num_nodes=4, node_names=nodes)

        for i, (np, nn) in enumerate([(1, 2), (2, 3), (3, 0)]):
            system.devices.append(DeviceInfo(
                name=f'R{i+1}', model_name='resistor', terminals=['p', 'n'],
                node_indices=[np, nn], params={'R': 1000.0, 'r': 1000.0},
                eval_fn=make_resistor_eval()
            ))

        system.build_device_groups()

        assert len(system.device_groups) == 1
        group = system.device_groups[0]

        assert group.node_indices.shape == (3, 2), f"Expected (3, 2), got {group.node_indices.shape}"


class TestVectorizedVsScalar:
    """Compare vectorized and scalar implementations"""

    def test_simple_circuit(self):
        """Test that vectorized produces same results as scalar for simple circuit"""
        # Circuit: V1 (1V) -> R1 (1k) -> R2||R3 -> GND
        nodes = {'gnd': 0, 'vdd': 1, 'mid': 2}
        system = MNASystem(num_nodes=3, node_names=nodes)

        system.devices.append(DeviceInfo(
            name='V1', model_name='vsource', terminals=['p', 'n'],
            node_indices=[1, 0], params={'v': 1.0, 'dc': 1.0},
            eval_fn=make_vsource_eval(1.0)
        ))
        system.devices.append(DeviceInfo(
            name='R1', model_name='resistor', terminals=['p', 'n'],
            node_indices=[1, 2], params={'R': 1000.0, 'r': 1000.0},
            eval_fn=make_resistor_eval()
        ))
        system.devices.append(DeviceInfo(
            name='R2', model_name='resistor', terminals=['p', 'n'],
            node_indices=[2, 0], params={'R': 2000.0, 'r': 2000.0},
            eval_fn=make_resistor_eval()
        ))
        system.devices.append(DeviceInfo(
            name='R3', model_name='resistor', terminals=['p', 'n'],
            node_indices=[2, 0], params={'R': 3000.0, 'r': 3000.0},
            eval_fn=make_resistor_eval()
        ))

        system.build_device_groups()

        V = jnp.array([0.0, 1.0, 0.5])
        context = AnalysisContext(time=None, dt=None, analysis_type='dc', gmin=1e-12)

        # Scalar version
        csr_scalar, res_scalar = system.build_sparse_jacobian_and_residual(V, context)

        # Vectorized version
        csr_vec, res_vec = system.build_vectorized_jacobian_and_residual(V, context)

        # Compare residuals
        assert np.allclose(res_scalar, res_vec), \
            f"Residual mismatch: max diff = {np.max(np.abs(res_scalar - res_vec))}"

        # Build dense matrices to compare
        from scipy.sparse import csr_matrix

        J_scalar = csr_matrix(
            (csr_scalar[0], csr_scalar[1], csr_scalar[2]),
            shape=csr_scalar[3]
        ).toarray()

        J_vec = csr_matrix(
            (csr_vec[0], csr_vec[1], csr_vec[2]),
            shape=csr_vec[3]
        ).toarray()

        assert np.allclose(J_scalar, J_vec), \
            f"Jacobian mismatch: max diff = {np.max(np.abs(J_scalar - J_vec))}"

    def test_multiple_voltage_sources(self):
        """Test with multiple voltage sources"""
        nodes = {'gnd': 0, 'v1': 1, 'v2': 2, 'mid': 3}
        system = MNASystem(num_nodes=4, node_names=nodes)

        system.devices.append(DeviceInfo(
            name='V1', model_name='vsource', terminals=['p', 'n'],
            node_indices=[1, 0], params={'v': 1.0, 'dc': 1.0},
            eval_fn=make_vsource_eval(1.0)
        ))
        system.devices.append(DeviceInfo(
            name='V2', model_name='vsource', terminals=['p', 'n'],
            node_indices=[2, 0], params={'v': 2.0, 'dc': 2.0},
            eval_fn=make_vsource_eval(2.0)
        ))
        system.devices.append(DeviceInfo(
            name='R1', model_name='resistor', terminals=['p', 'n'],
            node_indices=[1, 3], params={'R': 1000.0, 'r': 1000.0},
            eval_fn=make_resistor_eval()
        ))
        system.devices.append(DeviceInfo(
            name='R2', model_name='resistor', terminals=['p', 'n'],
            node_indices=[2, 3], params={'R': 2000.0, 'r': 2000.0},
            eval_fn=make_resistor_eval()
        ))

        system.build_device_groups()

        V = jnp.array([0.0, 1.0, 2.0, 1.3])
        context = AnalysisContext(time=None, dt=None, analysis_type='dc', gmin=1e-12)

        csr_scalar, res_scalar = system.build_sparse_jacobian_and_residual(V, context)
        csr_vec, res_vec = system.build_vectorized_jacobian_and_residual(V, context)

        assert np.allclose(res_scalar, res_vec), "Residual mismatch"

    def test_ground_terminal_handling(self):
        """Test that ground terminal stamps are handled correctly"""
        # R1 connected between node 1 and ground
        nodes = {'gnd': 0, 'n1': 1}
        system = MNASystem(num_nodes=2, node_names=nodes)

        system.devices.append(DeviceInfo(
            name='V1', model_name='vsource', terminals=['p', 'n'],
            node_indices=[1, 0], params={'v': 1.0, 'dc': 1.0},
            eval_fn=make_vsource_eval(1.0)
        ))

        system.build_device_groups()

        V = jnp.array([0.0, 0.8])
        context = AnalysisContext(time=None, dt=None, analysis_type='dc', gmin=1e-12)

        csr_scalar, res_scalar = system.build_sparse_jacobian_and_residual(V, context)
        csr_vec, res_vec = system.build_vectorized_jacobian_and_residual(V, context)

        assert np.allclose(res_scalar, res_vec), \
            f"Residual mismatch: scalar={res_scalar}, vec={res_vec}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
