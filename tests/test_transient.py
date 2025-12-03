"""Test transient analysis on simple circuits

Tests RC circuit charging/discharging with backward Euler integration.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from jax_spice.devices.base import DeviceStamps
from jax_spice.analysis.context import AnalysisContext
from jax_spice.analysis.mna import MNASystem, DeviceInfo
from jax_spice.analysis.dc import dc_operating_point
from jax_spice.analysis.transient import transient_analysis


# Simple device implementations for testing
def resistor_eval(voltages, params, context):
    """Resistor evaluation function"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    R = float(params.get('r', params.get('R', 1000.0)))
    G = 1.0 / R
    I = G * (Vp - Vn)
    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G), ('p', 'n'): jnp.array(-G),
            ('n', 'p'): jnp.array(-G), ('n', 'n'): jnp.array(G)
        }
    )


def capacitor_eval(voltages, params, context):
    """Capacitor evaluation function"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    C = float(params.get('c', params.get('C', 1e-12)))
    V = Vp - Vn
    Q = C * V
    G_small = 1e-15  # Small leakage
    return DeviceStamps(
        currents={'p': jnp.array(0.0), 'n': jnp.array(0.0)},
        conductances={
            ('p', 'p'): jnp.array(G_small), ('p', 'n'): jnp.array(-G_small),
            ('n', 'p'): jnp.array(-G_small), ('n', 'n'): jnp.array(G_small)
        },
        charges={'p': jnp.array(Q), 'n': jnp.array(-Q)},
        capacitances={
            ('p', 'p'): jnp.array(C), ('p', 'n'): jnp.array(-C),
            ('n', 'p'): jnp.array(-C), ('n', 'n'): jnp.array(C)
        }
    )


def vsource_eval(voltages, params, context):
    """Voltage source evaluation function (DC only for now)"""
    Vp = voltages.get('p', 0.0)
    Vn = voltages.get('n', 0.0)
    V_target = float(params.get('v', params.get('dc', 0.0)))
    V_actual = Vp - Vn
    
    G_big = 1e12  # Large conductance to force voltage
    I = G_big * (V_actual - V_target)
    
    return DeviceStamps(
        currents={'p': jnp.array(I), 'n': jnp.array(-I)},
        conductances={
            ('p', 'p'): jnp.array(G_big), ('p', 'n'): jnp.array(-G_big),
            ('n', 'p'): jnp.array(-G_big), ('n', 'n'): jnp.array(G_big)
        }
    )


class TestDCOperatingPoint:
    """Test DC operating point analysis"""
    
    def test_voltage_divider(self):
        """Test simple voltage divider"""
        # Create MNA system manually
        # Circuit: Vs -- R1 -- node_mid -- R2 -- GND
        system = MNASystem(
            num_nodes=3,  # GND(0), Vs_p(1), mid(2)
            node_names={'0': 0, 'vs_p': 1, 'mid': 2}
        )
        
        # Add devices
        # Voltage source: vs_p to ground
        system.devices.append(DeviceInfo(
            name='Vs',
            model_name='vsource',
            terminals=['p', 'n'],
            node_indices=[1, 0],
            params={'v': 5.0},
            eval_fn=vsource_eval
        ))
        
        # R1: vs_p to mid, 1k
        system.devices.append(DeviceInfo(
            name='R1',
            model_name='resistor',
            terminals=['p', 'n'],
            node_indices=[1, 2],
            params={'r': 1000.0},
            eval_fn=resistor_eval
        ))
        
        # R2: mid to ground, 1k
        system.devices.append(DeviceInfo(
            name='R2',
            model_name='resistor',
            terminals=['p', 'n'],
            node_indices=[2, 0],
            params={'r': 1000.0},
            eval_fn=resistor_eval
        ))
        
        # Solve
        solution, info = dc_operating_point(system)
        
        assert info['converged'], f"Did not converge: {info}"
        
        # Check voltages
        V_vs = float(solution[1])  # Should be 5V
        V_mid = float(solution[2])  # Should be 2.5V
        
        assert abs(V_vs - 5.0) < 0.01, f"Vs voltage error: {V_vs}"
        assert abs(V_mid - 2.5) < 0.01, f"Mid voltage error: {V_mid}"


class TestTransientRC:
    """Test transient analysis on RC circuits"""
    
    def test_rc_charging(self):
        """Test RC circuit charging from 0V to 5V
        
        Circuit: Vs(5V) -- R(1k) -- C(1uF) -- GND
        Time constant: tau = RC = 1ms
        """
        R = 1000.0      # 1k ohm
        C = 1e-6        # 1 uF
        tau = R * C     # 1 ms
        V_s = 5.0       # 5V step
        
        # Create MNA system
        system = MNASystem(
            num_nodes=3,  # GND(0), Vs(1), cap_p(2)
            node_names={'0': 0, 'vs': 1, 'cap': 2}
        )
        
        # Voltage source at node 1
        system.devices.append(DeviceInfo(
            name='Vs',
            model_name='vsource',
            terminals=['p', 'n'],
            node_indices=[1, 0],
            params={'v': V_s},
            eval_fn=vsource_eval
        ))
        
        # Resistor from vs to cap
        system.devices.append(DeviceInfo(
            name='R1',
            model_name='resistor',
            terminals=['p', 'n'],
            node_indices=[1, 2],
            params={'r': R},
            eval_fn=resistor_eval
        ))
        
        # Capacitor from cap to ground
        system.devices.append(DeviceInfo(
            name='C1',
            model_name='capacitor',
            terminals=['p', 'n'],
            node_indices=[2, 0],
            params={'c': C},
            eval_fn=capacitor_eval
        ))
        
        # Run transient analysis
        t_stop = 5 * tau  # 5 time constants
        t_step = tau / 100  # 100 points per time constant
        
        times, solutions, info = transient_analysis(
            system,
            t_stop=t_stop,
            t_step=t_step,
            initial_conditions={'vs': V_s, 'cap': 0.0}
        )
        
        # Check that capacitor voltage follows exponential
        V_cap = solutions[:, 2]  # Node 2 is capacitor
        
        # Analytical solution: V_cap(t) = V_s * (1 - exp(-t/tau))
        V_analytical = V_s * (1 - np.exp(-np.array(times) / tau))
        
        # Check at t = tau (should be ~63% of V_s)
        idx_tau = int(tau / t_step)
        V_at_tau = float(V_cap[idx_tau])
        expected = V_s * (1 - np.exp(-1))  # ~3.16V
        
        # Allow 5% error due to discretization
        assert abs(V_at_tau - expected) < 0.05 * V_s, \
            f"V(tau) = {V_at_tau:.3f}V, expected {expected:.3f}V"
        
        # Check at t = 5*tau (should be ~99.3% of V_s)
        V_final = float(V_cap[-1])
        expected_final = V_s * (1 - np.exp(-5))  # ~4.97V
        
        assert abs(V_final - expected_final) < 0.05 * V_s, \
            f"V(5*tau) = {V_final:.3f}V, expected {expected_final:.3f}V"


class TestMNASystem:
    """Test MNA system construction"""
    
    def test_create_system(self):
        """Test creating MNA system"""
        system = MNASystem(
            num_nodes=3,
            node_names={'gnd': 0, 'n1': 1, 'n2': 2}
        )
        
        assert system.num_nodes == 3
        assert system.node_names['gnd'] == 0
        assert system.ground_node == 0
    
    def test_add_device(self):
        """Test adding devices to system"""
        system = MNASystem(
            num_nodes=2,
            node_names={'0': 0, 'n1': 1}
        )
        
        system.devices.append(DeviceInfo(
            name='R1',
            model_name='resistor',
            terminals=['p', 'n'],
            node_indices=[1, 0],
            params={'r': 1000.0},
            eval_fn=resistor_eval
        ))
        
        assert len(system.devices) == 1
        assert system.devices[0].name == 'R1'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
