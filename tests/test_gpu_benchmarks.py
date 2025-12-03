"""GPU Performance Benchmarks

Measures CPU vs GPU performance and data transfer overhead.
Run with: pytest tests/test_gpu_benchmarks.py -v -s
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os

jax_spice_path = Path(__file__).parent.parent
if str(jax_spice_path) not in sys.path:
    sys.path.insert(0, str(jax_spice_path))

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from jax_spice.analysis.mna import MNASystem, DeviceInfo, DeviceType
from jax_spice.analysis.context import AnalysisContext
from jax_spice.devices.base import DeviceStamps
from jax_spice.benchmarks.profiler import (
    JAXProfiler, BenchmarkReport, measure_transfer_overhead, quick_benchmark
)


# Skip benchmarks on CPU-only systems unless explicitly requested
def has_gpu():
    """Check if GPU/CUDA backend is available"""
    # Check default devices for GPU/CUDA platforms
    for device in jax.devices():
        platform = device.platform.lower()
        if platform in ('gpu', 'cuda'):
            return True
    return False


requires_gpu = pytest.mark.skipif(
    not has_gpu() and os.environ.get('RUN_GPU_BENCHMARKS') != '1',
    reason="GPU not available (set RUN_GPU_BENCHMARKS=1 to run anyway)"
)


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


class TestTransferOverhead:
    """Measure host<->device data transfer overhead"""

    @requires_gpu
    def test_transfer_overhead_sweep(self, capsys):
        """Measure transfer times for various array sizes"""
        sizes = [100, 1000, 10_000, 100_000, 1_000_000]

        print("\n" + "="*60)
        print("CPU <-> GPU Transfer Overhead")
        print("="*60)

        results = measure_transfer_overhead(sizes)

        # Skip if CPU or GPU device not available for transfer measurement
        if not results['cpu_to_gpu']:
            pytest.skip("CPU<->GPU transfer measurement not available (need both CPU and GPU devices)")

        print(f"\n{'Size':>12} | {'Bytes':>12} | {'CPU→GPU (ms)':>12} | {'GPU→CPU (ms)':>12} | {'BW (GB/s)':>10}")
        print("-"*65)

        for i, size in enumerate(sizes):
            cpu_to_gpu = results['cpu_to_gpu'][i]
            gpu_to_cpu = results['gpu_to_cpu'][i]
            bytes_val = results['bytes'][i]

            print(f"{size:>12,} | {bytes_val:>12,} | "
                  f"{cpu_to_gpu['time_ms']:>12.3f} | "
                  f"{gpu_to_cpu['time_ms']:>12.3f} | "
                  f"{cpu_to_gpu['bandwidth_gb_s']:>10.2f}")

        print("="*60 + "\n")

        # Verify we got results
        assert len(results['cpu_to_gpu']) == len(sizes)
        assert all(r['time_ms'] > 0 for r in results['cpu_to_gpu'])


class TestBatchFunctionBenchmarks:
    """Benchmark individual batch functions"""

    @requires_gpu
    def test_resistor_batch_cpu_vs_gpu(self, capsys):
        """Compare resistor_batch on CPU vs GPU"""
        from jax_spice.devices.resistor import resistor_batch

        # Create test data with varying batch sizes
        batch_sizes = [10, 100, 1000, 10000]

        print("\n" + "="*60)
        print("Resistor Batch: CPU vs GPU")
        print("="*60)

        profiler = JAXProfiler(warmup_iterations=3, benchmark_iterations=20)

        for batch_size in batch_sizes:
            V_batch = jnp.ones((batch_size, 2)) * jnp.array([1.0, 0.0])
            R_batch = jnp.ones(batch_size) * 1000.0

            # JIT compile the function
            jit_resistor_batch = jax.jit(resistor_batch)

            with profiler.benchmark(f"resistor_batch_{batch_size}",
                                   batch_size=batch_size) as report:
                for backend in profiler.get_available_backends():
                    profiler.time_function(
                        jit_resistor_batch,
                        args=(V_batch, R_batch),
                        name=f"batch_{batch_size}",
                        backend=backend,
                    )

            print(report.summary())

    @requires_gpu
    def test_vsource_batch_cpu_vs_gpu(self, capsys):
        """Compare vsource_batch on CPU vs GPU"""
        from jax_spice.devices.vsource import vsource_batch

        batch_sizes = [10, 100, 1000, 10000]

        print("\n" + "="*60)
        print("Vsource Batch: CPU vs GPU")
        print("="*60)

        profiler = JAXProfiler(warmup_iterations=3, benchmark_iterations=20)

        for batch_size in batch_sizes:
            V_batch = jnp.ones((batch_size, 2)) * jnp.array([0.9, 0.0])
            V_target = jnp.ones(batch_size) * 1.0

            jit_vsource_batch = jax.jit(vsource_batch)

            with profiler.benchmark(f"vsource_batch_{batch_size}",
                                   batch_size=batch_size) as report:
                for backend in profiler.get_available_backends():
                    profiler.time_function(
                        jit_vsource_batch,
                        args=(V_batch, V_target),
                        name=f"batch_{batch_size}",
                        backend=backend,
                    )

            print(report.summary())


class TestJacobianAssemblyBenchmarks:
    """Benchmark vectorized vs scalar Jacobian assembly"""

    def _create_ladder_circuit(self, n_rungs: int):
        """Create an RC ladder circuit with n rungs"""
        # Ladder: V1 -> R1 -> R2 -> ... -> Rn -> GND
        #              |      |           |
        #              R      R           R (to ground)

        n_nodes = n_rungs + 2  # gnd + vdd + internal nodes
        nodes = {'gnd': 0, 'vdd': 1}
        for i in range(n_rungs):
            nodes[f'n{i}'] = i + 2

        system = MNASystem(num_nodes=n_nodes, node_names=nodes)

        # Voltage source
        system.devices.append(DeviceInfo(
            name='V1', model_name='vsource', terminals=['p', 'n'],
            node_indices=[1, 0], params={'v': 1.0, 'dc': 1.0},
            eval_fn=make_vsource_eval(1.0)
        ))

        # Series resistors
        prev_node = 1
        for i in range(n_rungs):
            curr_node = i + 2
            system.devices.append(DeviceInfo(
                name=f'Rs{i}', model_name='resistor', terminals=['p', 'n'],
                node_indices=[prev_node, curr_node],
                params={'R': 1000.0, 'r': 1000.0},
                eval_fn=make_resistor_eval()
            ))
            # Shunt resistor to ground
            system.devices.append(DeviceInfo(
                name=f'Rp{i}', model_name='resistor', terminals=['p', 'n'],
                node_indices=[curr_node, 0],
                params={'R': 10000.0, 'r': 10000.0},
                eval_fn=make_resistor_eval()
            ))
            prev_node = curr_node

        system.build_device_groups()
        return system, n_nodes

    @requires_gpu
    def test_jacobian_assembly_scaling(self, capsys):
        """Benchmark Jacobian assembly for different circuit sizes"""
        circuit_sizes = [10, 50, 100, 500]

        print("\n" + "="*60)
        print("Jacobian Assembly: Vectorized vs Scalar, CPU vs GPU")
        print("="*60)

        profiler = JAXProfiler(warmup_iterations=2, benchmark_iterations=10)

        for n_rungs in circuit_sizes:
            system, n_nodes = self._create_ladder_circuit(n_rungs)
            n_devices = len(system.devices)

            # Initial voltage guess
            V = jnp.linspace(0.0, 1.0, n_nodes)
            context = AnalysisContext(time=None, dt=None, analysis_type='dc', gmin=1e-12)

            print(f"\n--- Circuit: {n_rungs} rungs, {n_devices} devices, {n_nodes} nodes ---")

            # Benchmark scalar assembly
            def scalar_assembly():
                return system.build_sparse_jacobian_and_residual(V, context)

            for backend in profiler.get_available_backends():
                result = profiler.time_function(
                    scalar_assembly, name=f"scalar_{n_rungs}",
                    backend=backend, include_jit=True
                )
                print(f"  Scalar  [{backend.upper()}]: "
                      f"exec={result.execution_time_ms:.3f}ms, "
                      f"jit={result.jit_time_ms:.3f}ms")

            # Benchmark vectorized assembly
            def vectorized_assembly():
                return system.build_vectorized_jacobian_and_residual(V, context)

            for backend in profiler.get_available_backends():
                result = profiler.time_function(
                    vectorized_assembly, name=f"vectorized_{n_rungs}",
                    backend=backend, include_jit=True
                )
                print(f"  Vector  [{backend.upper()}]: "
                      f"exec={result.execution_time_ms:.3f}ms, "
                      f"jit={result.jit_time_ms:.3f}ms")

        print("\n" + "="*60)


class TestDeviceInfo:
    """Report device information"""

    def test_device_info(self, capsys):
        """Print available device information"""
        profiler = JAXProfiler()
        info = profiler.get_device_info()

        print("\n" + "="*60)
        print("JAX Device Information")
        print("="*60)
        print(f"\nDefault backend: {info['default_backend']}")
        print(f"Available backends: {profiler.get_available_backends()}")

        for backend, devices in info['devices'].items():
            print(f"\n{backend.upper()} Devices:")
            for d in devices:
                print(f"  - ID: {d['id']}, Platform: {d['platform']}, Kind: {d['device_kind']}")

        print("\n" + "="*60)


class TestEndToEndBenchmark:
    """End-to-end benchmark with realistic workload"""

    @requires_gpu
    def test_mna_solve_iteration(self, capsys):
        """Benchmark a single Newton-Raphson iteration"""
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve

        # Create a moderately sized circuit
        n_rungs = 100
        n_nodes = n_rungs + 2
        nodes = {'gnd': 0, 'vdd': 1}
        for i in range(n_rungs):
            nodes[f'n{i}'] = i + 2

        system = MNASystem(num_nodes=n_nodes, node_names=nodes)

        # Voltage source
        system.devices.append(DeviceInfo(
            name='V1', model_name='vsource', terminals=['p', 'n'],
            node_indices=[1, 0], params={'v': 1.0, 'dc': 1.0},
            eval_fn=make_vsource_eval(1.0)
        ))

        # Resistor ladder
        prev_node = 1
        for i in range(n_rungs):
            curr_node = i + 2
            system.devices.append(DeviceInfo(
                name=f'Rs{i}', model_name='resistor', terminals=['p', 'n'],
                node_indices=[prev_node, curr_node],
                params={'R': 1000.0, 'r': 1000.0},
                eval_fn=make_resistor_eval()
            ))
            system.devices.append(DeviceInfo(
                name=f'Rp{i}', model_name='resistor', terminals=['p', 'n'],
                node_indices=[curr_node, 0],
                params={'R': 10000.0, 'r': 10000.0},
                eval_fn=make_resistor_eval()
            ))
            prev_node = curr_node

        system.build_device_groups()

        V = jnp.linspace(0.0, 1.0, n_nodes)
        context = AnalysisContext(time=None, dt=None, analysis_type='dc', gmin=1e-12)

        print("\n" + "="*60)
        print(f"End-to-End Newton Iteration Benchmark")
        print(f"Circuit: {n_rungs} rungs, {len(system.devices)} devices, {n_nodes} nodes")
        print("="*60)

        profiler = JAXProfiler(warmup_iterations=2, benchmark_iterations=10)

        # Full iteration: Jacobian assembly + linear solve
        def full_iteration_vectorized():
            csr_data, residual = system.build_vectorized_jacobian_and_residual(V, context)
            # Note: Linear solve would typically use scipy on CPU
            # Here we just return to measure assembly time
            return csr_data, residual

        def full_iteration_scalar():
            csr_data, residual = system.build_sparse_jacobian_and_residual(V, context)
            return csr_data, residual

        print("\nJacobian Assembly Only:")
        for backend in profiler.get_available_backends():
            result = profiler.time_function(
                full_iteration_vectorized,
                name=f"vectorized_assembly",
                backend=backend,
            )
            print(f"  Vectorized [{backend.upper()}]: "
                  f"exec={result.execution_time_ms:.3f}ms "
                  f"(per iter: {result.time_per_iter_ms:.3f}ms)")

            result = profiler.time_function(
                full_iteration_scalar,
                name=f"scalar_assembly",
                backend=backend,
            )
            print(f"  Scalar     [{backend.upper()}]: "
                  f"exec={result.execution_time_ms:.3f}ms "
                  f"(per iter: {result.time_per_iter_ms:.3f}ms)")

        print("\n" + "="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
