#!/usr/bin/env python3
"""CPU Profiling Script for JAX-SPICE

Profiles CPU performance of VACASK benchmark circuits using VACASKBenchmarkRunner.
Compares dense vs sparse solvers across different circuit sizes.

Usage:
    # Run all benchmarks
    uv run scripts/profile_cpu.py

    # Run specific benchmarks
    uv run scripts/profile_cpu.py --benchmark ring,rc

    # Sparse only (for large circuits)
    uv run scripts/profile_cpu.py --benchmark c6288 --sparse-only
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Ensure jax-spice is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force CPU backend
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp

# Enable float64
jax.config.update('jax_enable_x64', True)

from jax_spice.benchmarks.runner import VACASKBenchmarkRunner


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    name: str
    nodes: int
    devices: int
    openvaf_devices: int
    timesteps: int
    total_time_s: float
    time_per_step_ms: float
    solver: str  # 'dense' or 'sparse'
    converged: bool = True
    error: Optional[str] = None


def log(msg="", end="\n"):
    """Print with flush for real-time output"""
    print(msg, end=end, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def get_vacask_benchmarks(names: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    """Get list of VACASK benchmark .sim files"""
    base = Path(__file__).parent.parent / "vendor" / "VACASK" / "benchmark"
    all_benchmarks = ['rc', 'graetz', 'mul', 'ring', 'c6288']

    if names:
        all_benchmarks = [n for n in names if n in all_benchmarks]

    benchmarks = []
    for name in all_benchmarks:
        sim_path = base / name / "vacask" / "runme.sim"
        if sim_path.exists():
            benchmarks.append((name, sim_path))

    return benchmarks


def run_benchmark(sim_path: Path, name: str, use_sparse: bool,
                  num_steps: int = 20, warmup_steps: int = 5) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    try:
        # Parse circuit
        log(f"      parsing...")
        runner = VACASKBenchmarkRunner(sim_path, verbose=True)
        runner.parse()
        log("      parsing done")

        nodes = runner.num_nodes
        devices = len(runner.devices)
        openvaf_devices = sum(1 for d in runner.devices if d.get('is_openvaf'))

        # Skip sparse for non-OpenVAF circuits
        if use_sparse and not runner._has_openvaf_devices:
            return BenchmarkResult(
                name=name,
                nodes=nodes,
                devices=devices,
                openvaf_devices=openvaf_devices,
                timesteps=0,
                total_time_s=0,
                time_per_step_ms=0,
                solver='sparse',
                converged=True,
                error="Sparse not applicable (no OpenVAF devices)"
            )

        # Get analysis params
        dt = runner.analysis_params.get('step', 1e-12)

        # Warmup run (includes JIT compilation)
        log(f"      warmup ({warmup_steps} steps, includes JIT)...")
        warmup_start = time.perf_counter()
        runner.run_transient(t_stop=dt * warmup_steps, dt=dt,
                            max_steps=warmup_steps, use_sparse=use_sparse)
        warmup_time = time.perf_counter() - warmup_start
        log(f"      warmup done ({warmup_time:.1f}s)")

        # Create fresh runner for timing (reuse compiled models)
        runner2 = VACASKBenchmarkRunner(sim_path, verbose=False)
        runner2.parse()
        if runner._has_openvaf_devices:
            runner2._compiled_models = runner._compiled_models

        # Timed run
        start = time.perf_counter()
        times, voltages, stats = runner2.run_transient(
            t_stop=dt * num_steps, dt=dt,
            max_steps=num_steps, use_sparse=use_sparse
        )
        elapsed = time.perf_counter() - start

        actual_steps = len(times)
        time_per_step = (elapsed / actual_steps * 1000) if actual_steps > 0 else 0

        return BenchmarkResult(
            name=name,
            nodes=nodes,
            devices=devices,
            openvaf_devices=openvaf_devices,
            timesteps=actual_steps,
            total_time_s=elapsed,
            time_per_step_ms=time_per_step,
            solver='sparse' if use_sparse else 'dense',
            converged=True,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return BenchmarkResult(
            name=name,
            nodes=0,
            devices=0,
            openvaf_devices=0,
            timesteps=0,
            total_time_s=0,
            time_per_step_ms=0,
            solver='sparse' if use_sparse else 'dense',
            converged=False,
            error=str(e)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Profile VACASK benchmarks on CPU"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks to run (default: all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20,
        help="Number of timesteps per benchmark (default: 20)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5)",
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="Only run sparse solver (skip dense comparison)",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Only run dense solver (skip sparse comparison)",
    )
    args = parser.parse_args()

    log("=" * 70)
    log("JAX-SPICE CPU Profiling")
    log("=" * 70)
    log()

    log("[Stage 1/3] Checking JAX configuration...")
    log(f"  JAX backend: {jax.default_backend()}")
    log(f"  JAX devices: {jax.devices()}")
    log(f"  Float64 enabled: {jax.config.jax_enable_x64}")
    log()

    # Parse benchmark names
    benchmark_names = None
    if args.benchmark:
        benchmark_names = [b.strip() for b in args.benchmark.split(',')]

    log("[Stage 2/3] Discovering benchmarks...")
    benchmarks = get_vacask_benchmarks(benchmark_names)
    log(f"  Found {len(benchmarks)} benchmarks: {[b[0] for b in benchmarks]}")
    log()

    results: List[BenchmarkResult] = []
    start_time = time.perf_counter()

    log("[Stage 3/3] Running benchmarks...")
    log()

    for name, sim_path in benchmarks:
        log(f"  {name}:")

        # Determine which solvers to run
        run_dense = not args.sparse_only and name != 'c6288'  # c6288 too large for dense
        run_sparse = not args.dense_only

        if name == 'c6288' and not args.sparse_only:
            log(f"    Skipping dense (86k nodes would need ~56GB)")

        # Run dense
        if run_dense:
            result_dense = run_benchmark(
                sim_path, name, use_sparse=False,
                num_steps=args.timesteps, warmup_steps=args.warmup_steps
            )
            results.append(result_dense)
            if result_dense.error:
                log(f"    dense:  ERROR - {result_dense.error}")
            else:
                log(f"    dense:  {result_dense.time_per_step_ms:.1f}ms/step ({result_dense.timesteps} steps)")

        # Run sparse
        if run_sparse:
            result_sparse = run_benchmark(
                sim_path, name, use_sparse=True,
                num_steps=args.timesteps, warmup_steps=args.warmup_steps
            )
            results.append(result_sparse)
            if result_sparse.error:
                log(f"    sparse: {result_sparse.error}")
            else:
                log(f"    sparse: {result_sparse.time_per_step_ms:.1f}ms/step ({result_sparse.timesteps} steps)")

        log()

    total_time = time.perf_counter() - start_time

    # Print summary
    log("=" * 70)
    log("Summary")
    log("=" * 70)
    log()
    log(f"Total time: {total_time:.1f}s")
    log()

    # Results table
    log("| Benchmark | Nodes | Solver | Steps | Total (s) | Per Step (ms) | Status |")
    log("|-----------|-------|--------|-------|-----------|---------------|--------|")
    for r in results:
        status = "OK" if r.converged and not r.error else (r.error or "Failed")[:15]
        log(f"| {r.name:9} | {r.nodes:5} | {r.solver:6} | {r.timesteps:5} | "
            f"{r.total_time_s:9.3f} | {r.time_per_step_ms:13.1f} | {status:6} |")

    log()
    log("=" * 70)
    log("Profiling complete!")
    log("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n*** FATAL ERROR ***")
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
