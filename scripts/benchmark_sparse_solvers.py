#!/usr/bin/env python3
"""Benchmark sparse solvers for large circuit simulation.

This script compares:
1. JAX native spsolve (jax.experimental.sparse.linalg.spsolve)
2. UMFPACK via pure_callback (with cached symbolic factorization)

The goal is to determine if JAX native spsolve can match or beat UMFPACK
for large circuits like c6288 (5156 unknowns).

Usage:
    # Run on small benchmark first (faster)
    JAX_PLATFORMS=cpu uv run scripts/benchmark_sparse_solvers.py --benchmark ring

    # Run on c6288 (the target benchmark)
    JAX_PLATFORMS=cpu uv run scripts/benchmark_sparse_solvers.py --benchmark c6288

    # More timesteps for better statistics
    JAX_PLATFORMS=cpu uv run scripts/benchmark_sparse_solvers.py --benchmark c6288 --timesteps 50
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Force CPU backend
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# Ensure jax-spice is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import numpy as np

from jax_spice.analysis import CircuitEngine
from jax_spice.analysis.umfpack_solver import is_umfpack_available


def log(msg: str = "") -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def run_benchmark(sim_path: Path, name: str, sparse_solver: str,
                  num_steps: int = 20) -> Dict:
    """Run a single benchmark configuration.

    Args:
        sim_path: Path to .sim file
        name: Benchmark name
        sparse_solver: 'jax' or 'umfpack'
        num_steps: Number of timesteps

    Returns:
        Dict with timing results
    """
    try:
        # Parse circuit
        log(f"      parsing...")
        engine = CircuitEngine(sim_path)
        engine.parse()

        nodes = engine.num_nodes
        devices = len(engine.devices)

        # Get analysis params
        dt = engine.analysis_params.get('step', 1e-12)

        # Warmup run (includes JIT compilation)
        log(f"      warmup ({num_steps} steps, solver={sparse_solver})...")
        warmup_start = time.perf_counter()
        engine.run_transient(
            t_stop=dt * num_steps, dt=dt,
            max_steps=num_steps, use_sparse=True,
            sparse_solver=sparse_solver
        )
        warmup_time = time.perf_counter() - warmup_start
        log(f"      warmup done ({warmup_time:.1f}s)")

        # Timed run - use same engine with cached JIT functions
        log(f"      timed run...")
        start = time.perf_counter()
        result = engine.run_transient(
            t_stop=dt * num_steps, dt=dt,
            max_steps=num_steps, use_sparse=True,
            sparse_solver=sparse_solver
        )
        elapsed = time.perf_counter() - start

        actual_steps = result.num_steps
        time_per_step = (elapsed / actual_steps * 1000) if actual_steps > 0 else 0

        return {
            'name': name,
            'solver': sparse_solver,
            'nodes': nodes,
            'devices': devices,
            'timesteps': actual_steps,
            'total_time_s': elapsed,
            'time_per_step_ms': time_per_step,
            'warmup_time_s': warmup_time,
            'converged': True,
            'error': None,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'solver': sparse_solver,
            'nodes': 0,
            'devices': 0,
            'timesteps': 0,
            'total_time_s': 0,
            'time_per_step_ms': 0,
            'warmup_time_s': 0,
            'converged': False,
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sparse solvers for circuit simulation"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ring",
        help="Benchmark circuit to use (default: ring). Options: rc, graetz, ring, c6288",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20,
        help="Number of timesteps per benchmark (default: 20)",
    )
    args = parser.parse_args()

    log("=" * 70)
    log("Sparse Solver Benchmark")
    log("=" * 70)
    log()

    log("JAX Configuration:")
    log(f"  Backend: {jax.default_backend()}")
    log(f"  Devices: {jax.devices()}")
    log(f"  Float64 enabled: {jax.config.jax_enable_x64}")
    log(f"  UMFPACK available: {is_umfpack_available()}")
    log()

    # Find benchmark
    from scripts.benchmark_utils import get_vacask_benchmarks
    benchmarks = get_vacask_benchmarks([args.benchmark])
    if not benchmarks:
        log(f"ERROR: Benchmark '{args.benchmark}' not found")
        log("Available benchmarks: rc, graetz, ring, c6288")
        sys.exit(1)

    name, sim_path = benchmarks[0]
    log(f"Benchmark: {name}")
    log(f"Timesteps: {args.timesteps}")
    log()

    results: List[Dict] = []

    # Run JAX native spsolve
    log(f"  Running with JAX native spsolve...")
    jax_result = run_benchmark(sim_path, name, 'jax', args.timesteps)
    results.append(jax_result)
    if jax_result['error']:
        log(f"    ERROR: {jax_result['error']}")
    else:
        log(f"    Done: {jax_result['time_per_step_ms']:.2f}ms/step")
    log()

    # Run UMFPACK
    if is_umfpack_available():
        log(f"  Running with UMFPACK...")
        umf_result = run_benchmark(sim_path, name, 'umfpack', args.timesteps)
        results.append(umf_result)
        if umf_result['error']:
            log(f"    ERROR: {umf_result['error']}")
        else:
            log(f"    Done: {umf_result['time_per_step_ms']:.2f}ms/step")
        log()
    else:
        log("  Skipping UMFPACK (not available)")
        log()

    # Summary
    log("=" * 70)
    log("Summary")
    log("=" * 70)
    log()

    log("| Solver   | Nodes | Steps | Warmup (s) | Total (s) | Per Step (ms) |")
    log("|----------|-------|-------|------------|-----------|---------------|")
    for r in results:
        if r['error']:
            log(f"| {r['solver']:8} | ERROR: {r['error'][:40]} |")
        else:
            log(f"| {r['solver']:8} | {r['nodes']:5} | {r['timesteps']:5} | "
                f"{r['warmup_time_s']:10.2f} | {r['total_time_s']:9.3f} | "
                f"{r['time_per_step_ms']:13.2f} |")
    log()

    # Compare
    if len(results) == 2 and all(r['converged'] for r in results):
        jax_time = results[0]['time_per_step_ms']
        umf_time = results[1]['time_per_step_ms']

        if jax_time < umf_time:
            speedup = umf_time / jax_time
            log(f"JAX native spsolve is {speedup:.1f}x FASTER than UMFPACK")
        else:
            speedup = jax_time / umf_time
            log(f"UMFPACK is {speedup:.1f}x faster than JAX native spsolve")

        log()
        log("Breakdown:")
        log(f"  JAX warmup:    {results[0]['warmup_time_s']:.2f}s (includes JIT compilation)")
        log(f"  UMFPACK warmup: {results[1]['warmup_time_s']:.2f}s (includes symbolic factorization)")

    log()
    log("=" * 70)
    log("Benchmark complete!")
    log("=" * 70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n*** FATAL ERROR ***")
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
