#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "jax",
#     "jax-spice",
# ]
# ///
"""Benchmark AdaptiveStrategy (Python loop) vs AdaptiveWhileLoopStrategy (lax.while_loop).

This script performs a fair comparison now that both implementations use jnp.where
for all conditionals, eliminating the previous bool() transitions in the Python loop.

Usage:
    uv run scripts/benchmark_adaptive_strategies.py [--steps N] [--warmup-runs N]
"""

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp


def run_benchmark(strategy_class, engine_factory, t_stop: float, dt: float,
                  max_steps: int, warmup_runs: int = 2, timed_runs: int = 3):
    """Benchmark a strategy with warmup and timed runs."""

    times = []
    stats_list = []

    for run in range(warmup_runs + timed_runs):
        # Create fresh engine for each run
        engine = engine_factory()
        engine.parse()

        # Create strategy
        strategy = strategy_class(engine)

        # Run
        start = time.perf_counter()
        result_times, voltages, currents, stats = strategy.run(
            t_stop=t_stop, dt=dt, max_steps=max_steps
        )
        jax.block_until_ready(result_times)
        elapsed = time.perf_counter() - start

        if run >= warmup_runs:
            times.append(elapsed)
            stats_list.append(stats)

    return times, stats_list


def main():
    parser = argparse.ArgumentParser(description="Benchmark adaptive strategies")
    parser.add_argument("--t-stop", type=float, default=10e-9, help="Simulation stop time")
    parser.add_argument("--dt", type=float, default=1e-12, help="Initial timestep")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max simulation steps")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Number of warmup runs")
    parser.add_argument("--timed-runs", type=int, default=3, help="Number of timed runs")
    args = parser.parse_args()

    from jax_spice.analysis.engine import CircuitEngine
    from jax_spice.analysis.transient.adaptive import (
        AdaptiveStrategy,
        AdaptiveWhileLoopStrategy,
        AdaptiveConfig,
    )

    print("="*70)
    print("Adaptive Strategy Benchmark")
    print("="*70)
    print(f"Circuit: ring oscillator")
    print(f"t_stop: {args.t_stop:.2e}s, dt: {args.dt:.2e}s, max_steps: {args.max_steps}")
    print(f"Warmup runs: {args.warmup_runs}, Timed runs: {args.timed_runs}")
    print()

    def engine_factory():
        return CircuitEngine("vendor/VACASK/benchmark/ring/vacask/runme.sim")

    # Benchmark Python loop version (AdaptiveStrategy)
    print("Benchmarking AdaptiveStrategy (Python loop with 1 float() transition)...")
    py_times, py_stats = run_benchmark(
        AdaptiveStrategy, engine_factory,
        args.t_stop, args.dt, args.max_steps,
        args.warmup_runs, args.timed_runs
    )

    print("Benchmarking AdaptiveWhileLoopStrategy (lax.while_loop, 0 transitions)...")
    while_times, while_stats = run_benchmark(
        AdaptiveWhileLoopStrategy, engine_factory,
        args.t_stop, args.dt, args.max_steps,
        args.warmup_runs, args.timed_runs
    )

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print("\nAdaptiveStrategy (Python loop):")
    print(f"  Times: {[f'{t:.3f}s' for t in py_times]}")
    print(f"  Mean:  {sum(py_times)/len(py_times):.3f}s")
    print(f"  Steps: {py_stats[0]['accepted_steps']}, Rejected: {py_stats[0]['rejected_steps']}")
    print(f"  NR iterations: {py_stats[0]['total_nr_iterations']}")

    print("\nAdaptiveWhileLoopStrategy (lax.while_loop):")
    print(f"  Times: {[f'{t:.3f}s' for t in while_times]}")
    print(f"  Mean:  {sum(while_times)/len(while_times):.3f}s")
    print(f"  Steps: {while_stats[0]['accepted_steps']}, Rejected: {while_stats[0]['rejected_steps']}")
    print(f"  NR iterations: {while_stats[0]['total_nr_iterations']}")

    # Comparison
    py_mean = sum(py_times) / len(py_times)
    while_mean = sum(while_times) / len(while_times)
    speedup = py_mean / while_mean

    print("\n" + "-"*70)
    print("COMPARISON")
    print("-"*70)
    if speedup > 1:
        print(f"lax.while_loop is {speedup:.2f}x FASTER than Python loop")
    else:
        print(f"Python loop is {1/speedup:.2f}x FASTER than lax.while_loop")

    print(f"\nPer-step overhead:")
    py_steps = py_stats[0]['accepted_steps'] + py_stats[0]['rejected_steps']
    while_steps = while_stats[0]['accepted_steps'] + while_stats[0]['rejected_steps']
    print(f"  Python loop:     {py_mean/py_steps*1000:.2f} ms/step")
    print(f"  lax.while_loop:  {while_mean/while_steps*1000:.2f} ms/step")

    # Verify results match
    print("\n" + "-"*70)
    print("VERIFICATION")
    print("-"*70)
    steps_match = py_stats[0]['accepted_steps'] == while_stats[0]['accepted_steps']
    print(f"Steps match: {steps_match}")
    if not steps_match:
        print(f"  Python: {py_stats[0]['accepted_steps']}, While: {while_stats[0]['accepted_steps']}")


if __name__ == "__main__":
    main()
