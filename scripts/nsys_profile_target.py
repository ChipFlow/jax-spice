#!/usr/bin/env python3
"""Target script for nsys GPU profiling - runs circuit simulation.

Usage:
    nsys profile -o profile uv run python scripts/nsys_profile_target.py ring 500

Arguments:
    circuit: One of rc, graetz, mul, ring, c6288 (default: ring)
    timesteps: Number of timesteps to simulate (default: 500)

Use 500+ timesteps so JIT warmup overhead is <5% of total profile.
"""

import argparse
import logging
import sys
from pathlib import Path

import jax

sys.path.insert(0, ".")

# Enable INFO logging so solver selection messages are visible
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

# Import vajax first to auto-configure precision based on backend
from vajax.analysis import CircuitEngine


def main():
    parser = argparse.ArgumentParser(description="nsys profiling target for VAJAX")
    parser.add_argument(
        "circuit",
        nargs="?",
        default="ring",
        choices=["rc", "graetz", "mul", "ring", "c6288"],
        help="Circuit to profile (default: ring)",
    )
    parser.add_argument(
        "timesteps",
        nargs="?",
        type=int,
        default=500,
        help="Number of timesteps to simulate (default: 500)",
    )
    parser.add_argument(
        "--backend",
        default="gpu",
        choices=["cpu", "gpu", "auto"],
        help="Backend to use (default: gpu)",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse solver (for large circuits)",
    )
    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Circuit: {args.circuit}")
    print(f"Timesteps: {args.timesteps}")

    # Explicit solver availability check
    print()
    print("=== Solver Availability ===")
    try:
        from spineax.cudss.dense_baspacho_solver import is_available

        print("  BaSpaCho dense import: OK")
        print(f"  BaSpaCho dense available: {is_available()}")
    except ImportError as e:
        print(f"  BaSpaCho dense import: FAILED ({e})")
    try:
        from spineax.cudss.solver import CuDSSSolver  # noqa: F401

        print("  cuDSS sparse import: OK")
    except ImportError as e:
        print(f"  cuDSS sparse import: FAILED ({e})")
    try:
        from spineax import baspacho_dense_solve as _mod

        print(f"  baspacho_dense_solve C++ module: OK ({_mod})")
    except ImportError as e:
        print(f"  baspacho_dense_solve C++ module: FAILED ({e})")
    print()

    # Find benchmark .sim file
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    sim_path = repo_root / "vendor" / "VACASK" / "benchmark" / args.circuit / "vacask" / "runme.sim"

    if not sim_path.exists():
        print(f"ERROR: Benchmark file not found: {sim_path}")
        sys.exit(1)

    # Setup circuit using CircuitEngine
    print(f"Setting up circuit from {sim_path}...")
    engine = CircuitEngine(sim_path)
    engine.parse()

    print(f"Circuit size: {engine.num_nodes} nodes, {len(engine.devices)} devices")
    print()

    # Timestep from analysis params or default
    dt = engine.analysis_params.get("step", 1e-12)
    print(f"Using dt={dt}")
    print()

    # Prepare (includes 1-step JIT warmup)
    print(f"Preparing ({args.timesteps} timesteps, includes JIT warmup)...")
    engine.prepare(
        t_stop=args.timesteps * dt,
        dt=dt,
        use_sparse=args.sparse,
    )
    print("Prepare complete")
    print()

    # Profiled run — nsys captures everything including warmup above,
    # but with 500+ steps the warmup is a small fraction of total time
    print(f"Starting profiled run ({args.timesteps} timesteps)...")
    result = engine.run_transient()

    print()
    print(f"Completed: {result.num_steps} timesteps")
    print(f"Wall time: {result.stats.get('wall_time', 0):.3f}s")


if __name__ == "__main__":
    main()
