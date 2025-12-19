#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
View JAX profiling traces in Perfetto UI.

This script:
1. Lists available trace files in a directory
2. Opens Perfetto UI in the browser
3. Provides instructions for loading traces

Usage:
    # View traces from latest CI run (default)
    uv run scripts/view_traces.py

    # View traces from a specific workflow run
    uv run scripts/view_traces.py --run 20378600298

    # View traces from a local directory
    uv run scripts/view_traces.py /tmp/jax-spice-traces

    # Download and view traces from GCS
    uv run scripts/view_traces.py --gcs gs://jax-spice-cuda-test-traces/abc123
"""

import argparse
import json
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path


def list_trace_files(trace_dir: Path) -> list[Path]:
    """List all trace files in a directory."""
    trace_files = []
    for pattern in ["*.pb", "*.pb.gz", "*.json", "*.perfetto-trace"]:
        trace_files.extend(trace_dir.glob(pattern))
        trace_files.extend(trace_dir.glob(f"**/{pattern}"))
    return sorted(set(trace_files))


def download_from_github(run_id: str | None = None) -> Path:
    """Download traces from GitHub workflow artifact.

    If run_id is None, downloads from the latest successful GPU Tests run.
    """
    # Find the run ID if not specified
    if run_id is None:
        print("Finding latest GPU Tests workflow run...")
        result = subprocess.run(
            [
                "gh", "run", "list",
                "--workflow=GPU Tests (Cloud Run)",
                "--status=success",
                "--limit=1",
                "--json=databaseId,headSha,createdAt",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error listing workflow runs: {result.stderr}")
            sys.exit(1)

        runs = json.loads(result.stdout)
        if not runs:
            print("No successful GPU Tests runs found.")
            sys.exit(1)

        run_id = str(runs[0]["databaseId"])
        commit = runs[0]["headSha"][:8]
        created = runs[0]["createdAt"]
        print(f"  Found run {run_id} (commit {commit}, {created})")

    # Create download directory
    local_dir = Path(tempfile.gettempdir()) / f"jax-traces-run-{run_id}"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download the artifact
    print(f"Downloading profiling traces from run {run_id}...")
    result = subprocess.run(
        [
            "gh", "run", "download", run_id,
            "--name", f"profiling-traces-*",
            "--dir", str(local_dir),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Try with pattern matching (older gh versions)
        result = subprocess.run(
            [
                "gh", "run", "download", run_id,
                "--pattern", "profiling-traces-*",
                "--dir", str(local_dir),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error downloading artifact: {result.stderr}")
            print()
            print("Make sure you have gh CLI installed and authenticated:")
            print("  gh auth login")
            sys.exit(1)

    return local_dir


def download_from_gcs(gcs_path: str) -> Path:
    """Download traces from GCS to a temporary directory."""
    # Extract bucket and path for unique local dir name
    gcs_path = gcs_path.rstrip('/')
    path_parts = gcs_path.replace('gs://', '').split('/')
    dir_name = path_parts[-1] if len(path_parts) > 1 else 'traces'

    local_dir = Path(tempfile.gettempdir()) / f"jax-traces-{dir_name}"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading traces from {gcs_path}...")

    # Use gsutil rsync for recursive download (handles nested directories)
    result = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_path, str(local_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Try cp -r as fallback
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{gcs_path}/**", str(local_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error downloading traces: {result.stderr}")
            sys.exit(1)

    return local_dir


def main():
    parser = argparse.ArgumentParser(
        description="View JAX profiling traces in Perfetto UI"
    )
    parser.add_argument(
        "trace_dir",
        type=str,
        nargs="?",
        help="Directory containing trace files (default: download from latest CI run)",
    )
    parser.add_argument(
        "--run",
        type=str,
        nargs="?",
        const="latest",
        help="Download from GitHub workflow run (default: latest successful run)",
    )
    parser.add_argument(
        "--gcs",
        type=str,
        help="Download traces from GCS path (e.g., gs://bucket/path)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser, just list trace files",
    )
    args = parser.parse_args()

    # Determine trace source
    if args.gcs:
        trace_dir = download_from_gcs(args.gcs)
    elif args.run is not None:
        # --run with optional run ID
        run_id = None if args.run == "latest" else args.run
        trace_dir = download_from_github(run_id)
    elif args.trace_dir:
        trace_dir = Path(args.trace_dir)
    else:
        # Default: download from latest CI run
        trace_dir = download_from_github(None)

    if not trace_dir.exists():
        print(f"Error: Trace directory does not exist: {trace_dir}")
        print()
        print("To generate traces, run benchmarks with profiling enabled:")
        print("  JAX_SPICE_PROFILE_JAX=1 uv run python scripts/compare_vacask.py --profile")
        print()
        print("Or run on Cloud Run GPU:")
        print("  uv run scripts/profile_gpu_cloudrun.py")
        sys.exit(1)

    # List trace files
    trace_files = list_trace_files(trace_dir)

    print("=" * 60)
    print("JAX-SPICE Profiling Trace Viewer")
    print("=" * 60)
    print()
    print(f"Trace directory: {trace_dir}")
    print()

    if not trace_files:
        print("No trace files found.")
        print()
        print("Expected file types: .pb, .pb.gz, .json, .perfetto-trace")
        sys.exit(1)

    print(f"Found {len(trace_files)} trace file(s):")
    for f in trace_files[:10]:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")
    if len(trace_files) > 10:
        print(f"  ... and {len(trace_files) - 10} more")
    print()

    print("To view traces in Perfetto:")
    print("  1. Open https://ui.perfetto.dev/")
    print("  2. Click 'Open trace file' or drag & drop")
    print(f"  3. Select file(s) from: {trace_dir}")
    print()

    if not args.no_browser:
        print("Opening Perfetto UI in browser...")
        webbrowser.open("https://ui.perfetto.dev/")
    else:
        print("(Browser opening skipped with --no-browser)")


if __name__ == "__main__":
    main()
