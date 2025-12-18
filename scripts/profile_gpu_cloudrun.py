#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Run JAX-SPICE vs VACASK benchmark comparison on Cloud Run GPU.

This script:
1. Triggers a Cloud Run job that runs scripts/compare_vacask.py
2. Streams logs and waits for completion
3. Downloads the benchmark results

Usage:
    uv run scripts/profile_gpu_cloudrun.py [--benchmark rc,graetz,ring,c6288]

Prerequisites:
    - gcloud CLI authenticated with access to jax-spice-cuda-test project
"""

import argparse
import base64
import subprocess
import sys
import time


GCP_PROJECT = "jax-spice-cuda-test"
GCP_REGION = "us-central1"
JOB_NAME = "jax-spice-gpu-benchmark"


def run_cmd(
    cmd: list[str], check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    """Run a command and optionally capture output."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    if capture:
        stdout = result.stdout.strip() if result.stdout else None
        stderr = result.stderr.strip() if result.stderr else None
        return (result, stdout, stderr)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run JAX-SPICE vs VACASK benchmark on Cloud Run GPU"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="rc,graetz,ring,c6288",
        help="Comma-separated benchmarks to run (default: rc,graetz,ring,c6288)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum timesteps per benchmark (default: 200)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("JAX-SPICE vs VACASK Benchmark on Cloud Run GPU")
    print("=" * 60)
    print(f"Benchmarks: {args.benchmark}")
    print(f"Max steps: {args.max_steps}")
    print()

    # Check gcloud auth
    print("[1/4] Checking gcloud authentication...")
    (result, stdout, stderr) = run_cmd(
        ["gcloud", "auth", "print-identity-token"], capture=True, check=False
    )
    if result.returncode != 0:
        print("Error: Not authenticated with gcloud. Run: gcloud auth login")
        sys.exit(1)
    print("  Authenticated")
    print()

    # Build the compare_vacask.py command
    compare_cmd = [
        "uv", "run", "python", "scripts/compare_vacask.py",
        "--benchmark", args.benchmark,
        "--max-steps", str(args.max_steps),
        "--use-scan",
    ]
    compare_cmd_str = " ".join(compare_cmd)

    # Create the bash script that runs on Cloud Run
    benchmark_script = f'''#!/bin/bash
set -e

cd /app

# Clone repo at main
git clone --depth 1 --recurse-submodules https://github.com/ChipFlow/jax-spice.git source
cd source

# Install deps (with CUDA support)
uv sync --locked --extra cuda12

# Check GPU detection
echo "=== Checking JAX GPU Detection ==="
uv run python -c "import jax; print('Backend:', jax.default_backend()); print('Devices:', jax.devices())"

echo ""
echo "=== Starting Benchmark Comparison ==="
echo "Benchmarks: {args.benchmark}"
echo "Max steps: {args.max_steps}"
echo ""

# Run the benchmark comparison
{compare_cmd_str}

echo ""
echo "=== Benchmark Complete ==="
'''

    # Create or update the Cloud Run job
    print("[2/4] Creating/updating Cloud Run job...")

    # Use base64-encoded script to avoid shell escaping issues
    script_b64 = base64.b64encode(benchmark_script.encode()).decode()
    job_args = f"echo {script_b64} | base64 -d | bash"

    job_cmd = [
        "gcloud", "run", "jobs", "create", JOB_NAME,
        f"--region={GCP_REGION}",
        f"--project={GCP_PROJECT}",
        "--image=us-central1-docker.pkg.dev/jax-spice-cuda-test/ghcr-remote/chipflow/jax-spice/gpu-base:latest",
        "--execution-environment=gen2",
        "--gpu=1",
        "--gpu-type=nvidia-l4",
        "--no-gpu-zonal-redundancy",
        "--cpu=4",
        "--memory=16Gi",
        "--task-timeout=30m",
        "--max-retries=0",
        "--command=bash",
        f"--args=-c,{job_args}",
    ]

    result = run_cmd(job_cmd, check=False)
    if result.returncode != 0:
        job_cmd[3] = "update"
        run_cmd(job_cmd)
    print()

    # Execute the job
    print("[3/4] Executing Cloud Run job...")
    (result, exec_id, _) = run_cmd(
        [
            "gcloud", "run", "jobs", "execute", JOB_NAME,
            f"--region={GCP_REGION}",
            f"--project={GCP_PROJECT}",
            "--async",
            "--format=value(metadata.name)",
        ],
        capture=True,
    )

    print(f"Job Execution ID: {exec_id}")

    # Start log tailing in background
    log_proc = subprocess.Popen(
        [
            "gcloud", "beta", "run", "jobs", "executions", "logs", "tail",
            exec_id,
            f"--region={GCP_REGION}",
            f"--project={GCP_PROJECT}",
        ],
        stdout=None,
        stderr=None,
    )

    # Poll job status until completion
    print("[4/4] Waiting for job to complete (streaming logs)...")
    print()
    job_succeeded = False
    try:
        while True:
            time.sleep(5)
            result = subprocess.run(
                [
                    "gcloud", "run", "jobs", "executions", "describe", exec_id,
                    f"--region={GCP_REGION}",
                    f"--project={GCP_PROJECT}",
                    "--format=value(status.conditions[0].type,status.conditions[0].status)",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                continue

            output = result.stdout.strip()
            if not output:
                continue

            parts = output.split(";")
            if len(parts) >= 2:
                condition_type, status = parts[0], parts[1]
                if condition_type == "Completed":
                    job_succeeded = status == "True"
                    break
    finally:
        log_proc.terminate()
        try:
            log_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_proc.kill()

    print()
    if not job_succeeded:
        print("Job failed! Check logs with:")
        print(f"  gcloud beta run jobs executions logs read {exec_id} --region={GCP_REGION}")
        sys.exit(1)

    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    print()
    print("To view full logs:")
    print(f"  gcloud beta run jobs executions logs read {exec_id} --region={GCP_REGION}")


if __name__ == "__main__":
    main()
