#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Run GPU tests on Google Cloud Run with GPU support.

This script triggers the test-gpu-cloudrun workflow manually, allowing you to
run GPU tests outside of CI. Useful for testing large circuits like C6288
that may timeout in regular CI runs.

Usage:
    uv run scripts/run_gpu_tests.py [--watch] [--full]

Options:
    --watch    Wait for the workflow to complete and show results
    --full     Run full benchmark suite (including slow C6288 tests)
"""

import subprocess
import sys
import argparse


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run GPU tests on Cloud Run")
    parser.add_argument("--watch", action="store_true", help="Wait for workflow completion")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    args = parser.parse_args()

    # Check if gh is available
    result = subprocess.run(["which", "gh"], capture_output=True)
    if result.returncode != 0:
        print("Error: gh CLI not found. Install it with: brew install gh")
        sys.exit(1)

    # Get current branch
    result = run_command(["git", "branch", "--show-current"])
    branch = result.stdout.strip()
    print(f"Current branch: {branch}")

    # Trigger the workflow
    print("\nTriggering test-gpu-cloudrun workflow...")
    workflow_cmd = [
        "gh",
        "workflow",
        "run",
        "test-gpu-cloudrun.yml",
        "--ref",
        branch,
    ]

    if args.full:
        # Pass input to run full benchmarks
        workflow_cmd.extend(["-f", "run_full_benchmarks=true"])

    run_command(workflow_cmd)
    print("Workflow triggered successfully!")

    if args.watch:
        print("\nWaiting for workflow to start...")
        # Give it a moment to register
        import time

        time.sleep(3)

        # Get the latest run
        result = run_command(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                "test-gpu-cloudrun.yml",
                "--limit",
                "1",
                "--json",
                "databaseId,status,conclusion",
            ]
        )

        import json

        runs = json.loads(result.stdout)
        if runs:
            run_id = runs[0]["databaseId"]
            print(f"\nWatching run {run_id}...")
            run_command(["gh", "run", "watch", str(run_id)])

            # Show the logs if it failed
            result = run_command(
                [
                    "gh",
                    "run",
                    "view",
                    str(run_id),
                    "--json",
                    "conclusion",
                ]
            )
            conclusion = json.loads(result.stdout).get("conclusion")
            if conclusion != "success":
                print("\nWorkflow failed. Fetching logs...")
                run_command(["gh", "run", "view", str(run_id), "--log-failed"], check=False)
    else:
        print("\nTo watch the workflow progress, run:")
        print("  gh run watch")
        print("\nOr check the Actions tab:")
        print("  gh run list --workflow test-gpu-cloudrun.yml")


if __name__ == "__main__":
    main()
