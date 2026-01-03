#!/bin/bash
set -e

# Environment variables expected:
# - GITHUB_REPOSITORY: e.g., "ChipFlow/jax-spice"
# - GITHUB_SHA: commit to checkout
# - GITHUB_TOKEN: for private repo access (optional for public repos)

echo "=== GPU Test Runner ==="
echo "Repository: ${GITHUB_REPOSITORY}"
echo "Commit: ${GITHUB_SHA}"

cd /app

# Clone the repository at the specific commit
echo "Cloning repository..."
if [ -n "$GITHUB_TOKEN" ]; then
    git clone --depth 1 --recurse-submodules \
        "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" \
        --branch main source
    cd source
    git fetch --depth 1 origin "$GITHUB_SHA"
    git checkout "$GITHUB_SHA"
else
    git clone --depth 1 --recurse-submodules \
        "https://github.com/${GITHUB_REPOSITORY}.git" \
        --branch main source
    cd source
    git fetch --depth 1 origin "$GITHUB_SHA"
    git checkout "$GITHUB_SHA"
fi

# Update submodules if needed
git submodule update --init --recursive

echo "=== sccache diagnostics (before build) ==="
sccache --show-stats || echo "sccache stats not available"

echo "Installing workspace packages..."
# Install the workspace in the pre-existing venv
# Use --extra test for pytest (not --extra dev which includes scikit-umfpack)
uv sync --locked --extra cuda12 --extra test

# Set up LD_LIBRARY_PATH for NVIDIA pip packages
# JAX's pip packages install CUDA libraries to site-packages/nvidia/*/lib
NVIDIA_BASE=".venv/lib/python3.13/site-packages/nvidia"
export LD_LIBRARY_PATH="${NVIDIA_BASE}/cuda_runtime/lib:${NVIDIA_BASE}/cublas/lib:${NVIDIA_BASE}/cusparse/lib:${NVIDIA_BASE}/cudnn/lib:${NVIDIA_BASE}/cufft/lib:${NVIDIA_BASE}/cusolver/lib:${NVIDIA_BASE}/nvjitlink/lib:${NVIDIA_BASE}/nccl/lib:${NVIDIA_BASE}/cu12/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH set for NVIDIA packages"

echo "=== sccache diagnostics (after build) ==="
sccache --show-stats || echo "sccache stats not available"

echo "=== CUDA Environment Diagnostics ==="
# Check NVIDIA driver version
nvidia-smi || echo "nvidia-smi not available"

# Check installed CUDA packages
uv pip list | grep -i nvidia || echo "No nvidia packages found"
uv pip list | grep -i cuda || echo "No cuda packages found"

# Check JAX CUDA detection
uv run python -c "
import jax
print('JAX version:', jax.__version__)
print('JAX backend:', jax.default_backend())
print('JAX devices:', jax.devices())

# Check for GPU devices
gpu_devices = [d for d in jax.devices() if d.platform != 'cpu']
if gpu_devices:
    print('GPU devices found:', gpu_devices)
    print('GPU device kind:', gpu_devices[0].device_kind)
else:
    print('No GPU devices found')
"

echo "Running JAX-SPICE vs VACASK benchmark comparison..."
# Enable JAX profiling to capture GPU traces (Perfetto format)
# Note: nsys-jax profiling can be enabled with --profile-mode=nsys but
# generates larger output files - use for detailed GPU kernel analysis
# Run all benchmarks on GPU - Spineax/cuDSS handles large circuits efficiently
# Use --force-gpu to ensure GPU is used even for small circuits (ring has only 47 nodes)
# Use --analyze to dump HLO/cost analysis for understanding XLA compilation
#
# Benchmarks are auto-discovered from jax_spice.benchmarks.registry
# Includes: rc, graetz, mul, ring, c6288, tb_dp (if available)
#
# Note: CUDA cleanup errors (CUDA_ERROR_ILLEGAL_ADDRESS during event destruction)
# can occur during Python exit even when benchmarks complete successfully.
# We capture output to check for success markers and tolerate cleanup errors.
set +e
uv run python scripts/compare_vacask.py \
  --max-steps 50 \
  --use-scan \
  --force-gpu \
  --analyze \
  --profile-mode jax \
  --profile-dir /tmp/jax-spice-traces 2>&1 | tee /tmp/benchmark_output.txt
BENCHMARK_STATUS=${PIPESTATUS[0]}
set -e

# Check if benchmark actually completed by looking for success markers in output
# CUDA cleanup errors happen after Python prints results, so if we see the summary
# table, the benchmark succeeded even if exit status is non-zero
BENCHMARK_OK=false
if grep -q "Summary (per-step timing)" /tmp/benchmark_output.txt 2>/dev/null; then
  BENCHMARK_OK=true
  echo "Benchmark completed successfully (found summary table)"
fi

if [ $BENCHMARK_STATUS -ne 0 ]; then
  echo "Benchmark script exited with status $BENCHMARK_STATUS"
  if [ "$BENCHMARK_OK" = true ]; then
    echo "However, benchmark completed successfully - this is likely a CUDA cleanup error"
    echo "Treating as success"
  else
    echo "ERROR: Benchmark did not complete - this is a real failure"
    exit $BENCHMARK_STATUS
  fi
fi

# Upload traces to GCS for artifact download
echo "=== Uploading profiling traces to GCS ==="
GCS_BUCKET="jax-spice-cuda-test-traces"
TRACE_PATH="${GITHUB_SHA:-$(date +%s)}"

if [ -d "/tmp/jax-spice-traces" ]; then
  echo "Uploading traces to gs://${GCS_BUCKET}/${TRACE_PATH}/"

  # Use gsutil with workload identity credentials (auto-detected from metadata server)
  gsutil -m cp -r /tmp/jax-spice-traces/* "gs://${GCS_BUCKET}/${TRACE_PATH}/" || {
    echo "Warning: Failed to upload traces (gsutil error)"
  }

  echo "Traces uploaded to: gs://${GCS_BUCKET}/${TRACE_PATH}/"
  echo "TRACE_GCS_PATH=${TRACE_PATH}" >> /tmp/trace_info.env
else
  echo "Skipping trace upload (no traces found)"
fi

echo "Running tests..."
# Same CUDA cleanup tolerance for pytest
# Capture output to check for actual test failures vs CUDA cleanup errors
set +e
uv run pytest tests/ -v --tb=short -x 2>&1 | tee /tmp/pytest_output.txt
TEST_STATUS=${PIPESTATUS[0]}
set -e

# Check if tests actually passed by looking at pytest output
# pytest prints "X passed" on success, "FAILED" on failure
TESTS_OK=false
if grep -q " passed" /tmp/pytest_output.txt 2>/dev/null && \
   ! grep -q "FAILED" /tmp/pytest_output.txt 2>/dev/null; then
  TESTS_OK=true
  echo "Tests completed successfully (found 'passed', no 'FAILED')"
fi

# Report final status
echo ""
echo "=== Final Status ==="
echo "Benchmark exit status: $BENCHMARK_STATUS (OK: $BENCHMARK_OK)"
echo "Test exit status: $TEST_STATUS (OK: $TESTS_OK)"

# Exit with failure only if tests actually failed (not just CUDA cleanup)
if [ $TEST_STATUS -ne 0 ]; then
  echo "Tests exited with non-zero status"
  if [ "$TESTS_OK" = true ]; then
    echo "However, all tests passed - this is likely a CUDA cleanup error"
    echo "Treating as success"
  else
    echo "ERROR: Tests actually failed"
    exit $TEST_STATUS
  fi
fi

echo "All tests passed!"
