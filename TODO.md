# jax-spice TODO

Central tracking for development tasks and known issues.

## High Priority

### VACASK Test Integration
Create tests that use VACASK's test cases with openvaf_jax.

**Tasks**:
- [ ] **Create test suite using VACASK sim files**
  - Parse VACASK `.sim` files (parser complete: 37/37 pass)
  - Compile VA models with openvaf_jax instead of loading OSDI
  - Compare JAX results with expected values from embedded Python

- [ ] **Test cases to implement**:
  - `test_resistor.sim`: V=1V, R=2kΩ, mfactor=3 → I = 1.5mA
  - `test_diode.sim`: Forward/reverse bias with parameter sweeps
  - `test_capacitor.sim`: Capacitive charging
  - `test_inductor.sim`: Inductive behavior
  - `test_op.sim`: Operating point analysis
  - `test_inverter.sim`: MOSFET inverter

## Medium Priority

### openvaf_jax Complex Model Support
The JAX translator produces NaN outputs for complex models due to init variable handling.

**Root cause**: Complex models (BSIM3/4/6, HiSIM, HICUM, etc.) have init functions that compute many cached values. The JAX translator expects these as inputs, but with default values they're wrong → NaN.

**Tasks**:
- [ ] **Use equivalent approach as OSDI compile uses**
  - see docs/vacask_osdi_inputs.md

**Affected models** (currently xfailed in tests):
- BSIM3, BSIM4, BSIM6, BSIMBulk, BSIMCMG, BSIMSOI
- HiSIM2, HiSIMHV
- HICUM L2
- MEXTRAM

**Working models** (should remove xfail markers):
- PSP102, PSP103, JUNCAP
- diode_cmc
- EKV


### GPU Solver Convergence
The GPU solver has convergence issues with circuits containing floating nodes (e.g., AND gates with series NMOS stacks).

**Root cause**: Autodiff-computed Jacobians give extremely small `gds` (~1e-16 S) in cutoff, while VACASK enforces a minimum `gds` (~1e-9 S). This creates near-singular Jacobians.

**Tasks**:
- [ ] **Integrate analytical Jacobian into GPU solver** (see `docs/gpu_solver_jacobian.md`)
  - Use `openvaf_jax`-generated functions that return both residual and Jacobian
  - Build Jacobian from analytical stamps instead of autodiff
  - Files: `jax_spice/analysis/dc_gpu.py`, `jax_spice/analysis/transient_gpu.py`

- [ ] **Test AND gate convergence with analytical Jacobian**
  - Test circuit: `and_test` from `c6288.sim` (6 MOSFETs, floating `int` node)
  - VACASK converges in 34 iterations; GPU solver diverges

- [ ] **Benchmark full C6288 multiplier** on GPU with analytical Jacobians
  - 5123 nodes, 10112 MOSFETs
  - Currently ~30min on CPU, should be much faster on GPU

## Low Priority

### Documentation
- [ ] **Update README** with current project status
- [ ] **Add architecture overview** diagram
- [ ] **Document openvaf_jax API** for external users

### Code Cleanup
- [ ] **Remove xfail markers** from PSP/JUNCAP/diode_cmc tests (they pass)
- [ ] **Consolidate test files** in `openvaf-py/tests/` (some at root level)

### Build System
- [ ] **Upstream VACASK macOS fixes** to original repo
  - Current workaround: `robtaylor/VACASK` fork with `macos-fixes` branch
  - Fixes: C++20, PTBlockSequence, VLA→vector, KLU destructor, <numbers> header, CMake var escaping

## Completed

- [x] ~~Fix VACASK netlist parser~~ (all 37 test files pass)
  - Added @if/@endif directive handling
  - Added vector parameter `[...]` support
  - Fixed title parsing for keywords

- [x] ~~Fix multi-way PHI nodes in openvaf_jax~~
  - MOSFET JAX output now matches MIR interpreter to 6 significant figures
  - Added `_build_multi_way_phi()` for >2 predecessor blocks

- [x] ~~Fix PMOS current sign convention~~ in GPU solvers
  - Both `dc_gpu.py` and `transient_gpu.py` updated

- [x] ~~Add gds_min leakage~~ to GPU MOSFET model (partial fix)

- [x] ~~Document VACASK OSDI input handling~~ (`docs/vacask_osdi_inputs.md`)

- [x] ~~Add OpenVAF/VACASK build scripts~~ for macOS
  - `scripts/build_openvaf.sh`
  - `scripts/build_vacask.sh`

## Reference

### Key Files
| Purpose | Location |
|---------|----------|
| GPU DC solver | `jax_spice/analysis/dc_gpu.py` |
| GPU transient solver | `jax_spice/analysis/transient_gpu.py` |
| OpenVAF→JAX translator | `openvaf-py/openvaf_jax.py` |
| VACASK parser | `jax_spice/netlist/parser.py` |
| Jacobian issue analysis | `docs/gpu_solver_jacobian.md` |

### Test Commands
```bash
# Run all tests
JAX_PLATFORMS=cpu uv run pytest tests/ -v

# Run openvaf-py tests
cd openvaf-py && JAX_PLATFORMS=cpu ../.venv/bin/python -m pytest tests/ -v

# Run GPU benchmarks (slow)
JAX_PLATFORMS=cpu RUN_GPU_BENCHMARKS=1 uv run pytest tests/test_transient_gpu.py -v

# Run Cloud Run GPU tests
uv run scripts/run_gpu_tests.py --watch
```
