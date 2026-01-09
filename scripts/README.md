# Scripts Directory

Collection of scripts for development, analysis, benchmarking, and investigation.

---

## 🎯 Phase 1 Starting Point

**test_vacask_osdi_psp103.py** ⭐
- Loads VACASK OSDI library via ctypes
- **This is where we start** for the fresh implementation
- Status: Incomplete stub, needs full implementation
- Goal: Load capacitor.osdi, call init/eval, validate results

See `../planning/IMPLEMENTATION_PLAN.md` → Phase 1 for details.

---

## 📂 Script Categories

### Investigation Scripts (Archive After Phase 1)

These scripts were created during the January 2026 investigation that discovered the parameter mapping bug and proved PHI nodes work correctly. They'll be archived once Phase 1 is complete.

**Parameter mapping discovery**:
- `check_init_param_arrays.py` ⭐ - Found duplicate 'c' name collision
- `check_init_param_mapping.py` - Analyzed parameter mapping
- `fix_and_test_init.py` ⭐ - Proved PHI nodes work with correct mapping!
- `identify_v18_v38.py` - Identified what PHI operands represent

**Code generation investigation**:
- `check_generated_phi_code.py` - Showed PHI code generation works
- `inspect_phi_nodes.py` - Analyzed PHI node structure

**Cache investigation**:
- `check_cache_mapping.py` - Analyzed cache system
- `check_init_cache_values.py` - Checked cache value computation
- `simple_check_cache.py` - Simple cache checks

**MIR investigation**:
- `check_mir_structure.py` - Understood MIR dict format
- `investigate_mir_derivatives.py` - Looked for derivative calls
- `investigate_optbarrier.py` - Traced through optbarrier operations
- `trace_v37_v40_origin.py` - Traced cache variable origins
- `check_native_init_params.py` - Checked native parameter handling
- `identify_v37_v40.py` - Identified cache variables

### Production Scripts

**Benchmarking**:
- `benchmark_utils.py` - Benchmark utilities
- `run_ring_long_transient.py` - Long transient simulation
- `runme.py` - General runner script

**GPU Profiling**:
- `profile_gpu.py` - Profile GPU performance locally
- `profile_gpu_cloudrun.py` - Profile GPU on Cloud Run
- `profile_cpu.py` - Profile CPU performance
- `profile_nsys_cloudrun.py` - Nsys profiling on Cloud Run
- `nsys_profile_target.py` - Nsys profiling target
- `run_gpu_tests.py` - GPU test runner

**Comparison & Analysis**:
- `compare_openvaf_vs_osdi.py` - Compare OpenVAF vs OSDI
- `compare_vacask.py` - Compare against VACASK
- `compare_results.py` - General result comparison
- `compare_device_params.py` - Device parameter comparison
- `compare_dc_operating_point.py` - DC operating point comparison
- `compare_psp103_dc_eval.py` - PSP103 DC evaluation comparison
- `compare_ring_debug.py` - Ring oscillator debug comparison

**Debugging & Tracing**:
- `trace_dc_solver_calls.py` - Trace DC solver calls
- `trace_single_timestep.py` - Trace single timestep execution
- `trace_q_explosion.py` - Trace charge explosion issues
- `trace_Q_values.py` - Trace charge values
- `view_traces.py` - View execution traces
- `dump_model_param_setup_mir.py` - Dump MIR for model param setup

**DC Analysis**:
- `check_dc_stability.py` - Check DC stability
- `check_jacobian_stiffness.py` - Check Jacobian stiffness

**Waveform Analysis**:
- `plot_ring_vi.py` - Plot ring oscillator voltage/current
- `run_ring_with_q_debug.py` - Run ring with charge debugging
- `simple_q_debug.py` - Simple charge debugging

**Utilities**:
- `parse_psp103_model_card.py` - Parse PSP103 model cards
- `check_pulse_periodicity.py` - Check pulse signal periodicity

### Generated Code Examples

These are snapshots of generated code for reference:
- `generated_eval_psp103.py` - PSP103 eval function
- `generated_eval_bsim4.py` - BSIM4 eval function
- `generated_setup_instance_psp103.py` - PSP103 setup_instance
- `generated_setup_instance_bsim4.py` - BSIM4 setup_instance
- `generated_setup_instance_capacitor_fixed.py` - Capacitor setup with fix

### Analysis Tools

**vacask/**:
- `parse_debug_output.py` - Parse VACASK debug output
- `extract_comparison_data.py` - Extract comparison data
- `generate_graph.py` - Generate visualization graphs
- `README.md` - Documentation

Documentation moved to `docs/reference/osdi-vacask/`.

---

## 🗄️ Archive Directories

### archive/november-2024-debug/

Contains 24 scripts from the broken November 2024 implementation:
- 11 debug_*.py scripts (debugging broken run_init_eval)
- 12 test_*.py scripts (testing against broken reference)
- 1 validate_codegen_vs_native.py (validated against broken code!)

See `scripts/archive/november-2024-debug/README.md` for details.

**Why archived**: These debugged symptoms of the broken implementation that validated against its own buggy code instead of OSDI ground truth.

### archive/investigation-jan-2026/

Ready for investigation scripts once Phase 1 is complete. See README for what will go here.

---

## 📋 Usage Guide by Phase

### Phase 1: OSDI Interface (Current)

**Start here**:
```bash
# Read the incomplete stub
cat scripts/test_vacask_osdi_psp103.py

# Reference docs
cat docs/reference/osdi-vacask/SIMULATOR_INTERNALS.md
cat docs/reference/osdi-vacask/osdi_parameter_architecture.md
```

**Relevant investigation scripts** (for reference):
- `check_init_param_arrays.py` - Shows the parameter mapping bug we found
- `fix_and_test_init.py` - Shows PHI nodes work correctly

### Phase 2: Rebuild openvaf-py

**After rebuild, test with**:
```bash
python scripts/compare_openvaf_vs_osdi.py
```

### Phase 3-7: Code Generation & Validation

**Use for comparison**:
- `compare_vacask.py` - Compare full simulation vs VACASK
- `compare_openvaf_vs_osdi.py` - Compare code gen vs OSDI
- `compare_device_params.py` - Compare parameter handling

**Use for debugging**:
- `trace_*.py` scripts for execution tracing
- `check_*.py` scripts for validation
- `profile_*.py` scripts for performance analysis

---

## 🧹 Cleanup Policy

**Investigation scripts will be archived** to `archive/investigation-jan-2026/` once Phase 1 is complete and we no longer need them for reference.

**Generated code examples** serve as reference snapshots and will remain until they're superseded by the new implementation.

**Production scripts** stay active as they're used for ongoing development and testing.

---

## 📖 Key References

- **Phase 1 plan**: `../planning/IMPLEMENTATION_PLAN.md` → Phase 1 section
- **OSDI API**: `docs/reference/osdi-vacask/SIMULATOR_INTERNALS.md`
- **Parameter mapping bug**: `docs/PHI_NODE_BUG.md`
- **Overall context**: `docs/RESTART_PROMPT.md`

---

## 🎯 Current Focus

**Phase 1: OSDI ctypes Interface**

Start with `test_vacask_osdi_psp103.py` and build a working OSDI interface that can:
1. Load capacitor.osdi
2. Call init function with correct parameters
3. Call eval function with terminal voltages
4. Validate results match VACASK output

This establishes OSDI as our ground truth for all future validation.
