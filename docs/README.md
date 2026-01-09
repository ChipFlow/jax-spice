# JAX-SPICE Documentation

**Start here for context on the current implementation effort.**

---

## 🎯 Starting Fresh (January 2026)

Our November 2024 MIR-to-Python implementation was fundamentally broken - it validated against its own buggy code instead of OSDI ground truth. We're restarting with a clean approach.

### Essential Reading (in order)

1. **[RESTART_PROMPT.md](RESTART_PROMPT.md)** ⭐ **START HERE**
   - Why we're restarting the code generation effort
   - What to ignore (run_init_eval = our old broken code)
   - What we learned (PHI nodes work correctly!)
   - Where to start (OSDI ctypes interface)

2. **[PHI_NODE_BUG.md](PHI_NODE_BUG.md)** - Technical reference
   - PHI node implementation deep-dive
   - Why code generation works correctly
   - Parameter mapping bug explanation and fix
   - Read when implementing code generation (Phase 3+)

---

## 📚 Reference Documentation

External reference material for OpenVAF, OSDI, and VACASK.

### OSDI/VACASK Reference
**Location**: `reference/osdi-vacask/`

Essential for Phase 1 (OSDI interface):
- **SIMULATOR_INTERNALS.md** (30KB) - OSDI API, descriptors, function signatures
- **osdi_parameter_architecture.md** - Parameter handling in OSDI
- **vacask_osdi_inputs.md** - Input structure for init/eval functions

Understanding OpenVAF compilation:
- **CACHE_SLOTS_ANALYSIS.md** - Init/eval cache system
- **VACASK_MIR_TO_OSDI_PIPELINE.md** - VA → MIR → OSDI compilation

Analysis and formats:
- **ANALYSIS.md** - Ring benchmark analysis
- **VACASK-ANALYSIS.md** - Circuit analysis examples
- **vacask_sim_format.md** - VACASK .sim netlist format

See [reference/osdi-vacask/README.md](reference/osdi-vacask/README.md) for details.

### OpenVAF Internals
**Location**: `reference/openvaf/`

- **OPENVAF_MIR_CONSTANTS.md** - Pre-allocated MIR constants (v0-v7)

See [reference/openvaf/README.md](reference/openvaf/README.md) for details.

---

## 📖 General Documentation

### Code Generation & Metadata
- **CODEGEN_METADATA_API.md** - Metadata extraction API (needs update for param_given fix)
- **PARAMETER_MAPPING_SOLUTION.md** - Parameter mapping solution (needs update)

### Architecture & Development
- **ARCHITECTURE.md** - Overall system architecture
- **DEVELOPMENT.md** - Development workflow and conventions
- **GPU_OPTIMIZATION.md** - GPU optimization strategies
- **SPARSE_SOLVER_ANALYSIS.md** - Sparse matrix solver details
- **CONTRIBUTING.md** - Contributing guidelines
- **TODO.md** - General TODOs

---

## 🗄️ Archives

**Location**: `archive/`

### November 2024 Implementation
**Location**: `archive/november-2024/`

Documentation from the broken November implementation:
- PHASE3_COMPLETE_SUMMARY.md
- COMPLEX_MODELS_SUCCESS.md (false positives!)
- And 5 other docs

See [archive/november-2024/README.md](archive/november-2024/README.md) for why these were archived.

---

## 🚀 Quick Start

### For Fresh Context (New Session)
```bash
# Read the restart prompt for context
cat docs/RESTART_PROMPT.md

# Check active work in planning/
cat planning/IMPLEMENTATION_PLAN.md
```

### Key Reference Documents
```bash
# OSDI API reference (for Phase 1 interface work)
cat docs/reference/osdi-vacask/SIMULATOR_INTERNALS.md

# Technical deep-dive on PHI nodes and parameter mapping
cat docs/PHI_NODE_BUG.md
```

---

## 📋 Reading by Topic

### OSDI Interface Work
Essential references:
1. RESTART_PROMPT.md (context)
2. reference/osdi-vacask/SIMULATOR_INTERNALS.md (OSDI API)
3. reference/osdi-vacask/osdi_parameter_architecture.md (parameters)
4. reference/osdi-vacask/vacask_osdi_inputs.md (input structure)

### Code Generation Work
Essential references:
1. PHI_NODE_BUG.md (how PHI nodes work, parameter mapping fix)
2. reference/osdi-vacask/CACHE_SLOTS_ANALYSIS.md (cache system)
3. CODEGEN_METADATA_API.md (metadata extraction)
4. PARAMETER_MAPPING_SOLUTION.md (parameter handling)

---

## 🎯 The Mission

**Use VACASK OSDI as ground truth, not our own code.**

Build incrementally:
- Simple (capacitor) before complex (PSP103)
- Validate every step against OSDI
- No false positives, no confusion

**Starting point**: `scripts/test_vacask_osdi_psp103.py`

See `../planning/` for current implementation plans.
