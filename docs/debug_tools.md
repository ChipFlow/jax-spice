# Debug Tools Reference

This document describes the debugging utilities in `jax_spice.debug` for troubleshooting OSDI vs JAX discrepancies.

## Quick Start

```python
from jax_spice.debug import quick_compare, inspect_model

# Compare OSDI vs JAX at a bias point
result = quick_compare(
    va_path="vendor/OpenVAF/integration_tests/PSP102/psp102.va",
    osdi_path="/tmp/osdi_jax_test_cache/psp102.osdi",
    params={'TYPE': 1, 'W': 1e-6, 'L': 1e-7},
    voltages=[0.5, 0.6, 0.0, 0.0],
)
print(result)

# Inspect MIR structure
inspect_model("vendor/OpenVAF/integration_tests/PSP102/psp102.va")
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `model_comparison` | Compare OSDI vs JAX outputs (residuals, Jacobians, cache) |
| `mir_inspector` | Inspect MIR data (params, PHI nodes, constants) |
| `jacobian` | Format-aware Jacobian comparison (OSDI sparse vs JAX dense) |
| `mir_tracer` | Trace value flow through MIR |
| `param_analyzer` | Analyze parameter kinds and OSDI comparison |
| `mir_analysis` | CFG analysis with networkx (optional dependency) |

---

## Model Comparison (`model_comparison.py`)

### ModelComparator

Full-featured comparison between OSDI and JAX implementations.

```python
from jax_spice.debug import ModelComparator

comparator = ModelComparator(
    va_path="path/to/model.va",
    osdi_path="path/to/model.osdi",
    params={'TYPE': 1, 'W': 1e-6, 'L': 1e-7},
    temperature=300.0,
)

# Compare at a single bias point
result = comparator.compare_at_bias([0.5, 0.6, 0.0, 0.0])
print(result)
print(f"Passed: {result.passed}")
print(f"Issues: {result.issues}")

# Print side-by-side residual table
comparator.print_residual_table([0.5, 0.6, 0.0, 0.0])

# Analyze cache for potential issues
cache_analysis = comparator.analyze_cache()
print(cache_analysis)

# Sweep comparison
results = comparator.sweep_comparison(
    base_voltages=[0.5, 0.0, 0.0, 0.0],
    sweep_index=1,  # Sweep Vgs
    sweep_values=[0.0, 0.3, 0.6, 0.9, 1.2],
)
```

### CacheAnalysis

Detects potential issues in JAX cache values:

- **inf/nan detection**: Catches numerical instabilities
- **Large values**: Flags values > 1e10 that might cause overflow
- **Temperature-related**: Finds VT values (0.025-0.030) and their implied temperatures

```python
cache = comparator.analyze_cache()
print(f"Cache size: {cache.size}")
print(f"Non-zero: {cache.nonzero_count}")
print(f"Has inf: {cache.has_inf}")
print(f"Has nan: {cache.has_nan}")

# Check temperature values
for idx, val, implied_t in cache.temperature_related:
    print(f"cache[{idx}] = {val:.6f} implies T = {implied_t:.1f}K")
```

---

## MIR Inspector (`mir_inspector.py`)

### MIRInspector

Examine MIR (Mid-level IR) structure for debugging translation issues.

```python
from jax_spice.debug import MIRInspector

inspector = MIRInspector("path/to/model.va")

# Overall statistics
inspector.print_mir_stats()

# Parameter summary
inspector.print_param_summary('eval')  # or 'init'

# PHI node analysis
inspector.print_phi_summary('eval')

# Find TYPE parameter (NMOS/PMOS models)
inspector.print_type_param_info()
```

### Finding Specific Values

```python
# Find constants near a value (e.g., P_CELSIUS0 = 273.15)
constants = inspector.find_constants_near(273.15, tolerance=0.01)
for name, value in constants:
    print(f"{name} = {value}")

# Find PHI nodes with zero operand (indicates conditional branch)
zero_phis = inspector.find_phi_nodes_with_value('v3')  # v3 is typically 0.0
for phi in zero_phis[:5]:
    print(f"PHI {phi.result} in {phi.block}")
    for pred, val in phi.operands:
        print(f"  {pred} -> {val}")
```

---

## Jacobian Comparison (`jacobian.py`)

Format-aware comparison between OSDI (sparse, column-major) and JAX (dense, row-major).

```python
from jax_spice.debug import compare_jacobians, print_jacobian_structure

# Compare Jacobians
result = compare_jacobians(
    osdi_jac, jax_jac, n_nodes, jacobian_keys,
    rtol=1e-4, atol=1e-10
)
print(result.report)
print(f"Passed: {result.passed}")
print(f"Max abs diff: {result.max_abs_diff}")
print(f"Mismatched positions: {result.mismatched_positions}")

# Print structure
print_jacobian_structure(jacobian_keys, n_nodes)
```

---

## CLI Tools

### MIR CFG Analyzer

Analyze control flow graph from command line:

```bash
# Find PHI nodes
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --find-phis

# Find branch points
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --branches

# Trace paths to a block
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --target block123

# Analyze specific block with PHIs
uv run scripts/analyze_mir_cfg.py vendor/OpenVAF/integration_tests/PSP102/psp102.va \
    --func eval --analyze-block block4654
```

---

## Debugging Workflow

### 1. Initial Comparison

```python
from jax_spice.debug import quick_compare

result = quick_compare(va_path, osdi_path, params, voltages)
print(result)

if not result.passed:
    print("Issues found:")
    for issue in result.issues:
        print(f"  - {issue}")
```

### 2. Cache Analysis

If residuals differ, check the cache first:

```python
from jax_spice.debug import ModelComparator

comparator = ModelComparator(va_path, osdi_path, params)
cache = comparator.analyze_cache()

# Check for problems
if cache.has_inf > 0:
    print(f"WARNING: {cache.has_inf} inf values in cache")
if cache.has_nan > 0:
    print(f"WARNING: {cache.has_nan} nan values in cache")
```

### 3. MIR Inspection

If cache looks OK, inspect MIR structure:

```python
from jax_spice.debug import MIRInspector

inspector = MIRInspector(va_path)
inspector.print_mir_stats()
inspector.print_phi_summary('eval')

# For NMOS/PMOS models, check TYPE handling
inspector.print_type_param_info()
```

### 4. PHI Node Analysis

If PHI nodes are suspected:

```python
# Find PHIs with zero operand (often indicates branch issue)
zero_phis = inspector.find_phi_nodes_with_value('v3')
print(f"Found {len(zero_phis)} PHIs with zero operand")

# Use CLI for detailed analysis
# uv run scripts/analyze_mir_cfg.py model.va --func eval --analyze-block blockXXX
```

---

## Common Issues

### 1. JAX returns near-zero current

**Symptom**: OSDI returns expected current, JAX returns ~1e-15

**Likely cause**: PHI node resolution in NMOS/PMOS branching

**Debug steps**:
1. Check TYPE parameter is passed correctly
2. Analyze PHI nodes for zero operands
3. Trace control flow with `--analyze-block`

### 2. Jacobian sparsity mismatch

**Symptom**: OSDI has N non-zeros, JAX has M << N

**Likely cause**: Branch not taken, computations skipped

**Debug steps**:
1. Use `print_jacobian_structure()` to see expected pattern
2. Check if missing entries follow a pattern (e.g., all in one row/column)

### 3. Temperature-related errors

**Symptom**: Current off by ~1% at room temperature

**Likely cause**: TNOM vs $temperature handling

**Debug steps**:
1. Check cache for VT values: `cache.temperature_related`
2. Verify expected VT at 300K: 0.02585 V
3. Check for sentinel values (1e21) in init params
