# sim_back - Simulator Backend

The `sim_back` crate transforms Verilog-A modules (already lowered to MIR) into a form suitable for circuit simulation. It's the bridge between the compiler's intermediate representation and the OSDI (Open Simulator Device Interface) output.

**Location:** `openvaf/sim_back/`

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CompilationDB                                  │
│                        (HIR from hir crate)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            ModuleInfo                                    │
│              (module_info.rs - parameter/port metadata)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              Context                                     │
│                    (context.rs - MIR + CFG + optimization)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌──────────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐
│      Topology        │ │    DaeSystem     │ │     Initialization       │
│   (topology.rs)      │ │    (dae.rs)      │ │      (init.rs)           │
│                      │ │                  │ │                          │
│ Branch extraction,   │ │ Residual/Jacobian│ │ OP-independent code      │
│ linearization,       │ │ matrix entries,  │ │ extraction, cache slots  │
│ small-signal network │ │ noise sources    │ │                          │
└──────────────────────┘ └──────────────────┘ └──────────────────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CompiledModule                                  │
│                        (lib.rs - final output)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            osdi crate                                    │
│                    (LLVM IR generation for OSDI)                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

| File | Description |
|------|-------------|
| `lib.rs` | Main entry point, `CompiledModule` struct, compilation orchestration |
| `context.rs` | `Context` struct holding MIR function, CFG, dominator tree, optimization |
| `module_info.rs` | `ModuleInfo` extraction (parameters, ports, operating variables) |
| `topology.rs` | Branch/contribution extraction, analog operator handling |
| `dae.rs` | DAE system construction (residuals, Jacobian, noise) |
| `init.rs` | Operating-point independent code extraction |
| `node_collapse.rs` | Node collapse pair detection |
| `noise.rs` | Noise source types |
| `util.rs` | Helper functions for MIR manipulation |

---

## Key Data Structures

### `CompiledModule`

The main output of `sim_back`, containing everything needed for OSDI code generation:

```rust
pub struct CompiledModule<'a> {
    pub info: &'a ModuleInfo,        // Module metadata
    pub dae_system: DaeSystem,       // Circuit equations
    pub eval: Function,              // Evaluation MIR
    pub intern: HirInterner,         // Symbol table for eval
    pub init: Initialization,        // Instance setup
    pub model_param_setup: Function, // Model parameter initialization
    pub model_param_intern: HirInterner,
    pub node_collapse: NodeCollapse, // Collapsible node pairs
}
```

### `SimUnknownKind`

Represents the types of unknowns in the DAE system:

```rust
pub enum SimUnknownKind {
    KirchoffLaw(Node),           // Node voltage (KCL equation)
    Current(CurrentKind),         // Branch current
    Implicit(ImplicitEquation),   // Internal equation (from ddt, noise)
}
```

---

## Compilation Pipeline

The `CompiledModule::new()` function orchestrates the full compilation:

### Stage 1: Build Initial MIR

```rust
let mut cx = Context::new(db, literals, module);
```

Creates MIR from HIR using `MirBuilder`, inserting variable initializations.

### Stage 2: Initial Optimization

```rust
cx.compute_outputs(true);
cx.compute_cfg();
cx.optimize(OptimiziationStage::Initial);
```

- Dead code elimination
- Sparse conditional constant propagation
- Instruction combining
- CFG simplification
- Global value numbering (GVN)

### Stage 3: Build Topology

```rust
let topology = Topology::new(&mut cx);
```

Extracts branches and contributions, linearizes analog operators. See [topology.md](sim_back-topology.md) for details.

### Stage 4: Build DAE System

```rust
let mut dae_system = DaeSystem::new(&mut cx, topology);
```

Constructs the circuit equations:
- **Residuals**: `I(x) + ddt(Q(x)) = 0`
- **Jacobian**: Sparse matrix entries `∂I/∂x`, `∂Q/∂x`
- **Noise sources**: White noise, flicker noise, noise tables
- **Limiting**: Newton-Raphson convergence aids

### Stage 5: Post-Derivative Optimization

```rust
cx.optimize(OptimiziationStage::PostDerivative);
dae_system.sparsify(&mut cx);
```

Further optimization after automatic differentiation, plus Jacobian sparsification.

### Stage 6: Extract Initialization

```rust
let mut init = Initialization::new(&mut cx, gvn);
```

Splits the MIR into:
- **Instance setup**: Operating-point independent code (cached)
- **Evaluation**: Operating-point dependent code (per iteration)

### Stage 7: Node Collapse Detection

```rust
let node_collapse = NodeCollapse::new(&init, &dae_system, &cx);
```

Identifies node pairs that can be collapsed (shorted) under certain conditions.

### Stage 8: Parameter Initialization

```rust
init.intern.insert_param_init(db, &mut init.func, literals, false, true, &inst_params);
// ... model_param_setup ...
```

Adds parameter initialization code to instance and model setup functions.

---

## Detailed Module Documentation

- **[Context](context.md)** - Compilation state, optimization pipeline, OP-dependence tracking
- **[Topology](topology.md)** - Branch extraction, linearization, small-signal network
- **[DaeSystem](dae.md)** - DAE equations, residuals, Jacobian matrix, noise sources
- **[Initialization](init.md)** - OP-independent code extraction, cache slot assignment

---

## Testing

Tests are organized by submodule, using snapshot testing with `expect_test`.

### Test Data Locations

| Directory | Contents |
|-----------|----------|
| `openvaf/test_data/contributions/` | Topology snapshots |
| `openvaf/test_data/dae/` | DAE system snapshots |
| `openvaf/test_data/init/` | Initialization snapshots |
| `integration_tests/` | Real device models (BSIM4, PSP103, etc.) |

### Test Structure

Each test module follows a common pattern:

```rust
fn compile(src: &str) -> (Function, Component, String) {
    // 1. Create virtual CompilationDB from inline Verilog-A
    let db = CompilationDB::new_virtual(src).unwrap();

    // 2. Collect module info
    let module = collect_modules(&db, ...).remove(0);

    // 3. Build context and run compilation stages
    let mut context = Context::new(&db, &mut literals, &module);
    // ... optimization and component construction ...

    // 4. Return for snapshot comparison
    (context.func, component, module.module.name(&db))
}

#[test]
fn test_case() {
    let src = indoc! {r#"
        `include "disciplines.vams"
        module test_module(...);
            // Verilog-A code
        endmodule
    "#};

    let (func, component, name) = compile(src);
    expect_file![test_dir.join(format!("{name}_component.snap"))].assert_eq(&format!("{component:#?}"));
}
```

### Running Tests

```bash
# Run all sim_back tests
cargo test -p sim_back

# Run specific test module
cargo test -p sim_back topology
cargo test -p sim_back dae
cargo test -p sim_back init

# Update snapshots when behavior changes
UPDATE_EXPECT=1 cargo test -p sim_back
```

### Coverage Markers

Tests use `cov_mark` to verify specific code paths are exercised:

```rust
#[test]
fn conditional_ddt() {
    cov_mark::check!(conditional_phi);  // Asserts this code path is hit
    // ...
}
```

Common markers:
- `conditional_phi` - OP-dependent phi handling in linearization
- `linear_operator` - Successful linearization of analog operator
- `dead_noise` - Dead noise source elimination
- `prune_small_signal` - Small-signal pruning
- `collapsible_ddt` - Collapsible ddt detection

---

## Integration with OSDI

The `osdi` crate consumes `CompiledModule` to generate LLVM IR:

```rust
// In osdi/src/compilation_unit.rs
use sim_back::{CompiledModule, ModuleInfo};
use sim_back::dae::DaeSystem;
use sim_back::init::Initialization;
```

Key consumers:
- `osdi/src/eval.rs` - Evaluation function codegen
- `osdi/src/load.rs` - Matrix loading codegen
- `osdi/src/setup.rs` - Instance setup codegen
- `osdi/src/metadata.rs` - OSDI metadata generation
- `osdi/src/inst_data.rs` - Instance data structure generation

---

## Debugging

### Print Functions

`sim_back` provides several print functions for debugging:

```rust
// Print module overview
print_module(prefix, db, module, dae_system, init);

// Print interner contents (parameters, outputs, etc.)
print_intern(prefix, db, intern);

// Print MIR function
print_mir(literals, func);
```

### Dump Flags

`CompiledModule::new()` accepts dump flags:

```rust
CompiledModule::new(db, module, literals,
    dump_unopt_mir,  // Print MIR before/after DAE construction
    dump_mir         // Print final optimized MIR
)
```

---

## Key Concepts

### Operating-Point Dependence

A value is **OP-dependent** if it depends on node voltages or branch currents. The compilation tracks this to:
- Determine what can be cached vs. what must be recomputed
- Decide if analog operators need implicit equations

### Implicit Equations

When `ddt()` or noise cannot be linearized (e.g., used in non-linear expressions), an **implicit equation** is created:
- Introduces a new unknown (internal node)
- Adds an equation: `unknown - operator_result = 0`

### Small-Signal Network

Nodes with zero large-signal voltage can be treated specially:
- Their contributions are separated into `_small_signal` fields
- Avoids generating unnecessary derivatives
- Important for AC and noise analysis

### Switch Branches

Branches where `is_voltage_src` is runtime-dependent:
- Can switch between voltage source and current source behavior
- Requires special handling in Jacobian construction
