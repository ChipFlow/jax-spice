# sim_back Context Module

The `Context` struct is the central compilation state holder during `sim_back` processing. It manages the MIR function, control flow graph, dominator tree, and tracks which instructions depend on operating point values.

**Location:** `openvaf/sim_back/src/context.rs`

## Data Structure

```rust
pub(crate) struct Context<'a> {
    pub func: Function,              // The MIR function being compiled
    pub cfg: ControlFlowGraph,       // Control flow graph
    pub dom_tree: DominatorTree,     // Dominator tree (for optimization)
    pub intern: HirInterner,         // Symbol table (params, outputs, callbacks)
    pub db: &'a CompilationDB,       // HIR database reference
    pub module: &'a ModuleInfo,      // Module metadata
    pub output_values: BitSet<Value>,       // Values that are outputs
    pub op_dependent_insts: BitSet<Inst>,   // OP-dependent instructions
    pub op_dependent_vals: Vec<Value>,      // OP-dependent values (roots)
}
```

## Purpose

Context serves three main purposes:

1. **State Management** - Holds all compilation state in one place, passed to submodules
2. **Optimization Pipeline** - Provides multi-stage optimization with GVN, DCE, SCCP
3. **OP-Dependence Tracking** - Determines which code depends on operating point

---

## Construction

### `Context::new()`

Creates a new context from a compiled module:

```rust
pub fn new(db: &'a CompilationDB, literals: &mut Rodeo, module: &'a ModuleInfo) -> Self
```

**Steps:**

1. **Build MIR** using `MirBuilder`:
   ```rust
   let (func, intern) = MirBuilder::new(db, module.module, &output_filter, &mut op_vars)
       .with_equations()      // Enable implicit equation handling
       .with_tagged_writes()  // Track variable writes
       .build(literals);
   ```

2. **Filter outputs** - Only keep relevant `PlaceKind` values:
   - `Contribute { .. }` - Branch contributions
   - `ImplicitResidual { .. }` - Implicit equation residuals
   - `CollapseImplicitEquation(_)` - Collapse hints
   - `IsVoltageSrc(_)` - Voltage source flags
   - `BoundStep` - Bound step values
   - `Var(var)` - Only if in `module.op_vars`

3. **Insert variable initialization**:
   ```rust
   intern.insert_var_init(db, &mut func, literals);
   ```

---

## Optimization Pipeline

### `Context::optimize()`

```rust
pub fn optimize(&mut self, stage: OptimiziationStage) -> GVN
```

Performs optimization based on the compilation stage:

```rust
pub enum OptimiziationStage {
    Initial,         // Before topology/DAE construction
    PostDerivative,  // After automatic differentiation
    Final,           // Final cleanup (unused in current code)
}
```

### Optimization Passes

| Pass | Description |
|------|-------------|
| `dead_code_elimination` | Remove unused instructions (Initial only) |
| `sparse_conditional_constant_propagation` | Propagate constants through control flow |
| `inst_combine` | Combine/simplify instruction patterns |
| `simplify_cfg` / `simplify_cfg_no_phi_merge` | Simplify control flow graph |
| `GVN` (Global Value Numbering) | Eliminate redundant computations |
| `aggressive_dead_code_elimination` | Control-flow aware DCE (Final only) |

### Stage Differences

| Stage | DCE | CFG Simplification | Aggressive DCE |
|-------|-----|-------------------|----------------|
| Initial | Yes | No phi merge | No |
| PostDerivative | No | No phi merge | No |
| Final | No | Full | Yes |

**Why no phi merge before Final?** Phi nodes need to be preserved during topology and DAE construction to maintain correct dataflow analysis.

---

## Operating Point Dependence

A value is **OP-dependent** if it depends on simulation unknowns (node voltages, branch currents, time, etc.). This is critical for:

1. **Initialization extraction** - OP-independent code can be cached
2. **Linearization decisions** - Determines if `ddt()` needs implicit equation
3. **Jacobian construction** - Only OP-dependent values need derivatives

### `init_op_dependent_insts()`

Initial OP-dependence analysis for topology construction:

```rust
pub fn init_op_dependent_insts(&mut self, dom_frontiers: &mut SparseBitMatrix<Block, Block>)
```

**Algorithm:**

1. Mark all **noise callback** uses as OP-dependent
2. Mark all **OP-dependent parameters** as roots:
   - `ParamKind::Voltage { .. }` - Node voltages
   - `ParamKind::Current(_)` - Branch currents
   - `ParamKind::ImplicitUnknown(_)` - Implicit equation unknowns
   - System functions like `$abstime`, `$temperature`

3. **Propagate taint** using `propagate_direct_taint()`:
   - Uses dominance frontiers for control dependence
   - Marks all instructions using OP-dependent values

### `refresh_op_dependent_insts()`

Re-computes OP-dependence after MIR modifications:

```rust
pub fn refresh_op_dependent_insts(&mut self)
```

Similar to `init_op_dependent_insts()` but:
- Uses full taint propagation (`propagate_taint()`)
- Considers all `op_dependent()` callbacks, not just noise
- Called before initialization extraction

---

## Helper Methods

### `compute_cfg()`

Recomputes the control flow graph:

```rust
pub fn compute_cfg(&mut self) {
    self.cfg.compute(&self.func);
}
```

### `compute_domtree()`

Computes dominator tree with options:

```rust
pub fn compute_domtree(&mut self, dom: bool, pdom: bool, postorder: bool)
```

| Parameter | Computes |
|-----------|----------|
| `dom` | Dominators (forward) |
| `pdom` | Post-dominators (backward) |
| `postorder` | Postorder traversal numbering |

### `compute_outputs()`

Updates the set of output values:

```rust
pub fn compute_outputs(&mut self, contributes: bool)
```

- If `contributes == true`: All interner outputs
- If `contributes == false`: Only operating variables, collapse hints, bound step

---

## Usage Pattern

```rust
// 1. Create context
let mut cx = Context::new(db, &mut literals, &module);

// 2. Initial setup
cx.compute_outputs(true);
cx.compute_cfg();
cx.optimize(OptimiziationStage::Initial);

// 3. Build topology (modifies func)
let topology = Topology::new(&mut cx);

// 4. Build DAE system (adds derivatives)
let mut dae_system = DaeSystem::new(&mut cx, topology);

// 5. Post-derivative optimization
cx.compute_cfg();
let gvn = cx.optimize(OptimiziationStage::PostDerivative);
dae_system.sparsify(&mut cx);

// 6. Extract initialization
cx.refresh_op_dependent_insts();
let init = Initialization::new(&mut cx, gvn);

// 7. Final result
CompiledModule {
    eval: cx.func,
    intern: cx.intern,
    // ...
}
```

---

## Relationship to Other Modules

```
Context
   │
   ├──► Topology::new(&mut cx)
   │       Uses: func, cfg, dom_tree, intern, db, op_dependent_insts
   │       Modifies: func, intern
   │
   ├──► DaeSystem::new(&mut cx, topology)
   │       Uses: func, cfg, dom_tree, intern, db, op_dependent_insts
   │       Modifies: func, cfg, intern, output_values
   │
   └──► Initialization::new(&mut cx, gvn)
           Uses: func, cfg, dom_tree, intern, db, op_dependent_insts, output_values
           Produces: Initialization { func, intern, cached_vals, cache_slots }
```

---

## Key Insights

1. **Context is mutable throughout** - Each compilation stage modifies the MIR function and related structures

2. **OP-dependence is computed twice**:
   - `init_op_dependent_insts()` for topology (uses dominance frontiers)
   - `refresh_op_dependent_insts()` for initialization (full propagation)

3. **Output values change** - `compute_outputs()` is called multiple times with different filters

4. **GVN provides equivalence classes** - Used by `Initialization` to deduplicate cached values
