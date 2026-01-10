# sim_back DaeSystem Module

The `DaeSystem` struct represents a Verilog-A module's circuit topology as a system of Differential-Algebraic Equations (DAE). This is the core output consumed by OSDI code generation.

**Location:** `openvaf/sim_back/src/dae.rs`

**Submodules:**
- `builder.rs` - DAE system construction and Jacobian generation

## Mathematical Background

Circuit simulation solves DAE equations of the form:

```
I(x) + ddt(Q(x)) = 0
```

Where:
- `x` = vector of unknowns (node voltages, branch currents)
- `I(x)` = resistive contributions (DC currents)
- `Q(x)` = reactive contributions (charges/fluxes)
- `ddt` = time derivative operator

**Newton-Raphson iteration:**

```
J(x) · Δx = I(x) + ddt(Q(x))
x' = x - Δx
```

Where `J` is the Jacobian matrix: `J_ij = ∂I_i/∂x_j + ddt(∂Q_i/∂x_j)`

---

## Data Structures

### `DaeSystem`

```rust
pub struct DaeSystem {
    /// Simulation unknowns (x)
    pub unknowns: TiSet<SimUnknown, SimUnknownKind>,

    /// Residual equations (I, Q for each unknown)
    pub residual: TiVec<SimUnknown, Residual>,

    /// Jacobian matrix entries (sparse)
    pub jacobian: TiVec<MatrixEntryId, MatrixEntry>,

    /// Small-signal parameters (always zero in DC)
    pub small_signal_parameters: IndexSet<Value>,

    /// Noise sources
    pub noise_sources: Vec<NoiseSource>,

    /// Model input pairs for AC analysis
    pub model_inputs: Vec<(u32, u32)>,

    /// Jacobian entry counts
    pub num_resistive: u32,
    pub num_reactive: u32,
}
```

### `SimUnknown` and `SimUnknownKind`

```rust
pub struct SimUnknown(u32);  // Index into unknowns

pub enum SimUnknownKind {
    KirchoffLaw(Node),           // Node voltage (KCL equation)
    Current(CurrentKind),         // Branch current
    Implicit(ImplicitEquation),   // Internal equation
}
```

### `Residual`

Each unknown has an associated residual equation:

```rust
pub struct Residual {
    pub resist: Value,              // Resistive part I(x)
    pub react: Value,               // Reactive part Q(x)
    resist_small_signal: Value,     // Small-signal resistive
    react_small_signal: Value,      // Small-signal reactive
    pub resist_lim_rhs: Value,      // Limiting correction (resistive)
    pub react_lim_rhs: Value,       // Limiting correction (reactive)
    pub nature_kind: ResidualNatureKind,
}

pub enum ResidualNatureKind {
    Flow,       // KCL equation (current conservation)
    Potential,  // Voltage source equation
    Switch,     // Dynamic voltage/current source
}
```

### `MatrixEntry`

Sparse Jacobian matrix entry:

```rust
pub struct MatrixEntry {
    pub row: SimUnknown,    // Equation index
    pub col: SimUnknown,    // Unknown index
    pub resist: Value,      // ∂I/∂x
    pub react: Value,       // ∂Q/∂x
}
```

---

## Limiting Correction

When models use **limiting** (e.g., `$limit` function), the simulator evaluates with a limited value `x_lim` instead of `x`. This requires a correction term in the Newton iteration:

```
lim_rhs = J(x_lim) · (x_lim - x)
```

The corrected Newton step becomes:

```
J(x_lim) · Δx = I(x_lim) + ddt(Q) - lim_rhs
```

This ensures convergence when limiting is active. Both `resist_lim_rhs` and `react_lim_rhs` store these corrections.

---

## Construction Process

### `DaeSystem::new()`

```rust
pub fn new(ctx: &mut Context, contributions: topology::Topology) -> DaeSystem
```

**Steps:**

1. **Initialize builder** with small-signal network from topology
2. **Build nodes** - Create unknowns for all ports and internal nodes
3. **Build branches** - Process each branch from topology
4. **Build implicit equations** - Process implicit equations from topology
5. **Compute derivatives** - Run automatic differentiation
6. **Build Jacobian** - Construct sparse matrix entries
7. **Build limiting RHS** - Compute limiting correction terms
8. **Finalize** - Add optbarriers, count entries

### Branch Processing

The builder handles three branch types:

#### Current Branch (`is_voltage_src == FALSE`)

```
I(a,b) <+ contribution
```

- If branch current is probed → add source equation with `nature_kind = Flow`
- Otherwise → add to Kirchhoff's Current Law (KCL) equations for nodes

#### Voltage Branch (`is_voltage_src == TRUE`)

```
V(a,b) <+ contribution
```

- Always adds source equation with `nature_kind = Potential`
- Branch current becomes an unknown

#### Switch Branch (runtime `is_voltage_src`)

```
// Runtime selection between voltage/current source
```

- Creates control flow: `br is_voltage_src, voltage_bb, current_bb`
- Uses phi nodes to merge voltage/current contributions
- `nature_kind = Switch`

### Automatic Differentiation

After building residuals, the builder:

1. Collects all `(residual, unknown)` pairs needing derivatives
2. Calls `auto_diff()` from `mir_autodiff` crate
3. Returns a map: `(Value, Unknown) → Value` (derivative values)

### Jacobian Construction

For each residual row:

1. Build dense row (all columns)
2. For each simulation unknown read (voltage/current/implicit):
   - Look up derivative `∂residual/∂unknown`
   - Handle limiting states (multiple derivative values)
   - Add to appropriate matrix entry
3. Sparsify: only keep non-zero entries

---

## Sparsification

### `DaeSystem::sparsify()`

```rust
pub fn sparsify(&mut self, ctx: &mut Context)
```

Post-optimization cleanup:

1. **Simplify optbarriers** - If underlying value is constant, unwrap it
2. **Remove dead noise sources** - Zero factor or zero power
3. **Remove zero Jacobian entries** - Both resist and react are zero

---

## Noise Sources

```rust
pub struct NoiseSource {
    pub name: Spur,              // Source name
    pub kind: NoiseSourceKind,   // White/flicker/table
    pub hi: SimUnknown,          // Positive terminal
    pub lo: Option<SimUnknown>,  // Negative terminal (optional)
    pub factor: Value,           // Multiplicative factor
}

pub enum NoiseSourceKind {
    WhiteNoise { pwr: Value },
    FlickerNoise { pwr: Value, exp: Value },
    NoiseTable { log: bool, vals: Vec<(f64, f64)> },
}
```

Noise sources are attached to branches during construction, with `mfactor` scaling applied:
- Current sources: `factor *= sqrt(mfactor)`
- Voltage sources: `factor /= sqrt(mfactor)`

---

## M-Factor Handling

The `$mfactor` system function represents parallel device multiplicity. The builder applies scaling:

| Contribution Type | Scaling |
|-------------------|---------|
| Current (KCL) | Multiply residual by `mfactor` |
| Noise (current) | Multiply factor by `sqrt(mfactor)` |
| Noise (voltage) | Divide factor by `sqrt(mfactor)` |

This is handled in `ensure_optbarriers()` for residuals and in branch processing for noise.

---

## Testing

Tests are in `dae/tests.rs` using snapshot testing:

```rust
fn compile(src: &str) -> (Function, DaeSystem, String) {
    // Compile Verilog-A to DaeSystem
}

#[test]
fn resistor_va() {
    let src = /* Verilog-A resistor */;
    let (func, system, name) = compile(src);
    expect_file![test_dir.join(format!("{name}_system.snap"))].assert_eq(&system);
    expect_file![test_dir.join(format!("{name}_mir.snap"))].assert_eq(&func);
}
```

### Test Cases

| Test | Description |
|------|-------------|
| `resistor_va` | Basic resistor (current source) |
| `diode_va` | Diode with limiting |
| `voltage_src` | Voltage source branch |
| `const_switch_branch` | Static switch branch |
| `dyn_switch_branch` | Dynamic switch branch |
| `lim_rhs` | Limiting correction |
| `lim_rhs_react` | Reactive limiting |

---

## Example: Simple Resistor

**Verilog-A:**
```verilog
module resistor(inout p, inout n);
    electrical p, n;
    parameter real r = 1k;
    analog I(p, n) <+ V(p, n) / r;
endmodule
```

**Resulting DaeSystem:**

```
unknowns: {
    sim_node0: KirchoffLaw(p),
    sim_node1: KirchoffLaw(n)
}

residual: {
    sim_node0: Residual {
        resist: V(p,n) / r,  // I flowing out of p
        react: 0,
        ...
    },
    sim_node1: Residual {
        resist: -V(p,n) / r, // I flowing into n (negated)
        react: 0,
        ...
    }
}

jacobian: [
    { row: sim_node0, col: sim_node0, resist: 1/r, react: 0 },
    { row: sim_node0, col: sim_node1, resist: -1/r, react: 0 },
    { row: sim_node1, col: sim_node0, resist: -1/r, react: 0 },
    { row: sim_node1, col: sim_node1, resist: 1/r, react: 0 },
]
```

---

## Integration with OSDI

The `osdi` crate uses `DaeSystem` for:

| OSDI Function | DaeSystem Field |
|---------------|-----------------|
| `osdi_eval` | `residual.resist`, `residual.react` |
| `osdi_load` | `jacobian` entries |
| `osdi_noise` | `noise_sources` |
| Matrix setup | `unknowns`, `jacobian` structure |

---

## Key Insights

1. **Sparse representation** - Jacobian only stores non-zero entries
2. **Automatic differentiation** - Derivatives computed from MIR, not symbolically
3. **Limiting is first-class** - Correction terms computed automatically
4. **M-factor scaling** - Parallel devices handled correctly
5. **Small-signal separation** - Enables efficient AC/noise analysis
