# sim_back Topology Module

This document describes the Topology module in `openvaf/sim_back/src/topology.rs` and its submodules.

## Overview

The Topology module transforms raw MIR output from HIR lowering into a structured representation of circuit contributions. This is an intermediate step before building the DAE (Differential-Algebraic Equation) system.

**Location:** `openvaf/sim_back/src/topology.rs`

**Submodules:**
- `builder.rs` - Dimension creation for linearized operators
- `lineralize.rs` - Analog operator evaluation strategy
- `small_signal_network.rs` - Small-signal network detection and pruning

### Purpose

From the module documentation:

> This module is responsible for building a set of coherent branch definitions from the raw lowering results. These can be used to build the model topology (residual, matrix, noise sources) without significant additional analysis.

The module performs three key transformations:

1. **Linearization** - Turn `ddt()` and noise functions into direct contributions when possible, or create implicit equations when not
2. **Small-signal detection** - Identify nodes with zero large-signal voltage
3. **Pruning** - Separate small-signal contributions to avoid unnecessary derivative generation

---

## Data Structures

### `Topology`

The main output structure representing circuit topology.

```rust
pub struct Topology {
    pub branches: TiMap<BranchId, BranchWrite, BranchInfo>,
    pub implicit_equations: TiVec<ImplicitEquation, Contribution>,
    pub small_signal_vals: IndexSet<Value>,
    contributes: AHashMap<Value, ContributeKind>,  // internal bookkeeping
}
```

| Field | Description |
|-------|-------------|
| `branches` | Map from branch write (e.g., `I(a,b)`) to branch information |
| `implicit_equations` | Internal equations created for non-linear `ddt`/noise |
| `small_signal_vals` | Values known to be zero in large-signal analysis |

### `BranchInfo`

Information about a single branch in the circuit.

```rust
pub struct BranchInfo {
    pub is_voltage_src: Value,      // Runtime flag (can be OP-dependent)
    pub voltage_src: Contribution,  // V(a,b) <+ ... contributions
    pub current_src: Contribution,  // I(a,b) <+ ... contributions
}
```

A branch can act as either a voltage source or current source. The `is_voltage_src` field can be:
- `TRUE` - Always a voltage source
- `FALSE` - Always a current source
- A runtime value - Switch branch (determined by operating point)

### `Contribution`

Represents what is contributed to a branch or implicit equation.

```rust
pub struct Contribution {
    pub unknown: Option<Value>,         // The probe value (voltage or current)
    pub resist: Value,                  // Resistive (DC) contribution
    pub react: Value,                   // Reactive (ddt) contribution
    pub resist_small_signal: Value,     // Small-signal resistive part
    pub react_small_signal: Value,      // Small-signal reactive part
    pub noise: Vec<Noise>,              // Noise sources on this branch
}
```

The separation of `resist`/`react` and their `_small_signal` variants enables optimized derivative generation - small-signal values don't need derivatives with respect to large-signal quantities.

### `Noise`

A noise source attached to a contribution.

```rust
pub struct Noise {
    pub name: Spur,              // Noise source name (interned string)
    pub kind: NoiseSourceKind,   // WhiteNoise, FlickerNoise, or NoiseTable
    pub factor: Value,           // Multiplicative factor
}
```

### `ContributeKind` (internal)

Tracks which contribution a MIR value belongs to.

```rust
enum ContributeKind {
    Branch { id: BranchId, is_voltage_src: bool, is_reactive: bool },
    ImplicitEquation { equation: ImplicitEquation, is_reactive: bool },
}
```

---

## Construction Process

`Topology::new(ctx: &mut Context)` builds the topology in several phases.

### Phase 1: Extract Branches and Implicit Equations

**Lines 161-280 in topology.rs**

1. **Collect implicit equations** from `HirInterner`:
   - Filter out dead values and collapsed equations
   - Create `Contribution` entries with the unknown value

2. **Process `PlaceKind` outputs** from the interner:

   | PlaceKind | Action |
   |-----------|--------|
   | `ImplicitResidual { reactive: false }` | Set `implicit_equations[eq].resist` |
   | `ImplicitResidual { reactive: true }` | Set `implicit_equations[eq].react` |
   | `IsVoltageSrc(branch)` | Create `BranchInfo` with probes and contributions |

3. **For each branch**, extract four contribution types:
   - Resistive voltage source: `PlaceKind::Contribute { reactive: false, voltage_src: true }`
   - Reactive voltage source: `PlaceKind::Contribute { reactive: true, voltage_src: true }`
   - Resistive current source: `PlaceKind::Contribute { reactive: false, voltage_src: false }`
   - Reactive current source: `PlaceKind::Contribute { reactive: true, voltage_src: false }`

### Phase 2: Analyze Analog Operators

**Lines 281-311 in topology.rs**

```rust
let operators = builder.analog_operator_evaluations(&postdom_frontiers, &mut ctx.intern);
builder.builid_analog_operators(operators, &mut ctx.intern);
```

This phase determines how to handle `ddt()` and noise function calls. See [Linearization](#linearization-lineralizerss) below.

### Phase 3: Prune Small-Signal Network

**Line 312 in topology.rs**

```rust
builder.prune_small_signal();
```

Identifies nodes with zero large-signal voltage and separates their contributions. See [Small-Signal Analysis](#small-signal-analysis-small_signal_networkrs) below.

---

## Linearization (`lineralize.rs`)

This submodule determines whether analog operators (`ddt`, `white_noise`, `flicker_noise`) can be "linearized" (turned into direct contributions) or need their own implicit equations.

### Evaluation Strategies

```rust
pub enum Evaluation {
    /// Needs its own implicit equation (creates internal unknown)
    Equation,
    /// Can be linearized - no additional unknown needed
    Linear { contributes: Box<[(Value, Value)]> },
    /// Unused, can be removed
    Dead,
}
```

### Decision Algorithm

`determine_evaluation()` traces uses of the analog operator result through the DFG:

| MIR Pattern | Result |
|-------------|--------|
| Only `fadd`, `fsub`, `fneg` | Potentially linear |
| `fmul` with **both** args OP-dependent | **Equation** (non-linear) |
| `fdiv` with OP-dependent denominator | **Equation** |
| OP-dependent phi (conditional) | **Equation** (unless add-chain optimization) |
| Multiple output uses | **Equation** (correlated noise) |
| No uses reaching contributions | **Dead** |

### Add-Chain Optimization

A special optimization handles conditional `ddt` usage that doesn't actually change reactive state:

```verilog
// This pattern does NOT require an implicit equation:
I(x) <+ ddt(foo);
if (op_dependent)
   I(x) <+ bar;
```

This creates an OP-dependent phi `[ddt(foo), ddt(foo) + bar]`, but both branches share the same `ddt` dependency. The `phi_add_chain_start()` function detects when all phi edges lead to the same "add chain start" value.

**Coverage marker:** `cov_mark::hit!(conditional_phi)`

### Creating Dimensions

When an operator can be linearized, `create_dimension()` in `builder.rs` extracts it as a separate "dimension":

1. The original operator result is replaced with `F_ZERO`
2. A parallel computation tracks the linearized coefficient
3. Only linear operations propagate: `fadd`, `fsub`, `fmul`, `fdiv`, `fneg`, `phi`, `OptBarrier`

Example transformation for `I(a,c) <+ V(a) + foo * ddt(V(a))`:

| Before | After |
|--------|-------|
| `resist = V(a) + foo * ddt(V(a))` | `resist = V(a)` |
| `react = 0` | `react = foo * V(a)` |

---

## Small-Signal Analysis (`small_signal_network.rs`)

This submodule identifies nodes that have **zero large-signal voltage** and can be treated specially for AC/noise analysis.

### FlatSet Lattice

Similar to constant propagation, values are analyzed using a three-valued lattice:

```rust
enum FlatSet {
    Top,     // Known to be non-zero
    Bottom,  // Value not yet determined
    Zero,    // Known to be zero
}
```

### Analysis Algorithm

`solve()` iteratively analyzes candidates until reaching a fixed point:

```
1. For each candidate node:
   a. Speculatively add to small_signal_vals
   b. Analyze all resist/react contributions
   c. If proven Zero → keep in set
   d. If proven Top (non-zero) → remove from set
   e. If Bottom → keep candidate for next iteration
2. Repeat until no changes
```

### Value Analysis Rules

`analyze_value()` determines if a value is statically zero:

| Operation | Result |
|-----------|--------|
| `fadd(a, b)` | `min(a, b)` - zero only if **both** zero |
| `fsub(a, b)` | `min(a, b)` - zero only if **both** zero |
| `fmul(a, b)` | `max(a, b)` - zero if **either** zero |
| `fdiv(a, b)` | analyze(a) if b not OP-dependent |
| `phi` | `min` over all edges |
| `fneg(a)`, `optbarrier(a)` | analyze(a) |
| Parameter in `small_signal_vals` | `Zero` |
| Non-zero constant | `Top` |
| Other parameter | `Bottom` |

### Pruning

`prune_small_signal()` separates small-signal contributions:

1. For each value proven to be small-signal:
2. Find all contributions that use it linearly
3. Create a dimension separating small-signal from large-signal
4. Move the small-signal part to `resist_small_signal` / `react_small_signal`

**Coverage marker:** `cov_mark::hit!(prune_small_signal)`

This optimization prevents generating derivatives of small-signal values with respect to large-signal quantities.

---

## Example Walkthrough

Given this Verilog-A:

```verilog
module example(inout a, inout c);
    electrical a, c;
    parameter real foo = 1.0, bar = 2.0;
    analog begin
        I(a, c) <+ V(a) + foo*ddt(V(a)) + white_noise(bar, "thermal");
    end
endmodule
```

### Step 1: Initial Branch Extraction

- Branch `(a, c)` created with `is_voltage_src = FALSE`
- `current_src.resist = V(a)` (from `I(a,c) <+ V(a)`)

### Step 2: ddt Analysis

- `ddt(V(a))` traced through DFG
- Only used in linear chain (`fmul` with non-OP-dependent `foo`, then `fadd`)
- **Result:** `Evaluation::Linear`
- `create_dimension()` extracts coefficient `foo * V(a)`
- `current_src.react = foo * V(a)`

### Step 3: Noise Analysis

- `white_noise(bar)` traced through DFG
- Only used in linear chain (directly added to contribution)
- **Result:** `Evaluation::Linear`
- Noise source added: `Noise { name: "thermal", kind: WhiteNoise { pwr: bar }, factor: 1.0 }`

### Step 4: Small-Signal Analysis

- Both nodes are ports, so they're not candidates
- `small_signal_vals` remains empty

### Final Topology

```
Topology {
    branches: {
        branch0: BranchInfo {
            is_voltage_src: FALSE,
            current_src: Contribution {
                unknown: Some(I(a,c)),
                resist: V(a,c),
                react: foo * V(a),
                resist_small_signal: 0,
                react_small_signal: 0,
                noise: [Noise { name: "thermal", kind: WhiteNoise { pwr: bar }, factor: 1 }]
            },
            voltage_src: Contribution { ... all zeros ... }
        }
    },
    implicit_equations: {},
    small_signal_vals: {}
}
```

---

## Testing

Tests are in `topology/test.rs` using snapshot testing with `expect_test`.

### Test Categories

| Test | Coverage |
|------|----------|
| `linear_analog_operators` | Basic linearization of `ddt` and `white_noise` |
| `conditional_ddt` | Add-chain optimization for OP-dependent phis |
| `collapsible_ddt` | `ddt` in collapsible conditions |
| `conditional_noise` | Noise in conditional code (always linear) |
| `unused_noise` | Dead noise elimination |
| `correlated_noise` | Multiple uses of same noise source |
| `manual_correlated_noise` | Internal noise nodes |
| `psp103` | Complex real-world pattern from PSP model |
| `constant_offset_not_small_signal` | Regression test for small-signal bug |

### Running Tests

```bash
cargo test -p sim_back topology
```

To update snapshots:
```bash
UPDATE_EXPECT=1 cargo test -p sim_back topology
```

---

## Key Insights

1. **Linearization is crucial for performance** - Without it, every `ddt()` would create an implicit equation (internal unknown), significantly increasing system size.

2. **The add-chain optimization** specifically handles common Verilog-A patterns where conditional contributions don't change the reactive state.

3. **Small-signal separation** enables efficient AC/noise analysis by avoiding unnecessary Jacobian entries for values that are always zero in DC.

4. **Switch branches** (runtime-dependent `is_voltage_src`) are fully supported, allowing branches to dynamically switch between voltage and current source behavior.

5. **Correlated noise** requires implicit equations - when the same noise source appears in multiple contributions, it creates correlation that must be modeled with an internal node.
