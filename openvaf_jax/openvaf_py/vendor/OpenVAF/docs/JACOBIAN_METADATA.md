# OpenVAF Jacobian Metadata: Derivatives vs Values

## Summary

OpenVAF maintains clear separation between **values** (I, Q) and **derivatives** (dI/dx, dQ/dx) through metadata structures.

## Key Data Structures

### 1. DAE Residual Structure (`sim_back/src/dae.rs:117-132`)

```rust
pub struct Residual {
    /// The resistive part (I) of the DAE cost function
    pub resist: Value,
    /// The reactive part (Q) of the DAE cost function
    pub react: Value,
    resist_small_signal: Value,
    react_small_signal: Value,
    // ... limiting terms
}
```

**These are VALUES, not derivatives:**
- `residual.resist` → MIR Value for **I(x)** (current/flow)
- `residual.react` → MIR Value for **Q(x)** (charge/potential)

### 2. DAE MatrixEntry Structure (`sim_back/src/dae.rs:222-227`)

```rust
pub struct MatrixEntry {
    pub row: SimUnknown,       // Equation index
    pub col: SimUnknown,       // Unknown variable index
    pub resist: Value,         // ← Derivative: dI_row/dx_col
    pub react: Value,          // ← Derivative: dQ_row/dx_col
}
```

**These ARE DERIVATIVES:**
- `matrix_entry.resist` → MIR Value for **dI/dx** (resistive Jacobian entry)
- `matrix_entry.react` → MIR Value for **dQ/dx** (reactive Jacobian entry)

### 3. Derivatives HashMap (`mir_autodiff/src/lib.rs:16-26`)

```rust
pub fn auto_diff(
    func: &mut Function,
    dom_tree: &DominatorTree,
    derivatives: &KnownDerivatives,
    extra_derivatives: &[(Value, Unknown)],
) -> AHashMap<(Value, Unknown), Value>
```

**Output mapping:**
- Key: `(value_to_differentiate, unknown_variable)`
- Value: MIR Value containing the derivative

**Examples:**
- `derivatives[(residual.resist, voltage_node5)]` → dI/dV(node5)
- `derivatives[(residual.react, voltage_node5)]` → dQ/dV(node5)

## How Jacobian is Built

### Step 1: Compute Derivatives (`sim_back/src/dae/builder.rs:93-107`)

```rust
let derivatives = auto_diff(
    &mut *self.cursor.func,
    self.dom_tree,
    &derivative_info,
    &extra_derivatives
);
```

This generates MIR instructions that compute derivatives and returns a mapping.

### Step 2: Build Jacobian Matrix (`sim_back/src/dae/builder.rs:271-351`)

```rust
fn build_jacobian(
    &mut self,
    sim_unknown_reads: &[(ParamKind, Value)],
    derivative_info: &KnownDerivatives,
    derivatives: &AHashMap<(Value, Unknown), Value>,
) {
    // For each row (residual equation)
    for (row, residual) in self.system.residual.iter_enumerated() {
        // For each column (unknown variable)
        for &(kind, val) in sim_unknown_reads {
            let unknown = derivative_info.unknowns.index(&val);

            // Look up derivatives from the HashMap
            let resist_deriv = derivatives.get(&(residual.resist, unknown));
            let react_deriv = derivatives.get(&(residual.react, unknown));

            // Store in sparse matrix
            self.system.jacobian.push(MatrixEntry {
                row,
                col: sim_unknown,
                resist: resist_deriv.copied().unwrap_or(F_ZERO),
                react: react_deriv.copied().unwrap_or(F_ZERO),
            });
        }
    }
}
```

**Key insight:** The function looks up:
- `derivatives[(residual.resist, unknown)]` → **dI/dx**
- `derivatives[(residual.react, unknown)]` → **dQ/dx**

### Step 3: Map to OSDI Output Slots (`osdi/src/inst_data.rs:141-169`)

```rust
impl MatrixEntry {
    pub fn new<'ll>(
        entry: &dae::MatrixEntry,  // ← Contains derivative MIR Values
        module: &OsdiModule<'_>,
        slots: &mut TiMap<EvalOutputSlot, mir::Value, &'ll llvm_sys::LLVMType>,
        ty_real: &'ll llvm_sys::LLVMType,
        num_react: &mut u32,
    ) -> MatrixEntry {
        let mut get_output = |mut val| {
            val = strip_optbarrier(module.eval, val);
            if val == F_ZERO {
                None
            } else {
                Some(EvalOutput::new(module, val, slots, false, ty_real))
            }
        };

        MatrixEntry {
            resist: get_output(entry.resist),  // ← Maps dI/dx MIR value to slot
            react: get_output(entry.react),    // ← Maps dQ/dx MIR value to slot
            react_off: react_off.into(),
        }
    }
}
```

**This creates the mapping:**
- MIR Value (derivative) → EvalOutputSlot → Final output array index

### Step 4: Store to Output Arrays (`osdi/src/inst_data.rs:861-873`)

```rust
pub unsafe fn store_jacobian(
    &self,
    entry: MatrixEntryId,
    inst_ptr: &'ll llvm_sys::LLVMValue,
    builder: &mir_llvm::Builder<'_, '_, 'll>,
    reactive: bool,
) {
    let entry = &self.jacobian[entry];
    let dst = if reactive {
        entry.react  // ← EvalOutput for dQ/dx
    } else {
        entry.resist // ← EvalOutput for dI/dx
    };

    if let Some(EvalOutput::Calculated(slot)) = dst {
        self.store_eval_output_slot(slot, inst_ptr, builder)
    }
}
```

## MIR Example (PSP103)

Looking at the PSP103 output file, we can see derivative calls:

```
inst11 = const fn %ddx_node_node8(1) -> 1  // Derivative operator for node8
inst12 = const fn %ddx_node_node5(1) -> 1  // Derivative operator for node5
...

v215046 = phi [v215043, block2219], [v215045, block2220]  // Some value
v215047 = call inst16(v215046)  // dV/d(node4) - DERIVATIVE
v215049 = call inst11(v215046)  // dV/d(node8) - DERIVATIVE
v215051 = call inst13(v215046)  // dV/d(node6) - DERIVATIVE
v215053 = call inst12(v215046)  // dV/d(node5) - DERIVATIVE
```

**These MIR values (v215047, v215049, v215051, v215053) contain DERIVATIVES.**

## How to Identify Derivatives in MIR

### Method 1: Check if it's a derivative call result

```rust
// If you see:
v123 = call inst11(v100)  // where inst11 = const fn %ddx_node_nodeX

// Then v123 is a DERIVATIVE (dv100/d(nodeX))
```

### Method 2: Check the DAE MatrixEntry metadata

For PSP103 with nodes (d, g, s, b, noi):
- Row 0, Col 0: `MatrixEntry { row: node_d, col: node_d, resist: v1234, react: v5678 }`
  - v1234 contains **dI_d/dV_d** (resistive derivative)
  - v5678 contains **dQ_d/dV_d** (reactive derivative)

### Method 3: Check the output slot mapping

```rust
// In osdi/src/inst_data.rs, the inst_data.jacobian array maps:
// jacobian[entry_id] = MatrixEntry {
//     resist: Some(EvalOutput::Calculated(slot_42)),  // dI/dx is at output slot 42
//     react: Some(EvalOutput::Calculated(slot_97)),   // dQ/dx is at output slot 97
// }
```

## Complete Data Flow

```
1. Verilog-A code:
   I(d,s) <+ gds * V(d,s);

2. Residual (VALUE):
   residual[node_d].resist = gds * (V_d - V_s)  // MIR value v100

3. Auto-diff generates DERIVATIVES:
   derivatives[(v100, voltage_d)] = gds         // MIR value v200 (dI/dV_d)
   derivatives[(v100, voltage_s)] = -gds        // MIR value v201 (dI/dV_s)

4. Jacobian entries (DERIVATIVES):
   MatrixEntry { row: node_d, col: node_d, resist: v200, react: v0 }
   MatrixEntry { row: node_d, col: node_s, resist: v201, react: v0 }

5. OSDI output mapping:
   jacobian[0].resist → output_slot[42] → eval function computes v200
   jacobian[1].resist → output_slot[43] → eval function computes v201
```

## Key Takeaways

1. **Residual.resist/react** contain **VALUES** (I, Q)
2. **MatrixEntry.resist/react** contain **DERIVATIVES** (dI/dx, dQ/dx)
3. **Derivatives are MIR Values** computed by instructions generated by mir_autodiff
4. **Derivative calls** (`call %ddx_node_nodeX(value)`) explicitly compute derivatives
5. **The derivatives HashMap** maps `(value, unknown) → derivative_value`
6. **OSDI metadata** tracks which output slot corresponds to which Jacobian entry

## References

- Residual values: `openvaf/sim_back/src/dae.rs:117-132`
- Jacobian derivatives: `openvaf/sim_back/src/dae.rs:222-227`
- Auto-differentiation: `openvaf/mir_autodiff/src/lib.rs:16-26`
- Jacobian builder: `openvaf/sim_back/src/dae/builder.rs:271-351`
- OSDI mapping: `openvaf/osdi/src/inst_data.rs:141-169`
- OSDI output: `openvaf/osdi/src/eval.rs:377-388`
