# OpenVAF Eval Function Generation

This document analyzes how OpenVAF generates the OSDI `eval()` function from MIR, focusing on control flow handling for JAX translation.

## Function Signature

```c
int eval(void* handle, void* instance, void* model, osdi_sim_info* sim_info);
```

| Parameter | Description |
|-----------|-------------|
| `handle` | Simulator callback handle |
| `instance` | Instance data (params, cache, node mapping) |
| `model` | Model data |
| `sim_info` | Simulation info (voltages, flags, result arrays) |

**Returns**: Integer flags indicating limiting occurred, etc.

## Generation Flow

### 1. Parameter Setup (eval.rs:100-303)

The eval function first sets up all MIR parameters by mapping `ParamKind` to runtime values:

| ParamKind | Source | Varying? |
|-----------|--------|----------|
| `Param(param)` | Instance or model data | No |
| `Voltage { hi, lo }` | `V(hi) - V(lo)` from prev_solve | **Yes** |
| `Current(kind)` | prev_solve or 0 for ports | **Yes** |
| `Abstime` | sim_info field | **Yes** (time) |
| `Temperature` | Instance data | No |
| `ParamGiven { param }` | Instance or model flags | No |
| `PortConnected { port }` | Compare with connected_ports | No |
| `ParamSysFun(param)` | Instance (mfactor, etc.) | No |
| `EnableIntegration` | Check flags | No |
| `PrevState(state)` | State array | **Yes** |
| `NewState(state)` | State array | **Yes** |
| `EnableLim` | Check flags | No |
| `HiddenState` | Unreachable (moved to cache) | N/A |

**Key insight**: Hidden state params are computed at `setup_instance` and stored in cache. In eval, they're loaded as cache values (lines 296-302).

### 2. Cache Value Loading (eval.rs:296-302)

```rust
let cache_vals = (0..module.init.cache_slots.len()).map(|i| {
    let slot = i.into();
    let val = inst_data.load_cache_slot(module, builder.llbuilder, slot, instance);
    BuilderVal::Eager(val)
});
params.extend(cache_vals);
```

Cache values are appended to the params array, making them available as fixed inputs to the eval MIR.

### 3. MIR Execution (eval.rs:360-363)

```rust
builder.build_consts();  // Emit constant values
builder.build_func();    // Emit all MIR instructions
```

This is where the MIR CFG is translated to LLVM IR.

### 4. Control Flow Translation (builder.rs)

#### Branch Translation (builder.rs:617-624)

```rust
mir::InstructionData::Branch { cond, then_dst, else_dst, .. } => {
    LLVMBuildCondBr(
        self.llbuilder,
        self.values[cond].get(self),
        self.blocks[then_dst].unwrap(),
        self.blocks[else_dst].unwrap(),
    );
}
```

MIR branches map directly to LLVM conditional branches.

#### PHI Node Translation (builder.rs:626-644)

PHI nodes use a **two-phase approach**:

**Phase 1**: Create placeholder PHI
```rust
mir::InstructionData::PhiNode(ref phi) => {
    let ty = self.func.dfg.phi_edges(phi)
        .find_map(|(_, val)| self.values[val].get_ty(self))
        .unwrap();
    let llval = LLVMBuildPhi(self.llbuilder, ty, UNNAMED);
    self.unfinished_phis.push((phi.clone(), llval_ref));
    self.values[res] = BuilderVal::Eager(llval_ref);
}
```

**Phase 2**: Populate edges after all blocks (builder.rs:512-536)
```rust
for (phi, llval) in self.unfinished_phis.iter() {
    let (blocks, vals): (Vec<_>, Vec<_>) = self.func.dfg
        .phi_edges(phi)
        .map(|(bb, val)| {
            self.select_bb_before_terminator(bb);
            (self.blocks[bb].unwrap(), self.values[val].get(self))
        })
        .unzip();

    LLVMAddIncoming(llval, incoming_vals, incoming_blocks, vals.len());
}
```

**Key insight**: Values are retrieved by positioning before the terminator of each predecessor, ensuring correct value at that point.

### 5. Output Storage (eval.rs:366-434)

Results are stored conditionally based on flags:

```rust
for reactive in [false, true] {
    let (jacobian_flag, residual_flag, lim_rhs_flag) = if reactive {
        (CALC_REACT_JACOBIAN, CALC_REACT_RESIDUAL, CALC_REACT_LIM_RHS)
    } else {
        (CALC_RESIST_JACOBIAN, CALC_RESIST_RESIDUAL, CALC_RESIST_LIM_RHS)
    };

    // Store Jacobian if flag set
    Self::build_store_results(&mut builder, llfunc, &flags, jacobian_flag, |builder| {
        for entry in module.dae_system.jacobian.keys() {
            inst_data.store_jacobian(entry, instance, builder, reactive)
        }
    });
    // ... similar for residuals and lim_rhs
}
```

## JAX Translation Strategy

Based on this analysis, the JAX translation should:

### For Simple Models (no branches)

Generate straight-line JAX code:
```python
def eval_fn(params, cache):
    # Direct translation of MIR ops to JAX
    v10 = params[0] + params[1]  # fadd
    v11 = v10 * cache[0]         # fmul
    # ...
    return residuals, jacobian
```

### For Models with Control Flow

Use `lax.cond` for branches:
```python
def eval_fn(params, cache):
    # Branch: if cond then block_true else block_false
    def true_branch():
        # ... code from true block
        return value_for_phi

    def false_branch():
        # ... code from false block
        return value_for_phi

    v33 = lax.cond(cond, true_branch, false_branch)
    # v33 is the PHI result
```

### Optimization: XLA Constant Folding

When fixed params are captured in closure, XLA will optimize away dead branches:
```python
def make_eval(fixed_params):
    # fixed_params captured as constants
    def eval_fn(varying_params):
        # This branch will be optimized away if condition is constant
        result = lax.cond(
            fixed_params['type'] > 0,  # Constant after JIT
            nmos_branch,
            pmos_branch
        )
    return jax.jit(eval_fn)
```

## CFG to Nested Conditionals

The MIR CFG must be converted to nested conditionals:

1. **Dominator tree analysis** - identify which blocks dominate others
2. **Region identification** - find single-entry-single-exit regions
3. **Nesting structure** - convert regions to nested `lax.cond` calls
4. **PHI elimination** - PHI results become return values of conditional branches

### Example CFG Transformation

```
MIR CFG:
  block0:
    v1 = ...
    br v1, block1, block2
  block1:
    v2 = ...
    jmp block3
  block2:
    v3 = ...
    jmp block3
  block3:
    v4 = phi { v2 from block1, v3 from block2 }

JAX:
  v1 = ...
  def block1():
      v2 = ...
      return v2
  def block2():
      v3 = ...
      return v3
  v4 = lax.cond(v1, block1, block2)
```

## Varying vs Fixed Parameter Analysis

From PSP103 analysis:

| Kind | Count | Varying? |
|------|-------|----------|
| param | 840 | No |
| hidden_state | 1705 | No (in cache) |
| param_given | 34 | No |
| sysfun | 1 | No |
| temperature | 1 | No |
| **voltage** | 19 | **Yes** |
| **current** | 16 | **Yes** |

**Branch analysis result**:
- 174 branches (40.4%) depend only on fixed params → XLA will optimize away
- 257 branches (59.6%) depend on voltages/currents → must remain as `lax.cond`

## Implementation Priority

1. **Simple models** (resistor, capacitor): No control flow, direct translation
2. **Medium models** (diode): Few branches, manual CFG transformation feasible
3. **Complex models** (PSP103): Need automated CFG-to-conditional transformation

## File References

- `openvaf/osdi/src/eval.rs` - OSDI eval function generation
- `openvaf/mir_llvm/src/builder.rs` - MIR to LLVM IR builder
- `openvaf/mir/src/instructions.rs` - MIR instruction definitions
- `openvaf/sim_back/src/init.rs` - Cache value separation logic
