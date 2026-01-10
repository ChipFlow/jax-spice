# MIR to OSDI Pipeline: Implementation Clarifications

This document answers critical implementation questions about the OpenVAF MIR to OSDI pipeline.

## 1. Parameter Initialization Flow (setup_model)

**Answer**: Parameters ARE already in the model structure (put there by simulator from netlist).

### Flow in setup_model()

```rust
// Lines 152-162 in osdi/src/setup.rs
for (i, param) in model_data.params.keys().copied().enumerate() {
    // 1. LOAD parameter value FROM model structure (simulator put it there)
    let loc = unsafe { model_data.nth_param_loc(cx, i, &*model) };
    builder.params[dst] = BuilderVal::Load(Box::new(loc));

    // 2. LOAD param_given flag (simulator set this if param was in netlist)
    let is_given = unsafe { model_data.is_nth_param_given(cx, i, &*model, builder.llbuilder) };
    builder.params[dst_given] = BuilderVal::Eager(is_given);
}

// 3. Execute MIR (applies Verilog-A defaults and validation)
builder.build_func();  // This runs the model_param_setup MIR

// 4. STORE computed values back to model structure
for (i, param) in model_data.params.keys().enumerate() {
    let val = builder.values[val].get(&builder);
    model_data.store_nth_param(i as u32, &*model, val, builder.llbuilder);
}
```

### Where VA Defaults Get Applied

Verilog-A default values (from `parameter real tox = 1e-9;`) are applied in the **model_param_setup MIR**.

The MIR lowering (`hir_lower/src/parameters.rs:83-110`) creates this structure:

```rust
if param_given {
    // User provided value - validate it
    check_param_bounds(param_val, bounds, ops, invalid_callback);
    param_val  // Use user value
} else {
    // User didn't provide - use VA default
    let default_val = lower_expression(param.init(db));  // From .va file
    default_val
}
```

### Simulator's Responsibilities

1. **Before setup_model()**: Simulator allocates model structure and writes:
   - Parameter values from netlist
   - `param_given` flags (true if user specified, false otherwise)

2. **After setup_model()**: Simulator reads validated/defaulted parameter values back.

### Example Timeline

```
Netlist: resistor r1 (n1, n2) .model myres r=1000

1. Simulator creates model structure for "myres"
2. Simulator writes: model.params[r_idx] = 1000.0, param_given[r_idx] = true
3. Simulator calls setup_model(&handle, &model, &simparam, &result)
4. setup_model loads r=1000, param_given=true
5. setup_model runs MIR: if param_given { validate(1000) } else { use_default }
6. setup_model stores validated r=1000 back to model structure
7. Simulator uses validated parameters
```

---

## 2. Callback Semantics (ParamInfo::Invalid)

**Answer**: Callbacks push error messages to a result vector for the simulator to display.

### What ParamInfo::Invalid Does

**Source**: `osdi/src/setup.rs:113-123`

```rust
fn invalid_param_err(cx: &CodegenCx<'_, 'll>) -> (&'ll llvm_sys::LLVMType, &'ll llvm_sys::LLVMValue) {
    // Looks up stdlib function
    let val = cx.get_func_by_name("push_invalid_param_err").expect(...);
    let ty = cx.ty_func(&[cx.ty_ptr(), cx.ty_ptr(), cx.ty_ptr(), cx.ty_int()], cx.ty_void());
    (ty, val)
}
```

### Signature

```c
void push_invalid_param_err(
    char** err_ptr,      // Pointer to error string array
    size_t* err_len,     // Current length of error array
    size_t* err_cap,     // Current capacity of error array
    uint32_t param_idx   // Which parameter is invalid
);
```

### Behavior

1. **Appends** error to dynamic array in `osdi_init_info` result structure
2. Does **NOT** print to console directly
3. Does **NOT** abort execution
4. Allows **multiple** errors to accumulate

### MIR Usage

**Source**: `hir_lower/src/parameters.rs:88-96`

```rust
ctx.check_param(
    param_val,
    &bounds,
    &[],
    ConstraintKind::From,
    ops,
    invalid,  // <-- Callback invoked if bounds check fails
    exit,
);
```

This creates MIR like:

```
if param_val < min || param_val > max {
    call invalid_callback(param_idx);
    // Continue execution (don't abort)
}
```

### Other ParamInfo Callbacks

- `ParamInfoKind::MinInclusive`: Sets minimum bound metadata
- `ParamInfoKind::MaxInclusive`: Sets maximum bound metadata
- `ParamInfoKind::MinExclusive`: Sets exclusive lower bound metadata
- `ParamInfoKind::MaxExclusive`: Sets exclusive upper bound metadata

These are **metadata-setting** callbacks, not validation failures.

### For MIR→Python Implementation

**Can you skip these?**
- **Short term**: Yes - treat as no-ops for prototyping
- **Long term**: Implement as Python exceptions or warnings
- **Critical**: Bounds violations should at least warn users

**Recommended Python implementation**:
```python
def check_param_bounds(val, min_val, max_val, param_name):
    if val < min_val or val > max_val:
        raise ValueError(f"Parameter {param_name}={val} outside bounds [{min_val}, {max_val}]")
```

---

## 3. Cache Value Selection Mechanism

**Answer**: Operating-point dependency analysis determines cached values.

### How It Works

**Source**: `sim_back/src/context.rs:126-189` and `sim_back/src/init.rs:101-142`

### Step 1: Mark OP-Dependent Instructions

**Function**: `Context::init_op_dependent_insts()`

```rust
// Mark instructions that depend on operating point
op_dependent_insts.clear();

// 1. Mark voltage/current parameters as OP-dependent
for (param, &val) in self.intern.params.iter() {
    if param.op_dependent() {  // Voltage, Current, ImplicitUnknown
        op_dependent_vals.push(val);
    }
}

// 2. Propagate taint through data flow graph
propagate_taint(
    &self.func,
    &self.dom_tree,
    &self.cfg,
    op_dependent_vals.iter().copied(),
    &mut op_dependent_insts,  // Output: set of tainted instructions
);
```

### Step 2: Split Instructions Between Init and Eval

**Function**: `Initialization::Builder::split_block()`

```rust
for inst in block_insts {
    if self.op_dependent_insts.contains(inst) {
        // Keep in EVAL MIR - depends on operating point
        // Don't copy to init
    } else {
        // Copy to INIT MIR - can be computed once at setup
        self.copy_instruction(inst, bb);
    }
}
```

### Step 3: Determine Which Values to Cache

**Source**: `sim_back/src/init.rs:194-239`

A value gets cached if:

1. **It's computed in init** (OP-independent)
2. **It's used in eval** (after dead code elimination)
3. **OR** it's an output variable (even if not used in eval)

```rust
// Lines 194-229
let is_output = self.func.dfg.insts[inst].opcode() == Opcode::OptBarrier
    && self.output_values.contains(self.func.dfg.first_result(inst));

let cache_inst = !is_output
    && self.func.dfg.inst_results(inst).iter().any(|val| {
        self.func.dfg.tag(*val).is_some()  // Has a name (user variable)
    });

if is_output || cache_inst {
    // Create cache slot for this value
    let param = self.init_cache.insert_full(val, inst).0 + self.intern.params.len();
    self.func.dfg.values.make_param_at(param.into(), val);
}
```

### Step 4: Build Cache Slots with GVN

**Source**: `sim_back/src/init.rs:240-290`

```rust
// Group cached values by GVN equivalence class
let equiv_class = inst.and_then(|inst| gvn.inst_class(inst));

self.init.cached_vals = self
    .init_cache
    .iter()
    .filter_map(|(&val, &old_inst)| {
        if self.func.dfg.value_dead(val) {
            return None;  // Value not used in eval, don't cache
        }

        let slot = ensure_cache_slot(inst, res, ty);
        Some((val, slot))
    })
    .collect();
```

### What Marks a Parameter as OP-Dependent?

**Source**: `hir_lower/src/lib.rs` (ParamKind enum)

```rust
impl ParamKind {
    pub fn op_dependent(&self) -> bool {
        matches!(
            self,
            ParamKind::Voltage { .. }          // V(n1, n2)
                | ParamKind::Current(_)        // I(branch)
                | ParamKind::ImplicitUnknown(_) // Hidden state
                | ParamKind::PrevState(_)      // Integration state
                | ParamKind::NewState(_)
        )
    }
}
```

### Summary: Cache Selection Algorithm

```
1. Start with OP-dependent seeds: voltages, currents, implicit unknowns
2. Propagate taint forward through data flow graph
3. All tainted instructions stay in EVAL
4. All non-tainted instructions go to INIT
5. Values computed in INIT but used in EVAL → cached
6. GVN groups equivalent cached values into same slot
```

### For MIR→Python Implementation

**What you need to implement**:

1. **Taint propagation**:
   ```python
   def mark_op_dependent(func, params):
       tainted = set()
       worklist = [v for (k, v) in params if k.is_voltage() or k.is_current()]
       while worklist:
           val = worklist.pop()
           for use in val.uses():
               if use not in tainted:
                   tainted.add(use)
                   worklist.append(use.result)
       return tainted
   ```

2. **Split functions**:
   ```python
   def split_init_eval(func, tainted_insts):
       init_func = Function()
       eval_func = Function()
       for inst in func.instructions:
           if inst in tainted_insts:
               eval_func.add(inst)
           else:
               init_func.add(inst)
       return init_func, eval_func
   ```

3. **Identify cached values**:
   ```python
   cached_vals = {}
   for val in init_func.results:
       if val in eval_func.uses:
           cached_vals[val] = allocate_cache_slot()
   ```

---

## 4. Parameter Priority in eval()

**Answer**: Instance params override model params. This is CORRECT - instances specialize models.

### Priority Order (Lowest to Highest)

1. **Model-level defaults** (from `(* type="model" *)` params in .va file)
2. **Instance-level overrides** (from `(* type="instance" *)` params)

### Code Evidence

**Source**: `osdi/src/setup.rs:366-381`

```rust
for (i, param) in inst_data.params.keys().enumerate() {
    // Check if explicitly set on instance
    let is_inst_given = inst_data.is_nth_param_given(cx, i, instance, llbuilder);

    // If not, check model-level default
    let is_given = builder.select(
        is_inst_given,
        true_,
        model_data.is_nth_inst_param_given(cx, i, model, llbuilder)
    );

    // Load from instance or fall back to model
    let inst_val = inst_data.read_nth_param(i, instance, llbuilder);
    let model_val = model_data.read_nth_inst_param(inst_data, i, model, llbuilder);
    let val = builder.select(is_inst_given, inst_val, model_val);  // <-- Instance wins
}
```

### In eval()

**Source**: Documentation lines 638-641 (corrected interpretation)

```rust
ParamKind::Param(param) => {
    // Try instance first, then model
    inst_data.param_loc(cx, OsdiInstanceParam::User(param), instance)
        .unwrap_or_else(|| model_data.param_loc(cx, param, model).unwrap())
}
```

This means:
- If parameter is instance-level: load from instance
- If parameter is model-level only: load from model

### Why This Makes Sense

Consider MOSFET with model params:

```spice
.model nmos nmos level=54 tox=1e-9 vth0=0.5  ← Model defaults

m1 d g s b nmos w=1u l=0.18u                  ← Instance-specific geometry
m2 d g s b nmos w=10u l=0.18u                 ← Different instance, same model
```

- `tox`, `vth0`: Model params (shared by m1, m2)
- `w`, `l`: Instance params (different for m1 vs m2)

Instance params **must** override model params, otherwise every device would be identical.

### For MIR→Python Implementation

```python
def load_param(param_kind, inst_data, model_data):
    if param_kind.param in inst_data.params:
        # Instance-specific value
        return inst_data.params[param_kind.param]
    else:
        # Fall back to model-level default
        return model_data.params[param_kind.param]
```

---

## 5. Implementation Strategy Recommendation

### Option C: Start with Simple Model (Resistor)

**Recommended approach**:

1. **Phase 1**: Resistor with no cache slots
   - Only `setup_model()` and `eval()`
   - Skip `setup_instance()` initially
   - Hardcode parameter as constant for first prototype

2. **Phase 2**: Add parameter handling
   - Implement `setup_model()` with parameter loading
   - Add `param_given` flag checking

3. **Phase 3**: Capacitor with cache slots
   - Implement `setup_instance()`
   - Add cache slot allocation and loading

4. **Phase 4**: Diode with nonlinearity
   - Tests Jacobian computation
   - Tests conditional branches

### Why Start with Resistor?

**Resistor Verilog-A**:
```verilog
module resistor(p, n);
    inout p, n;
    electrical p, n;
    parameter real r = 1000;

    analog begin
        I(p, n) <+ V(p, n) / r;
    end
endmodule
```

**MIR (simplified)**:
```
entry:
  %r = param 0          // Load r parameter
  %v = voltage (p, n)   // Load V(p,n) from prev_result
  %i = fdiv %v, %r      // Compute I = V/r
  residual(%i, 2)       // Store to residual[2]
  %g = fdiv 1.0, %r     // dI/dV = 1/r
  jacobian(2, 0, %g)    // Store to J[2,0]
  exit
```

**No cache slots needed** because:
- `r` is a parameter (not computed)
- `v` comes from previous solution
- `i` and `g` are computed fresh each iteration

### Capacitor Requires Cache

**Capacitor Verilog-A**:
```verilog
module capacitor(p, n);
    inout p, n;
    parameter real c = 1e-12;
    parameter real temp = 300;

    analog begin
        real c_eff;
        c_eff = c * (1 + temp_coef * (temp - 300));  // OP-independent
        I(p, n) <+ ddt(c_eff * V(p, n));
    end
endmodule
```

`c_eff` is **OP-independent** (doesn't depend on V(p,n)), so it gets cached:
- Computed once in `setup_instance()`
- Loaded from cache in `eval()`

### Incremental Validation

```python
# Phase 1: Resistor eval only
def test_resistor_eval():
    r = 1000.0
    v_nodes = [5.0, 0.0]  # V(p)=5V, V(n)=0V
    residual, jacobian = eval_resistor(r, v_nodes)
    assert residual[2] == 5.0 / 1000.0  # I = V/R
    assert jacobian[2, 0] == 1.0 / 1000.0  # dI/dVp

# Phase 2: With setup_model
def test_resistor_with_setup():
    model = setup_model_resistor(param_given=False)
    assert model.r == 1000.0  # Default applied

# Phase 3: Capacitor with cache
def test_capacitor_with_cache():
    inst = setup_instance_capacitor(temp=350)
    assert inst.cache[0] == c * (1 + coef * 50)  # c_eff cached
```

---

## Summary Table

| Question | Answer | Implication for MIR→Python |
|----------|--------|---------------------------|
| **1. Parameter flow** | Simulator pre-populates, setup_model validates | Need struct with pre-filled params |
| **2. Callbacks** | Push errors to result array | Implement as Python exceptions |
| **3. Cache selection** | OP-dependency taint analysis | Implement taint propagation |
| **4. Parameter priority** | Instance overrides model (correct) | Dict lookup: instance first, then model |
| **5. Start strategy** | Resistor (Phase 1) → Capacitor (Phase 3) | Incremental complexity |

---

## Next Steps for Implementation

1. **Read resistor.va compiled MIR** - see actual structure
2. **Implement taint propagation** - mark OP-dependent instructions
3. **Write resistor eval() in Python** - validate against VACASK
4. **Add parameter handling** - implement setup_model()
5. **Move to capacitor** - implement cache slots

This incremental approach ensures each phase works before adding complexity.
