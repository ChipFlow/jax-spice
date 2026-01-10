# sim_back Initialization Module

The `Initialization` struct extracts operating-point independent code from the evaluation function. This code can be computed once during instance setup and cached, avoiding redundant computation during Newton iterations.

**Location:** `openvaf/sim_back/src/init.rs`

## Purpose

During circuit simulation, the evaluation function is called many times per operating point (Newton iterations). However, much of the computation depends only on:
- Model parameters (constant per instance)
- Instance parameters (constant per analysis)
- Temperature (constant per analysis)

By extracting this **OP-independent** code into a separate initialization function, we:
1. Reduce per-iteration computation
2. Enable caching of intermediate values
3. Improve simulation performance

---

## Data Structures

### `Initialization`

```rust
pub struct Initialization {
    /// The initialization MIR function
    pub func: Function,

    /// Symbol table for init function
    pub intern: HirInterner,

    /// Map from eval function values to cache slots
    pub cached_vals: IndexMap<Value, CacheSlot>,

    /// Cache slot metadata: (GVN class, result index) → type
    pub cache_slots: TiMap<CacheSlot, (PackedOption<ClassId>, u32), hir::Type>,
}
```

### `CacheSlot`

```rust
pub struct CacheSlot(pub u32);
```

A unique identifier for a cached value. The cache is an array of values stored in instance data, indexed by `CacheSlot`.

---

## Construction Process

### `Initialization::new()`

```rust
pub fn new(cx: &mut Context<'_>, gvn: GVN) -> Initialization
```

**Inputs:**
- `Context` with OP-dependence already computed (`refresh_op_dependent_insts()`)
- `GVN` from post-derivative optimization (provides equivalence classes)

**Process:**

#### Phase 1: Create Init Function Structure

```rust
// Create empty blocks matching eval function layout
for _ in 0..builder.func.layout.num_blocks() {
    builder.init.func.layout.make_block();
}
```

#### Phase 2: Split Blocks

For each block in the eval function:

```rust
fn split_block(&mut self, bb: Block) {
    for inst in bb.instructions() {
        if self.op_dependent_insts.contains(inst) {
            // Keep in eval, handle terminators specially
            self.handle_op_dependent(inst, bb);
        } else {
            // Copy to init, potentially cache result
            self.copy_instruction(inst, bb);
        }
    }
}
```

**OP-dependent instruction handling:**
- **Branch**: Convert to unconditional jump to else block
- **Jump/Exit**: Copy to init (maintain control structure)
- **Callbacks with `ignore_if_op_dependent`**: Remove entirely

**OP-independent instruction handling:**
- Copy instruction to init function
- Map values between functions
- Potentially mark for caching

#### Phase 3: Build Interner

```rust
fn build_init_intern(&mut self) -> AHashSet<Value>
```

- Copy relevant parameters to init's interner
- Track `CollapseImplicitEquation` outputs
- Return set of collapse values (need special handling)

#### Phase 4: Build Cache

```rust
fn build_init_cache(&mut self, gvn: &GVN, collapse_implicit: &AHashSet<Value>)
```

1. Run aggressive DCE on eval function to find actually-used cached values
2. For each cached value that survives DCE:
   - Determine its type (Real, Bool, etc.)
   - Assign a cache slot using GVN equivalence class
   - Convert eval value to a parameter reference
   - Add optbarrier in init for the cached value

**GVN deduplication**: Values in the same GVN equivalence class share a cache slot, avoiding redundant storage.

#### Phase 5: Optimize

```rust
fn optimize(&mut self, collapse_implicit: AHashSet<Value>)
```

- Simplify eval function CFG
- Run aggressive DCE on init function
- Simplify init function CFG

---

## Instruction Copying

### `copy_instruction()`

```rust
fn copy_instruction(&mut self, inst: Inst, bb: Block)
```

**Steps:**

1. **Copy instruction data** to init function's pools
2. **Handle callbacks** - Copy callback reference to init's interner
3. **Map values** - Create corresponding values in init function
4. **Determine caching**:

| Condition | Action |
|-----------|--------|
| Output value (optbarrier) | Cache underlying value |
| Tagged value (user variable) | Cache and convert to param |
| Safe to remove | Remove from eval |

**Why optbarriers?** Output values have optbarriers to prevent optimization from removing them. The underlying (pre-optbarrier) value is what we cache.

---

## Cache Slot Assignment

Cache slots are assigned using GVN equivalence classes:

```rust
fn ensure_cache_slot(&mut self, inst: Option<Inst>, res: usize, ty: Type) -> CacheSlot {
    let class = inst.and_then(|inst| gvn.inst_class(inst).expand());
    let equiv_class = class.unwrap_or_else(|| /* new class */);
    self.cache_slots.insert_full((equiv_class.into(), res as u32), ty).0
}
```

This means:
- Values computed by equivalent instructions share a slot
- Multi-result instructions use `res` index to distinguish
- Type is tracked for OSDI instance data generation

---

## Example

**Original Verilog-A:**
```verilog
module example(inout a, inout b);
    parameter real r = 1k;
    parameter real c = 1p;
    real g, tau;
    analog begin
        g = 1/r;           // OP-independent
        tau = r * c;       // OP-independent
        I(a,b) <+ g * V(a,b) + c * ddt(V(a,b));
    end
endmodule
```

**After Initialization extraction:**

**Init function:**
```
block0:
    v1 = fdiv 1.0, r      // g = 1/r
    v2 = optbarrier v1    // cache g
    v3 = fmul r, c        // tau = r*c
    v4 = optbarrier v3    // cache tau (if used)
    exit
```

**Eval function:**
```
block0:
    // g and tau are now parameters (loaded from cache)
    v10 = param[cache_slot_0]   // g
    v11 = voltage(a, b)
    v12 = fmul v10, v11         // g * V(a,b)
    // ... rest of evaluation
```

**cached_vals:**
```
{ v2 → CacheSlot(0), v4 → CacheSlot(1) }
```

---

## Integration with OSDI

The OSDI crate uses `Initialization` for:

### Instance Setup Function

```rust
// osdi_setup_instance()
for (val, slot) in init.cached_vals {
    let result = eval_init_function(val);
    instance_data.cache[slot] = result;
}
```

### Evaluation Function

```rust
// osdi_eval()
// Cached values accessed as parameters
let g = instance_data.cache[cache_slot_0];
```

### Instance Data Structure

```rust
struct InstanceData {
    // ... other fields
    cache: [CacheValue; num_cache_slots],
}
```

---

## Testing

Tests are in `init/tests.rs`:

```rust
#[test]
fn basic_caching() {
    let src = /* Verilog-A with OP-independent computations */;
    let (eval_func, init, _) = compile(src);
    expect_file![/* init snapshot */].assert_eq(&init);
    expect_file![/* eval snapshot */].assert_eq(&eval_func);
}
```

### Coverage Markers

| Marker | Description |
|--------|-------------|
| `ignore_if_op_dependent` | Callback removed when OP-dependent |
| `op_independent_output` | Output value cached |
| `cache_output` | Underlying value cached |

---

## Key Insights

1. **Two MIR functions** - Init and eval are separate functions after splitting

2. **GVN deduplication** - Equivalent computations share cache slots

3. **Optbarrier preservation** - Cached values get optbarriers in init to prevent DCE from removing them

4. **Control flow maintained** - OP-dependent branches become jumps in init, preserving structure

5. **Type tracking** - Cache slots know their types for correct OSDI instance data layout

6. **Collapse handling** - `CollapseImplicitEquation` outputs need special treatment as they affect simulation structure
