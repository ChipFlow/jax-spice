# OpenVAF MIR Opcodes

This document describes all opcodes in OpenVAF's Mid-level Intermediate Representation (MIR).

The opcodes are defined in `sourcegen/src/mir_instructions.rs` and generated to `openvaf/mir/src/instructions/generated.rs`.

## Instruction Formats

| Format | Args | Returns | Description |
|--------|------|---------|-------------|
| Unary | 1 | 1 | Single-operand operations |
| Binary | 2 | 1 | Two-operand operations |
| Branch | 1 | 0 | Conditional branch |
| Jump | 0 | 0 | Unconditional jump |
| Exit | 0 | 0 | Function exit |
| Call | var | var | Function call |
| PhiNode | var | 1 | SSA phi node |

---

## Unary Operations (1 arg → 1 result)

### Logical/Bitwise Negation

| Opcode | Name | Description |
|--------|------|-------------|
| `Inot` | inot | Integer bitwise NOT |
| `Bnot` | bnot | Boolean NOT |

### Arithmetic Negation

| Opcode | Name | Description |
|--------|------|-------------|
| `Fneg` | fneg | Float negation |
| `Ineg` | ineg | Integer negation |

### Type Casts

| Opcode | Name | Description |
|--------|------|-------------|
| `FIcast` | ficast | Float → Integer cast |
| `IFcast` | ifcast | Integer → Float cast |
| `BIcast` | bicast | Bool → Integer cast |
| `IBcast` | ibcast | Integer → Bool cast |
| `FBcast` | fbcast | Float → Bool cast |
| `BFcast` | bfcast | Bool → Float cast |

### Optimization Barrier

| Opcode | Name | Description |
|--------|------|-------------|
| `OptBarrier` | optbarrier | Prevents optimization across barrier |

### Math Functions

| Opcode | Name | Description |
|--------|------|-------------|
| `Sqrt` | sqrt | Square root |
| `Exp` | exp | Exponential (e^x) |
| `Ln` | ln | Natural logarithm |
| `Log` | log | Base-10 logarithm |
| `Clog2` | clog2 | Ceiling of log base 2 |
| `Floor` | floor | Floor function |
| `Ceil` | ceil | Ceiling function |

### Trigonometric Functions

| Opcode | Name | Description |
|--------|------|-------------|
| `Sin` | sin | Sine |
| `Cos` | cos | Cosine |
| `Tan` | tan | Tangent |
| `Asin` | asin | Arc sine |
| `Acos` | acos | Arc cosine |
| `Atan` | atan | Arc tangent |

### Hyperbolic Functions

| Opcode | Name | Description |
|--------|------|-------------|
| `Sinh` | sinh | Hyperbolic sine |
| `Cosh` | cosh | Hyperbolic cosine |
| `Tanh` | tanh | Hyperbolic tangent |
| `Asinh` | asinh | Inverse hyperbolic sine |
| `Acosh` | acosh | Inverse hyperbolic cosine |
| `Atanh` | atanh | Inverse hyperbolic tangent |

---

## Binary Operations (2 args → 1 result)

### Integer Arithmetic

| Opcode | Name | Description |
|--------|------|-------------|
| `Iadd` | iadd | Integer addition |
| `Isub` | isub | Integer subtraction |
| `Imul` | imul | Integer multiplication |
| `Idiv` | idiv | Integer division |
| `Irem` | irem | Integer remainder (modulo) |

### Integer Bitwise

| Opcode | Name | Description |
|--------|------|-------------|
| `Ishl` | ishl | Integer shift left |
| `Ishr` | ishr | Integer shift right |
| `Ixor` | ixor | Integer XOR |
| `Iand` | iand | Integer AND |
| `Ior` | ior | Integer OR |

### Float Arithmetic

| Opcode | Name | Description |
|--------|------|-------------|
| `Fadd` | fadd | Float addition |
| `Fsub` | fsub | Float subtraction |
| `Fmul` | fmul | Float multiplication |
| `Fdiv` | fdiv | Float division |
| `Frem` | frem | Float remainder |

### Integer Comparisons (→ bool)

| Opcode | Name | Description |
|--------|------|-------------|
| `Ilt` | ilt | Integer less than |
| `Igt` | igt | Integer greater than |
| `Ige` | ige | Integer greater or equal |
| `Ile` | ile | Integer less or equal |
| `Ieq` | ieq | Integer equal |
| `Ine` | ine | Integer not equal |

### Float Comparisons (→ bool)

| Opcode | Name | Description |
|--------|------|-------------|
| `Flt` | flt | Float less than |
| `Fgt` | fgt | Float greater than |
| `Fge` | fge | Float greater or equal |
| `Fle` | fle | Float less or equal |
| `Feq` | feq | Float equal |
| `Fne` | fne | Float not equal |

### String/Bool Comparisons

| Opcode | Name | Description |
|--------|------|-------------|
| `Seq` | seq | String equal |
| `Sne` | sne | String not equal |
| `Beq` | beq | Bool equal |
| `Bne` | bne | Bool not equal |

### Binary Math Functions

| Opcode | Name | Description |
|--------|------|-------------|
| `Hypot` | hypot | Hypotenuse: √(x² + y²) |
| `Atan2` | atan2 | Two-argument arc tangent |
| `Pow` | pow | Power: x^y |

---

## Control Flow

| Opcode | Format | Description |
|--------|--------|-------------|
| `Br` | Branch(1→0) | Conditional branch (takes condition, branches to then_dst or else_dst) |
| `Jmp` | Jump(0→0) | Unconditional jump to destination block |
| `Exit` | Exit(0→0) | Exit/return from function |
| `Call` | Call(var→var) | Function call with variable args/returns |
| `Phi` | PhiNode(var→1) | SSA phi node for merging values from different blocks |

---

## Summary

**Total: 69 opcodes** across 7 instruction formats.

### Opcode Distribution

- Unary operations: 30 opcodes
- Binary operations: 34 opcodes
- Control flow: 5 opcodes
