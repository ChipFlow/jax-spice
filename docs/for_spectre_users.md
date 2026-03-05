# For Spectre / ngspice Users

This guide maps familiar SPICE simulator concepts to VAJAX equivalents.

## Analysis Types

| Spectre | ngspice | VAJAX | Status |
|---------|---------|-------|--------|
| `.tran` | `.tran` | `engine.run_transient()` | Available |
| `.ac` | `.ac` | `engine.run_ac()` | Available |
| `.noise` | `.noise` | `engine.run_noise()` | Available |
| `.dc` | `.dc` | `engine.run_dc_sweep()` | Coming soon |
| `.op` | `.op` | Internal (computed automatically) | Available |
| `dcinc` | - | `engine.run_dcinc()` | Available |
| `dcxf` / `xf` | - | `engine.run_dcxf()` | Available |
| `acxf` / `xf` | - | `engine.run_acxf()` | Available |
| Corner / Monte Carlo | - | `engine.run_corners()` | Available (PVT sweep) |

## Netlist Conversion

VAJAX can convert netlists from ngspice, HSPICE, LTspice, and Spectre formats:

```bash
# Convert a Spectre netlist
vajax convert input.scs output.sim --dialect spectre

# Convert an ngspice netlist (default dialect)
vajax convert input.sp output.sim

# Other dialects
vajax convert input.sp output.sim --dialect hspice
vajax convert input.sp output.sim --dialect ltspice
```

### What Gets Converted

- Instance statements (resistors, capacitors, transistors, subcircuit calls)
- Model definitions and parameters
- Subcircuit definitions with parameters
- `.lib "file" section` library inclusion
- SI prefix notation (1k, 1u, 1n, 1p, etc.)
- Analysis statements (`.tran`, `.ac`)

### Spectre-Specific Features

The Spectre dialect handles:
- `library ... section ... endsection ... endlibrary` blocks
- Statistics blocks (`statistics { process { ... } mismatch { ... } }`)
- Spectre instance syntax (`name (nodes) model param=value`)
- Spectre SI prefixes (e.g., `M` for milli vs `Meg` for mega)

### Known Limitations

- Some advanced Spectre features (ocean scripts, Maestro setup) are not converted
- Mixed-signal (Verilog-AMS) wrappers are not supported
- Conversion may require manual review for complex designs

## Device Models

VAJAX supports Verilog-A compact models compiled at runtime via
[OpenVAF](https://openvaf.semimod.de/). If your PDK provides Verilog-A sources
(most modern PDKs do), VAJAX can use them directly.

### Bundled Models

| Model | Type | Notes |
|-------|------|-------|
| `resistor` | R | SPICE resistor with temperature coefficients |
| `capacitor` | C | Ideal capacitor |
| `inductor` | L | Ideal inductor |
| `diode` | D | SPICE diode model |
| `psp103` | MOSFET | PSP103 production MOSFET |
| `bsimbulk` | MOSFET | BSIM-BULK 106 |
| `bsimcmg` | MOSFET | BSIM-CMG (FinFET) |
| `bsimimg` | MOSFET | BSIM-IMG (FD-SOI) |
| `ekv` | MOSFET | EKV model |
| `hisim2` | MOSFET | HiSIM2 |
| `asmhemt` | HEMT | ASM-HEMT |
| `mvsg_cmc` | HEMT | MVSG CMC |

### Using Your PDK Models

To use a custom Verilog-A model, reference it in your netlist with a `load` statement:

```
load "path/to/your_model.va"
model mymod your_model_module_name (param1=value1 ...)
```

The `load` path is resolved relative to the netlist file location.

## Output Formats

| Format | Command | Compatible With |
|--------|---------|-----------------|
| Raw (binary) | `vajax circuit.sim -o results.raw` | ngspice, gwave |
| CSV | `vajax circuit.sim -o results.csv --format csv` | Excel, Python |
| JSON | `vajax circuit.sim -o results.json --format json` | Custom tools |

The default `.raw` format is binary-compatible with ngspice, so you can view
waveforms using ngspice's built-in plotter or external viewers like
[gwave](https://gwave.sourceforge.net/).

## Key Differences from Spectre

1. **No GUI** — VAJAX is a command-line and Python API tool. Use matplotlib
   for plotting, or export `.raw` files to your preferred waveform viewer.

2. **GPU acceleration** — Large circuits (1000+ nodes) benefit significantly
   from GPU acceleration. Use `engine.prepare(use_sparse=True)` for circuits
   over ~1000 nodes.

3. **JAX backend** — Jacobians are computed via automatic differentiation rather
   than hand-coded derivatives. This means any Verilog-A model works without
   modification.

4. **Python-first API** — While the CLI works for simple runs, the Python API
   gives you full control over simulation setup and post-processing.

## Quick Example: Spectre to VAJAX

**Spectre netlist** (`amp.scs`):
```spectre
// Simple common-source amplifier
simulator lang=spectre

include "pdk_models.scs" section=tt

subckt cs_amp (out in vdd vss)
  M0 (out in vss vss) nmos w=10u l=0.18u
  M1 (out bias vdd vdd) pmos w=20u l=0.18u
  R0 (vdd bias) resistor r=10k
ends cs_amp

I0 (out in vdd vss) cs_amp
V0 (vdd 0) vsource dc=1.8
V1 (in 0) vsource dc=0.9

myac ac freq=1k stop=1G dec=100
```

**Convert and run**:
```bash
# Convert to VAJAX format
vajax convert amp.scs amp.sim --dialect spectre

# Run the simulation
vajax amp.sim -o amp.raw
```

**Or use the Python API**:
```python
from vajax import CircuitEngine

engine = CircuitEngine("amp.sim")
engine.parse()

# AC analysis
ac_result = engine.run_ac(
    freq_start=1e3,
    freq_stop=1e9,
    mode='dec',
    points=100,
)
```
