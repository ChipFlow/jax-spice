#!/usr/bin/env -S uv run --script
"""Generate NxN array multiplier benchmark in VACASK format.

Produces a parameterized CMOS array multiplier using the same gate subcircuits
(and, nor, not) as the c6288 benchmark. The generated circuit uses half adders
in the first row and full adders in subsequent rows.

Gate decomposition:
  XOR(a,b) = NOT(NOR(AND(a, NOT(b)), AND(NOT(a), b)))
  Half adder: sum=XOR(a,b), cout=AND(a,b) — 7 gates, 22 MOSFETs
  Full adder: sum=XOR(XOR(a,b), cin), cout=OR(AND(a,b), AND(XOR(a,b), cin))
             where OR(x,y) = NOT(NOR(x,y)) — 16 gates, 50 MOSFETs

Usage:
    uv run python scripts/generate_multiplier.py --bits 64 \\
        --output vajax/benchmarks/data/mul64/
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def generate_multiplier_inc(n: int) -> str:
    """Generate multiplier.inc with gate defs, adder subcircuits, and NxN multiplier."""
    lines: list[str] = []
    lines.append(f"// Gate definitions, {n}x{n} array multiplier")
    lines.append("")
    lines.append("global vdd vss")
    lines.append("ground 0")
    lines.append("")

    # Gate subcircuits
    lines.append("subckt not(out in)")
    lines.append("  mp (out in vdd vdd) pmos w=1u l=0.2u")
    lines.append("  mn (out in vss vss) nmos w=0.5u l=0.2u")
    lines.append("ends")
    lines.append("")
    lines.append("subckt nor(out in1 in2)")
    lines.append("  mp2 (int in2 vdd vdd) pmos w=1u l=0.2u")
    lines.append("  mp1 (out in1 int vdd) pmos w=1u l=0.2u")
    lines.append("  mn1 (out in1 vss vss) nmos w=0.5u l=0.2u")
    lines.append("  mn2 (out in2 vss vss) nmos w=0.5u l=0.2u")
    lines.append("ends")
    lines.append("")
    lines.append("subckt and(out in1 in2)")
    lines.append("  mp2 (outx in2 vdd vdd) pmos w=1u l=0.2u")
    lines.append("  mp1 (outx in1 vdd vdd) pmos w=1u l=0.2u")
    lines.append("  mn1 (outx in1 int vss) nmos w=0.5u l=0.2u")
    lines.append("  mn2 (int  in2 vss vss) nmos w=0.5u l=0.2u")
    lines.append("  mp3 (out outx vdd vdd) pmos w=1u l=0.2u")
    lines.append("  mn3 (out outx vss vss) nmos w=0.5u l=0.2u")
    lines.append("ends")
    lines.append("")

    # XOR subcircuit
    lines.append("// XOR: NOT(NOR(AND(a, NOT(b)), AND(NOT(a), b)))")
    lines.append("subckt xor(out a b)")
    lines.append("  inv_a (na a) not")
    lines.append("  inv_b (nb b) not")
    lines.append("  a1 (t1 a nb) and")
    lines.append("  a2 (t2 na b) and")
    lines.append("  n1 (t3 t1 t2) nor")
    lines.append("  inv_out (out t3) not")
    lines.append("ends")
    lines.append("")

    # Half adder
    lines.append("subckt ha(sum cout a b)")
    lines.append("  x1 (sum a b) xor")
    lines.append("  a1 (cout a b) and")
    lines.append("ends")
    lines.append("")

    # Full adder
    lines.append("subckt fa(sum cout a b cin)")
    lines.append("  x1 (ab_xor a b) xor")
    lines.append("  x2 (sum ab_xor cin) xor")
    lines.append("  a1 (ab_and a b) and")
    lines.append("  a2 (xor_cin ab_xor cin) and")
    lines.append("  n1 (cout_n ab_and xor_cin) nor")
    lines.append("  inv_cout (cout cout_n) not")
    lines.append("ends")
    lines.append("")

    # Generate multiplier subcircuit with direct output naming
    a_ports = " ".join(f"a{i}" for i in range(n))
    b_ports = " ".join(f"b{i}" for i in range(n))
    p_ports = " ".join(f"p{i}" for i in range(2 * n - 1))

    lines.append(f"subckt multiplier_{n}x{n}({a_ports} {b_ports} {p_ports})")
    lines.append("")

    # Partial products
    lines.append("  // Partial products: pp_i_j = a[i] AND b[j]")
    for i in range(n):
        for j in range(n):
            lines.append(f"  pp_{i}_{j} (pp{i}_{j} a{i} b{j}) and")
    lines.append("")

    # Adder array with direct output naming.
    # Output bit mapping:
    #   p[0] = pp[0][0] (via NOT-NOT buffer)
    #   p[1..N-1]: column 0 sums of rows 0..N-2
    #   p[N..2N-3]: last row sums at columns 1..N-2
    #   p[2N-2]: last row carry at column N-2
    # Internal sums/carries use s{row}_{col} and c{row}_{col} names;
    # outputs are named p{k} directly where they connect to output ports.

    last_row = n - 2

    def sum_name(row: int, col: int) -> str:
        """Get the net name for a sum output."""
        if col == 0:
            return f"p{row + 1}"
        if row == last_row:
            return f"p{n + col - 1}" if col < n - 1 else f"s{row}_{col}"
        return f"s{row}_{col}"

    def carry_name(row: int, col: int) -> str:
        """Get the net name for a carry output."""
        if row == last_row and col == n - 2:
            return f"p{2 * n - 2}"
        return f"c{row}_{col}"

    # p[0] = pp[0][0] via NOT-NOT buffer (can't alias nets in VACASK)
    lines.append("  // Output buffer for p0 = pp[0][0]")
    lines.append("  buf_p0_inv (p0_n pp0_0) not")
    lines.append("  buf_p0 (p0 p0_n) not")
    lines.append("")

    # Row 0: half adders
    lines.append("  // Row 0: half adders")
    for j in range(n - 1):
        s = sum_name(0, j)
        c = carry_name(0, j)
        lines.append(f"  ha_0_{j} ({s} {c} pp0_{j + 1} pp1_{j}) ha")
    lines.append("")

    # Rows 1..N-2: mixed half/full adders
    for row in range(1, n - 1):
        lines.append(f"  // Row {row}")
        for j in range(n - 1):
            s = sum_name(row, j)
            c = carry_name(row, j)

            # "b" input: partial product from row+1
            b_in = f"pp{row + 1}_{j}"

            # "a" input: from previous row
            if j < n - 2:
                a_in = sum_name(row - 1, j + 1)
            else:
                # Last column: carry from previous row's last column
                a_in = carry_name(row - 1, n - 2)

            if j == 0:
                lines.append(f"  ha_{row}_{j} ({s} {c} {a_in} {b_in}) ha")
            else:
                cin = carry_name(row, j - 1)
                lines.append(f"  fa_{row}_{j} ({s} {c} {a_in} {b_in} {cin}) fa")
        lines.append("")

    lines.append("ends")
    lines.append("")

    return "\n".join(lines)


def generate_runme_sim(n: int) -> str:
    """Generate runme.sim top-level test circuit."""
    lines: list[str] = []
    lines.append(f"// {n}x{n} array multiplier benchmark")
    lines.append("")
    lines.append('load "psp103v4.osdi"')
    lines.append('load "spice/resistor.osdi"')
    lines.append("")
    lines.append('include "models.inc"')
    lines.append('include "multiplier.inc"')
    lines.append("model v vsource")
    lines.append("model r sp_resistor")
    lines.append("")
    lines.append("parameters vdd=1.2")
    lines.append("")

    # Driver subcircuit (same as c6288)
    lines.append("subckt drv(out)")
    lines.append("  parameters v0=0 v1=1")
    lines.append("  parameters rs=1 delay=0.1n rise=0.1n")
    lines.append('  vdrv (int 0) v type="pulse" val0=v0*vdd val1=v1*vdd delay=delay rise=rise')
    lines.append("  rdrv (int out) r r=rs")
    lines.append("ends")
    lines.append("")

    # Power supplies
    lines.append("vdd (vdd 0) v dc=vdd")
    lines.append("vss (vss 0) v dc=0")
    lines.append("")

    # Test circuit
    lines.append(f"subckt mul{n}_test()")

    # Multiplier instance with wrapped port list
    a_ports = " ".join(f"a{i}" for i in range(n))
    b_ports = " ".join(f"b{i}" for i in range(n))
    p_ports = " ".join(f"p{i}" for i in range(2 * n - 1))
    lines.append("  u1 (")
    lines.append(f"    {a_ports}")
    lines.append(f"    {b_ports}")
    lines.append(f"    {p_ports}")
    lines.append(f"  ) multiplier_{n}x{n}")
    lines.append("")

    # Input drivers
    for prefix in ("a", "b"):
        for i in range(n):
            pad = " " * (len(str(n - 1)) - len(str(i)))
            lines.append(f"  d{prefix}{i}{pad} ({prefix}{i}){pad} drv v0=0 v1=1")
        lines.append("")

    lines.append(f"  // All {n} bits high: {hex((1 << n) - 1)} x {hex((1 << n) - 1)}")
    lines.append("ends")
    lines.append("")

    # Control block
    lines.append("control")
    lines.append("  options nr_convtol=1 nr_bypasstol=1 nr_bypass=0 nr_contbypass=1")
    lines.append("  options tran_lteratio=1.5")
    lines.append("")

    # Save output signals (in groups of 10 for readability)
    n_out = 2 * n - 1
    for start in range(0, n_out, 10):
        end = min(start + 10, n_out)
        saves = " ".join(f"v(p{i})" for i in range(start, end))
        lines.append(f"  save {saves}")

    # Save a few input signals for debugging
    lines.append(f"  save v(a0) v(a{n - 1}) v(b0) v(b{n - 1})")
    lines.append("")

    lines.append(f'  elaborate circuit("mul{n}_test")')
    lines.append("")
    lines.append('  analysis tranmul tran stop=2n step=2p icmode="uic"')
    lines.append("")
    lines.append("  print stats")
    lines.append("endc")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NxN array multiplier benchmark")
    parser.add_argument("--bits", type=int, default=64, help="Multiplier width (default: 64)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: vajax/benchmarks/data/mul{N}/)",
    )
    args = parser.parse_args()

    n = args.bits
    assert n >= 2, "Multiplier must be at least 2x2"

    output_dir = args.output or (PROJECT_ROOT / "vajax" / "benchmarks" / "data" / f"mul{n}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate multiplier.inc
    multiplier_inc = generate_multiplier_inc(n)
    (output_dir / "multiplier.inc").write_text(multiplier_inc)

    # Generate runme.sim
    runme_sim = generate_runme_sim(n)
    (output_dir / "runme.sim").write_text(runme_sim)

    # Copy models.inc from c6288 (identical PSP103 model cards)
    c6288_models = (
        PROJECT_ROOT / "vendor" / "VACASK" / "benchmark" / "c6288" / "vacask" / "models.inc"
    )
    if c6288_models.exists():
        shutil.copy2(c6288_models, output_dir / "models.inc")
    else:
        print(f"WARNING: {c6288_models} not found, models.inc not copied")

    # Print statistics
    n_pp = n * n
    n_ha_row0 = n - 1
    n_ha_other = n - 2  # one per row (column 0)
    n_fa = (n - 2) * (n - 2)  # rows 1..N-2, columns 1..N-2
    n_ha_total = n_ha_row0 + n_ha_other
    n_adders = n_ha_total + n_fa

    # Gate counts: AND=6 MOSFETs, NOR=4, NOT=2
    # XOR = 3 NOT + 2 AND + 1 NOR = 6+12+4 = 22 MOSFETs
    # HA = 1 XOR + 1 AND = 22+6 = 28 MOSFETs (but XOR has 6 gate instances)
    # HA gates: 6 (xor) + 1 (and) = 7 gates
    # FA = 2 XOR + 2 AND + 1 NOR + 1 NOT = 44+12+4+2 = 62 MOSFETs
    # FA gates: 12 (2 xor) + 2 (and) + 1 (nor) + 1 (not) = 16 gates
    # PP AND: 6 MOSFETs each
    # p0 buffer: 2 NOT = 4 MOSFETs

    mosfets_pp = n_pp * 6
    mosfets_ha = n_ha_total * 28
    mosfets_fa = n_fa * 62
    mosfets_buf = 4
    mosfets_total = mosfets_pp + mosfets_ha + mosfets_fa + mosfets_buf

    # Drivers: each has 1 vsource + 1 resistor instance
    n_drivers = 2 * n

    print(f"Generated {n}x{n} array multiplier in {output_dir}")
    print(f"  Partial product ANDs: {n_pp:,}")
    print(f"  Half adders: {n_ha_total:,}")
    print(f"  Full adders: {n_fa:,}")
    print(f"  Total adder cells: {n_adders:,}")
    print(f"  Total MOSFETs: {mosfets_total:,}")
    print(f"  Input drivers: {n_drivers}")
    print("  Files: multiplier.inc, runme.sim, models.inc")


if __name__ == "__main__":
    main()
