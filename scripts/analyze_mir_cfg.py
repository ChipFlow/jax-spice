#!/usr/bin/env -S uv run --with networkx --with pydot --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "networkx",
#     "pydot",
# ]
# ///
"""
Analyze MIR control flow graph and SSA dependencies using networkx.

This script helps debug openvaf_jax issues by:
1. Generating DOT files from Verilog-A models using openvaf-viz
2. Loading CFG into networkx for programmatic analysis
3. Tracing SSA value dependencies through PHI nodes
4. Finding paths between blocks to understand control flow

Usage:
    # Generate DOT and analyze CFG
    uv run scripts/analyze_mir_cfg.py path/to/model.va --func eval

    # Analyze existing DOT file
    uv run scripts/analyze_mir_cfg.py --dot /tmp/model_eval.dot

    # Trace paths to a specific block
    uv run scripts/analyze_mir_cfg.py path/to/model.va --func eval --target block123

    # Find PHI nodes and their predecessors
    uv run scripts/analyze_mir_cfg.py path/to/model.va --func eval --find-phis

    # Trace SSA value dependencies
    uv run scripts/analyze_mir_cfg.py path/to/model.va --func eval --trace-value v12345
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import networkx as nx
import pydot


def generate_dot(va_path: Path, func: str = "eval", output_dir: Optional[Path] = None) -> Path:
    """Generate DOT file from Verilog-A using openvaf-viz.

    Args:
        va_path: Path to Verilog-A file
        func: Function to visualize ('init' or 'eval')
        output_dir: Output directory (default: /tmp)

    Returns:
        Path to generated DOT file
    """
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())

    model_name = va_path.stem
    dot_path = output_dir / f"{model_name}_{func}.dot"

    # Use openvaf-viz to generate DOT
    # openvaf-viz is part of OpenVAF and should be in PATH or vendor/OpenVAF/target/release
    openvaf_viz = Path("vendor/OpenVAF/target/release/openvaf-viz")
    if not openvaf_viz.exists():
        # Try system PATH
        openvaf_viz = "openvaf-viz"

    # openvaf-viz uses --eval-only/--init-only and --format dot
    cmd = [str(openvaf_viz), str(va_path), "--output", str(dot_path), "--format", "dot"]
    if func == "eval":
        cmd.append("--eval-only")
    elif func == "init":
        cmd.append("--init-only")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Generated DOT file: {dot_path}")
        return dot_path
    except subprocess.CalledProcessError as e:
        print(f"Error running openvaf-viz: {e.stderr}")
        raise
    except FileNotFoundError:
        print("openvaf-viz not found. Build it with: cd vendor/OpenVAF && cargo build --release -p openvaf-viz")
        raise


def load_cfg(dot_path: Path) -> nx.DiGraph:
    """Load CFG from DOT file into networkx DiGraph.

    Args:
        dot_path: Path to DOT file

    Returns:
        NetworkX directed graph with nodes and edges
    """
    graphs = pydot.graph_from_dot_file(str(dot_path))
    if not graphs:
        raise ValueError(f"No graphs found in {dot_path}")

    dot_graph = graphs[0]
    G = nx.DiGraph()

    # Add nodes with labels (containing instructions)
    for node in dot_graph.get_nodes():
        name = node.get_name().strip('"')
        if name and name not in ('node', 'edge', 'graph', '\\n'):
            label = node.get_label()
            if label:
                # pydot returns double-escaped strings like '\\\\n', so we need
                # to replace literal backslash-n sequences with newlines
                label = label.strip('"')
                label = label.replace('\\\\l', '\n').replace('\\\\n', '\n')
                label = label.replace('\\l', '\n').replace('\\n', '\n')
            G.add_node(name, label=label or "")

    # Add edges with branch labels (T/F for true/false)
    for edge in dot_graph.get_edges():
        src = edge.get_source().strip('"')
        dst = edge.get_destination().strip('"')
        label = edge.get_label()
        if label:
            label = label.strip('"')
        G.add_edge(src, dst, label=label)

    return G


def find_phis(G: nx.DiGraph) -> dict:
    """Find all PHI nodes in the CFG.

    The DOT format from openvaf-viz shows abbreviated labels like:
        block31\\n(148 insts)\\nphi\\n...\\nbr
    So we look for 'phi' as a standalone instruction, not 'phi('.

    Returns:
        Dict mapping block name -> list of PHI instructions (or just ['phi'] if abbreviated)
    """
    phis = {}
    for node, data in G.nodes(data=True):
        label = data.get('label', '')
        lines = label.split('\n')
        # Look for 'phi' as instruction type (standalone word or with parens)
        phi_lines = []
        for line in lines:
            line_stripped = line.strip().lower()
            # Match 'phi', 'phi(...)', or lines containing phi instructions
            if line_stripped == 'phi' or line_stripped.startswith('phi(') or line_stripped.startswith('phi '):
                phi_lines.append(line.strip())
        if phi_lines:
            phis[node] = phi_lines
    return phis


def find_block_with_value(G: nx.DiGraph, value_id: str) -> list:
    """Find blocks that define or use a specific SSA value.

    Args:
        G: CFG graph
        value_id: SSA value ID (e.g., 'v12345')

    Returns:
        List of (block_name, context) tuples
    """
    results = []
    for node, data in G.nodes(data=True):
        label = data.get('label', '')
        if value_id in label:
            # Find the line containing the value
            for line in label.split('\n'):
                if value_id in line:
                    if line.strip().startswith(value_id):
                        results.append((node, f"DEFINES: {line.strip()}"))
                    else:
                        results.append((node, f"USES: {line.strip()}"))
    return results


def trace_value_deps(G: nx.DiGraph, value_id: str, depth: int = 5) -> dict:
    """Trace dependencies of an SSA value through PHI nodes.

    Args:
        G: CFG graph
        value_id: SSA value ID to trace
        depth: Maximum depth to trace

    Returns:
        Dict with dependency information
    """
    result = {
        'value': value_id,
        'defined_in': [],
        'used_in': [],
        'phi_sources': [],
    }

    for node, data in G.nodes(data=True):
        label = data.get('label', '')
        for line in label.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Check if this line defines the value
            if line.startswith(f"{value_id} ="):
                result['defined_in'].append((node, line))

                # If it's a PHI, extract sources
                if 'phi(' in line.lower():
                    # Parse phi(v1 from block1, v2 from block2, ...)
                    phi_content = line[line.find('(')+1:line.rfind(')')]
                    sources = []
                    for part in phi_content.split(','):
                        part = part.strip()
                        if ' from ' in part:
                            val, blk = part.split(' from ')
                            sources.append({'value': val.strip(), 'block': blk.strip()})
                    result['phi_sources'] = sources

            # Check if this line uses the value
            elif value_id in line and not line.startswith(value_id):
                result['used_in'].append((node, line))

    return result


def find_paths_to_block(G: nx.DiGraph, target: str, max_depth: int = 10) -> dict:
    """Find all paths leading to a target block.

    Args:
        G: CFG graph
        target: Target block name
        max_depth: Maximum path length

    Returns:
        Dict with predecessor info and paths
    """
    if target not in G:
        return {'error': f"Block {target} not found in graph"}

    result = {
        'target': target,
        'direct_predecessors': list(G.predecessors(target)),
        'direct_successors': list(G.successors(target)),
        'paths_from_entry': [],
    }

    # Find entry block (usually has no predecessors or is named 'entry'/'block0')
    entry_blocks = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not entry_blocks:
        entry_blocks = [n for n in G.nodes() if n in ('entry', 'block0', 'block3')]

    # Find paths from entry to target
    for entry in entry_blocks[:3]:  # Limit entries to check
        try:
            paths = list(nx.all_simple_paths(G, entry, target, cutoff=max_depth))
            for path in paths[:5]:  # Limit paths shown
                path_with_labels = []
                for i in range(len(path) - 1):
                    edge_label = G.edges[path[i], path[i+1]].get('label', '')
                    path_with_labels.append((path[i], edge_label))
                path_with_labels.append((path[-1], ''))
                result['paths_from_entry'].append(path_with_labels)
        except nx.NetworkXNoPath:
            pass

    return result


def find_branch_points(G: nx.DiGraph) -> list:
    """Find all conditional branch points in the CFG.

    Returns:
        List of (block, [(succ1, label1), (succ2, label2)]) tuples
    """
    branches = []
    for node in G.nodes():
        succs = list(G.successors(node))
        if len(succs) == 2:
            labels = [G.edges[node, s].get('label', '') for s in succs]
            if 'T' in labels or 'F' in labels:
                branches.append((node, list(zip(succs, labels))))
    return branches


def analyze_phi_block(G: nx.DiGraph, block: str) -> dict:
    """Detailed analysis of a block containing PHI nodes.

    Args:
        G: CFG graph
        block: Block name to analyze

    Returns:
        Dict with PHI analysis
    """
    if block not in G:
        return {'error': f"Block {block} not found"}

    label = G.nodes[block].get('label', '')
    preds = list(G.predecessors(block))

    result = {
        'block': block,
        'predecessors': preds,
        'predecessor_edges': {},
        'phis': [],
    }

    # Get edge labels from predecessors
    for pred in preds:
        result['predecessor_edges'][pred] = G.edges[pred, block].get('label', 'jump')

    # Parse PHI nodes
    for line in label.split('\n'):
        line = line.strip()
        if 'phi(' in line.lower():
            phi_info = {'instruction': line, 'sources': {}}

            # Extract result variable
            if '=' in line:
                phi_info['result'] = line.split('=')[0].strip()

            # Parse sources
            phi_content = line[line.find('(')+1:line.rfind(')')]
            for part in phi_content.split(','):
                part = part.strip()
                if ' from ' in part:
                    val, blk = part.split(' from ')
                    phi_info['sources'][blk.strip()] = val.strip()

            result['phis'].append(phi_info)

    return result


def print_summary(G: nx.DiGraph):
    """Print summary of the CFG."""
    print(f"\n{'='*60}")
    print(f"CFG Summary")
    print(f"{'='*60}")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    # Count PHI blocks
    phi_blocks = find_phis(G)
    print(f"Blocks with PHIs: {len(phi_blocks)}")

    # Count branch points
    branches = find_branch_points(G)
    print(f"Branch points: {len(branches)}")

    # Find entry/exit
    entries = [n for n in G.nodes() if G.in_degree(n) == 0]
    exits = [n for n in G.nodes() if G.out_degree(n) == 0]
    print(f"Entry blocks: {entries[:5]}{'...' if len(entries) > 5 else ''}")
    print(f"Exit blocks: {exits[:5]}{'...' if len(exits) > 5 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MIR control flow graph and SSA dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('va_path', nargs='?', help='Path to Verilog-A file')
    parser.add_argument('--dot', help='Path to existing DOT file (skip generation)')
    parser.add_argument('--func', default='eval', choices=['init', 'eval'],
                        help='Function to analyze (default: eval)')
    parser.add_argument('--target', help='Target block to trace paths to')
    parser.add_argument('--find-phis', action='store_true', help='Find and list all PHI nodes')
    parser.add_argument('--trace-value', help='Trace dependencies of an SSA value (e.g., v12345)')
    parser.add_argument('--analyze-block', help='Detailed analysis of a specific block')
    parser.add_argument('--branches', action='store_true', help='List all branch points')
    parser.add_argument('--output-dir', help='Output directory for DOT file')

    args = parser.parse_args()

    # Get DOT file path
    if args.dot:
        dot_path = Path(args.dot)
        if not dot_path.exists():
            print(f"Error: DOT file not found: {dot_path}")
            sys.exit(1)
    elif args.va_path:
        va_path = Path(args.va_path)
        if not va_path.exists():
            print(f"Error: VA file not found: {va_path}")
            sys.exit(1)
        output_dir = Path(args.output_dir) if args.output_dir else None
        try:
            dot_path = generate_dot(va_path, args.func, output_dir)
        except Exception as e:
            print(f"Failed to generate DOT: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Load CFG
    print(f"Loading CFG from {dot_path}")
    G = load_cfg(dot_path)
    print_summary(G)

    # Execute requested analysis
    if args.find_phis:
        print(f"\n{'='*60}")
        print("PHI Nodes")
        print(f"{'='*60}")
        phis = find_phis(G)
        for block, phi_lines in sorted(phis.items()):
            print(f"\n{block} ({len(phi_lines)} PHIs):")
            for line in phi_lines[:10]:
                print(f"  {line[:100]}{'...' if len(line) > 100 else ''}")
            if len(phi_lines) > 10:
                print(f"  ... and {len(phi_lines) - 10} more")

    if args.target:
        print(f"\n{'='*60}")
        print(f"Paths to {args.target}")
        print(f"{'='*60}")
        paths = find_paths_to_block(G, args.target)
        if 'error' in paths:
            print(paths['error'])
        else:
            print(f"Direct predecessors: {paths['direct_predecessors']}")
            print(f"Direct successors: {paths['direct_successors']}")
            print(f"\nPaths from entry:")
            for path in paths['paths_from_entry']:
                path_str = ' -> '.join(f"{b}[{l}]" if l else b for b, l in path)
                print(f"  {path_str}")

    if args.trace_value:
        print(f"\n{'='*60}")
        print(f"Tracing value: {args.trace_value}")
        print(f"{'='*60}")
        deps = trace_value_deps(G, args.trace_value)

        if deps['defined_in']:
            print(f"\nDefined in:")
            for block, line in deps['defined_in']:
                print(f"  {block}: {line[:80]}{'...' if len(line) > 80 else ''}")

        if deps['phi_sources']:
            print(f"\nPHI sources:")
            for src in deps['phi_sources']:
                print(f"  {src['value']} from {src['block']}")

        if deps['used_in']:
            print(f"\nUsed in ({len(deps['used_in'])} places):")
            for block, line in deps['used_in'][:10]:
                print(f"  {block}: {line[:80]}{'...' if len(line) > 80 else ''}")

    if args.analyze_block:
        print(f"\n{'='*60}")
        print(f"Block Analysis: {args.analyze_block}")
        print(f"{'='*60}")
        analysis = analyze_phi_block(G, args.analyze_block)
        if 'error' in analysis:
            print(analysis['error'])
        else:
            print(f"Predecessors: {analysis['predecessors']}")
            print(f"Edge labels: {analysis['predecessor_edges']}")
            print(f"\nPHI nodes ({len(analysis['phis'])}):")
            for phi in analysis['phis']:
                print(f"\n  {phi.get('result', '?')} = phi()")
                for blk, val in phi['sources'].items():
                    print(f"    {val} from {blk}")

    if args.branches:
        print(f"\n{'='*60}")
        print("Branch Points")
        print(f"{'='*60}")
        branches = find_branch_points(G)
        for block, succs in branches[:50]:
            succs_str = ', '.join(f"{s}[{l}]" for s, l in succs)
            print(f"  {block} -> {succs_str}")
        if len(branches) > 50:
            print(f"  ... and {len(branches) - 50} more")


if __name__ == '__main__':
    main()
