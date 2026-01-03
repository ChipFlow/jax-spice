"""ngspice test discovery and registry.

Discovers test circuits from vendor/ngspice/tests/ and categorizes them
by device type and analysis compatibility with JAX-SPICE.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
NGSPICE_ROOT = PROJECT_ROOT / "vendor" / "ngspice"
NGSPICE_TESTS = NGSPICE_ROOT / "tests"


@dataclass
class NgspiceTestCase:
    """Specification for an ngspice regression test."""

    name: str
    netlist_path: Path
    analysis_type: str  # 'tran', 'dc', 'ac', 'op'
    device_types: Set[str] = field(default_factory=set)
    expected_nodes: List[str] = field(default_factory=list)
    rtol: float = 0.05  # Relative tolerance (5%)
    atol: float = 1e-9  # Absolute tolerance
    xfail: bool = False  # Expected to fail
    xfail_reason: str = ""
    skip: bool = False  # Skip this test
    skip_reason: str = ""


def discover_ngspice_tests() -> List[NgspiceTestCase]:
    """Discover test cases from vendor/ngspice/tests/.

    Scans for .sp, .cir, .spice files and analyzes them.

    Returns:
        List of discovered test cases
    """
    tests = []

    if not NGSPICE_TESTS.exists():
        return tests

    # Scan for test files
    for pattern in ['**/*.sp', '**/*.cir', '**/*.spice']:
        for netlist in NGSPICE_TESTS.glob(pattern):
            test_case = analyze_netlist(netlist)
            if test_case:
                tests.append(test_case)

    return tests


def analyze_netlist(netlist_path: Path) -> Optional[NgspiceTestCase]:
    """Analyze a netlist file to extract test parameters.

    Args:
        netlist_path: Path to ngspice netlist

    Returns:
        NgspiceTestCase or None if not a valid test
    """
    try:
        content = netlist_path.read_text()
    except Exception:
        return None

    content_lower = content.lower()

    # Skip files without analysis commands
    if not any(x in content_lower for x in ['.tran', '.dc', '.ac', '.op']):
        return None

    # Determine analysis type (priority: tran > dc > ac > op)
    analysis_type = 'op'
    if '.tran' in content_lower:
        analysis_type = 'tran'
    elif '.ac' in content_lower:
        analysis_type = 'ac'
    elif '.dc' in content_lower:
        analysis_type = 'dc'

    # Detect device types
    device_types = _detect_device_types(content)

    # Extract expected output nodes from .print or .plot commands
    expected_nodes = _extract_output_nodes(content)

    # Generate test name from path
    rel_path = netlist_path.relative_to(NGSPICE_TESTS)
    name = str(rel_path).replace('/', '_').replace('\\', '_').replace('.', '_')

    return NgspiceTestCase(
        name=name,
        netlist_path=netlist_path,
        analysis_type=analysis_type,
        device_types=device_types,
        expected_nodes=expected_nodes or ['1'],  # Default to node 1
    )


def _detect_device_types(content: str) -> Set[str]:
    """Detect device types used in the netlist.

    Args:
        content: Netlist content

    Returns:
        Set of device type names
    """
    device_types: Set[str] = set()

    # Device patterns (first character of instance name)
    device_patterns = {
        'r': 'resistor',
        'c': 'capacitor',
        'l': 'inductor',
        'd': 'diode',
        'm': 'mosfet',
        'q': 'bjt',
        'j': 'jfet',
        'v': 'vsource',
        'i': 'isource',
        'e': 'vcvs',
        'f': 'cccs',
        'g': 'vccs',
        'h': 'ccvs',
        'x': 'subckt',
        'b': 'bsource',
        't': 'tline',
    }

    for line in content.split('\n'):
        line = line.strip().lower()
        # Skip comments and directives
        if line.startswith('*') or line.startswith('.') or not line:
            continue

        first_char = line[0] if line else ''
        if first_char in device_patterns:
            device_types.add(device_patterns[first_char])

    return device_types


def _extract_output_nodes(content: str) -> List[str]:
    """Extract output node names from .print/.plot commands.

    Args:
        content: Netlist content

    Returns:
        List of node names
    """
    nodes = []

    # Match v(node) or i(source) patterns in .print/.plot
    for match in re.finditer(
        r'(?:\.print|\.plot)\s+(?:tran|ac|dc)?\s+.*?v\((\w+)\)',
        content,
        re.IGNORECASE
    ):
        node = match.group(1)
        if node not in nodes:
            nodes.append(node)

    return nodes


# Devices supported by JAX-SPICE (via OpenVAF or built-in)
SUPPORTED_DEVICES = {
    'resistor',
    'capacitor',
    'inductor',
    'diode',
    'vsource',
    'isource',
    'mosfet',  # Via PSP103 or other VA models
}

# Curated list of tests known to work with JAX-SPICE
# These are simple circuits using supported devices
CURATED_TESTS: Dict[str, NgspiceTestCase] = {}


def _build_curated_tests() -> Dict[str, NgspiceTestCase]:
    """Build curated test list from known-working circuits."""
    tests = {}

    # Simple resistor test
    res_simple = NGSPICE_TESTS / "resistance" / "res_simple.cir"
    if res_simple.exists():
        tests['res_simple'] = NgspiceTestCase(
            name='res_simple',
            netlist_path=res_simple,
            analysis_type='tran',
            device_types={'resistor', 'vsource'},
            expected_nodes=['1'],
            rtol=0.01,
        )

    # Lowpass filter (RC circuit)
    lowpass = NGSPICE_TESTS / "filters" / "lowpass.cir"
    if lowpass.exists():
        tests['lowpass'] = NgspiceTestCase(
            name='lowpass',
            netlist_path=lowpass,
            analysis_type='ac',
            device_types={'resistor', 'capacitor', 'vsource'},
            expected_nodes=['2'],
            skip=True,
            skip_reason="AC analysis not yet fully supported",
        )

    # RTL inverter (uses BJT - not supported yet)
    rtlinv = NGSPICE_TESTS / "general" / "rtlinv.cir"
    if rtlinv.exists():
        tests['rtlinv'] = NgspiceTestCase(
            name='rtlinv',
            netlist_path=rtlinv,
            analysis_type='tran',
            device_types={'resistor', 'bjt', 'vsource'},
            expected_nodes=['3', '5'],
            skip=True,
            skip_reason="BJT device not yet supported",
        )

    return tests


# Build curated tests on module load
CURATED_TESTS = _build_curated_tests()


def get_compatible_tests(
    analysis_types: Optional[List[str]] = None,
    device_types: Optional[Set[str]] = None,
    include_unsupported: bool = False,
) -> List[NgspiceTestCase]:
    """Get tests compatible with JAX-SPICE capabilities.

    Args:
        analysis_types: Filter by analysis type (default: ['tran'])
        device_types: Only include tests using these devices
            (default: SUPPORTED_DEVICES)
        include_unsupported: If True, include tests with unsupported devices

    Returns:
        List of compatible test cases
    """
    if analysis_types is None:
        analysis_types = ['tran']  # Start with transient only

    if device_types is None:
        device_types = SUPPORTED_DEVICES

    tests = discover_ngspice_tests()
    compatible = []

    for test in tests:
        # Filter by analysis type
        if test.analysis_type not in analysis_types:
            continue

        # Filter by device types
        if not include_unsupported:
            if not test.device_types.issubset(device_types):
                continue

        compatible.append(test)

    return compatible


def get_tests_by_category() -> Dict[str, List[NgspiceTestCase]]:
    """Get all tests organized by their parent directory.

    Returns:
        Dict mapping category name to list of tests
    """
    tests = discover_ngspice_tests()
    by_category: Dict[str, List[NgspiceTestCase]] = {}

    for test in tests:
        # Use parent directory as category
        rel_path = test.netlist_path.relative_to(NGSPICE_TESTS)
        category = rel_path.parts[0] if rel_path.parts else 'unknown'

        if category not in by_category:
            by_category[category] = []
        by_category[category].append(test)

    return by_category
