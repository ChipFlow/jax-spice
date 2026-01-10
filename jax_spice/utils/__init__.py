"""JAX-SPICE utilities."""

from jax_spice.utils.ngspice import (
    NgspiceError,
    find_ngspice_binary,
    parse_control_section,
    run_ngspice,
)
from jax_spice.utils.rawfile import RawData, RawFile, rawread
from jax_spice.utils.waveform_compare import (
    ComparisonResult,
    WaveformComparison,
    compare_transient,
    compare_waveforms,
    find_vacask_binary,
    run_comparison,
    run_vacask,
)

__all__ = [
    # Raw file parsing
    'rawread',
    'RawFile',
    'RawData',
    # Waveform comparison
    'WaveformComparison',
    'ComparisonResult',
    'compare_waveforms',
    'compare_transient',
    'run_comparison',
    'run_vacask',
    'find_vacask_binary',
    # ngspice utilities
    'find_ngspice_binary',
    'parse_control_section',
    'run_ngspice',
    'NgspiceError',
]
