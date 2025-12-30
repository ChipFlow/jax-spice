"""JAX-SPICE utilities."""

from jax_spice.utils.rawfile import rawread, RawFile, RawData
from jax_spice.utils.waveform_compare import (
    WaveformComparison,
    ComparisonResult,
    compare_waveforms,
    compare_transient,
    run_comparison,
    run_vacask,
    find_vacask_binary,
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
]
