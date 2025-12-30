"""Debug utilities for JAX-SPICE."""

from jax_spice.debug.trace_monitor import (
    trace_monitor,
    reset_traces,
    get_trace_counts,
    report_traces,
    print_traces,
    TraceScope,
    monitor_dict,
)

__all__ = [
    'trace_monitor',
    'reset_traces',
    'get_trace_counts',
    'report_traces',
    'print_traces',
    'TraceScope',
    'monitor_dict',
]
