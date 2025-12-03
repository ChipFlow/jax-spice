"""Benchmark circuits for JAX-SPICE

Provides benchmark circuits for testing and performance evaluation.
"""

from jax_spice.benchmarks.c6288 import (
    C6288Benchmark,
    run_c6288_sparse_dc,
    run_c6288_dense_dc,
)

__all__ = [
    "C6288Benchmark",
    "run_c6288_sparse_dc",
    "run_c6288_dense_dc",
]
