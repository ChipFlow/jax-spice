# Experimental Feature Tests

These tests cover features under active development.
They are excluded from CI and may fail.

## Contents

- `test_hb_analysis.py` - Harmonic balance analysis (not yet implemented)
- `test_noise_analysis.py` - Noise analysis (not yet implemented)
- `test_xfer_analysis.py` - Transfer function analysis (not yet implemented)

## Running Experimental Tests

```bash
# Run experimental tests (expect failures)
JAX_PLATFORMS=cpu uv run pytest tests/experimental/ -v

# Run main test suite (excludes experimental)
JAX_PLATFORMS=cpu uv run pytest tests/ -v --ignore=tests/experimental
```
