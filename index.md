# JAX-SPICE CI Summary

_Last updated: 2026-02-23 23:35 UTC_

_Commit: [3c580822](https://github.com/ChipFlow/jax-spice/commit/3c580822b3e103e7023b784b6895ec2aeab28cb3)_

## Test Coverage

| Suite | Passed | Failed | Errors | Skipped | Total | Time |
|-------|--------|--------|--------|---------|-------|------|
| benchmark-dense | 5 | 0 | 0 | 1 | 6 | 515.2s | PASS |
| benchmark-sparse | 6 | 0 | 0 | 1 | 7 | 625.7s | PASS |
| benchmarks | 5 | 0 | 0 | 0 | 5 | 736.0s | PASS |
| ngspice | 0 | 0 | 0 | 71 | 71 | 248.8s | PASS |
| openvaf-py | 320 | 0 | 0 | 92 | 412 | 1032.1s | PASS |
| unit | 72 | 0 | 0 | 5 | 77 | 20.5s | PASS |
| xyce | 36 | 0 | 0 | 1893 | 1929 | 240.1s | PASS |
| **Total** | **444** | **0** | **0** | **2063** | **2507** | 3418.4s |


## Performance

### CPU Benchmarks

| Benchmark | Steps | JAX-SPICE (ms/step) | VACASK (ms/step) | Ratio | Startup |
|-----------|-------|---------------------|------------------|-------|---------|
| rc | 1,000,000 | 0.0116 | 0.0019 | 6.29x | 3.9s |
| graetz | 1,000,000 | 0.0190 | 0.0038 | 4.97x | 10.0s |
| mul | 500,000 | 0.0394 | 0.0037 | 10.56x | 7.2s |
| ring | 19,999 | 0.5135 | 0.1079 | 4.76x | 156.6s |
| tb_dp | 299 | 0.0982 | N/A | N/A | 5.7s |

### GPU Benchmarks

| Benchmark | Steps | JAX-SPICE (ms/step) | VACASK (ms/step) | Ratio | Startup |
|-----------|-------|---------------------|------------------|-------|---------|
| mul | 500,000 | 0.4444 | N/A | N/A | 11.2s |
| c6288 | 1,000 | 19.7680 | 56.7875 | 0.35x | 223.2s |
| ring | 19,999 | 1.4883 | 0.0453 | 32.88x | 188.8s |
| rc | 1,000,000 | 0.2377 | 0.0009 | 254.18x | 3.1s |
| graetz | 1,000,000 | 0.3046 | 0.0019 | 161.15x | 7.9s |
| tb_dp | 299 | 5.1086 | N/A | N/A | 5.6s |


---

[View workflows](https://github.com/ChipFlow/jax-spice/actions) | 
[Repository](https://github.com/ChipFlow/jax-spice)
