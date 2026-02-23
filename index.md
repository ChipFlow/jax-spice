# JAX-SPICE CI Summary

_Last updated: 2026-02-23 23:24 UTC_

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
| rc | 1,000,000 | 0.0123 | 0.0019 | 6.60x | 3.7s |
| graetz | 1,000,000 | 0.0204 | 0.0038 | 5.41x | 9.5s |
| mul | 500,000 | 0.0406 | 0.0037 | 10.86x | 7.0s |
| ring | 19,999 | 0.5109 | 0.1092 | 4.68x | 153.0s |
| tb_dp | 299 | 0.0994 | N/A | N/A | 5.4s |

### GPU Benchmarks

| Benchmark | Steps | JAX-SPICE (ms/step) | VACASK (ms/step) | Ratio | Startup |
|-----------|-------|---------------------|------------------|-------|---------|
| mul | 500,000 | 0.4451 | N/A | N/A | 11.4s |
| c6288 | 1,000 | 19.8136 | 56.7875 | 0.35x | 225.1s |
| ring | 19,999 | 1.4938 | 0.0453 | 33.00x | 192.6s |
| rc | 1,000,000 | 0.2402 | 0.0009 | 256.83x | 3.1s |
| graetz | 1,000,000 | 0.3049 | 0.0019 | 161.34x | 8.0s |
| tb_dp | 299 | 4.8240 | N/A | N/A | 5.5s |


---

[View workflows](https://github.com/ChipFlow/jax-spice/actions) | 
[Repository](https://github.com/ChipFlow/jax-spice)
