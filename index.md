# VAJAX CI Summary

_Last updated: 2026-02-24 02:07 UTC_

_Commit: [2cb5651b](https://github.com/ChipFlow/vajax/commit/2cb5651be1331fcb08506718e91e5fb270ee4490)_

## Test Coverage

| Suite | Passed | Failed | Errors | Skipped | Total | Time |
|-------|--------|--------|--------|---------|-------|------|
| benchmark-dense | 5 | 0 | 0 | 1 | 6 | 369.8s | PASS |
| benchmark-sparse | 6 | 0 | 0 | 1 | 7 | 382.2s | PASS |
| benchmarks | 5 | 0 | 0 | 0 | 5 | 366.9s | PASS |
| ngspice | 0 | 0 | 0 | 71 | 71 | 246.4s | PASS |
| openvaf-py | 320 | 0 | 0 | 92 | 412 | 1024.0s | PASS |
| unit | 72 | 0 | 0 | 5 | 77 | 21.1s | PASS |
| xyce | 36 | 0 | 0 | 1893 | 1929 | 268.0s | PASS |
| **Total** | **444** | **0** | **0** | **2063** | **2507** | 2678.4s |


## Performance

### CPU Benchmarks

| Benchmark | Steps | VAJAX (ms/step) | VACASK (ms/step) | Ratio | Startup |
|-----------|-------|---------------------|------------------|-------|---------|
| rc | 1,000,000 | 0.0123 | 0.0019 | 6.49x | 3.7s |
| graetz | 1,000,000 | 0.0218 | 0.0040 | 5.51x | 9.5s |
| mul | 500,000 | 0.0351 | 0.0040 | 8.78x | 7.3s |
| ring | 19,999 | 0.5265 | 0.1122 | 4.69x | 152.0s |
| tb_dp | 299 | 0.0855 | N/A | N/A | 5.4s |

_No gpu benchmarks benchmark data available._


---

[View workflows](https://github.com/ChipFlow/vajax/actions) | 
[Repository](https://github.com/ChipFlow/vajax)
