# Repository Map

Generated from: /Users/roberttaylor/Code/ChipFlow/reference/jax-spice
Total symbols: 899

## Documentation Coverage

- **Classes**: 164/170 (96% documented)
- **Functions**: 164/174 (94% documented)
- **Methods**: 376/422 (89% documented)

## ⚠️ Potentially Similar Classes

These classes may have overlapping responsibilities:

- **BenchmarkResults** (benchmarks/test_benchmarks.py)
  ↔ **BenchmarkResult** (scripts/profile_cpu.py)
  Reason: similar names (97%)
  Doc 1: Collects and formats benchmark results.
  Doc 2: Results from a single benchmark run

- **BenchmarkResults** (benchmarks/test_benchmarks.py)
  ↔ **BenchmarkResult** (scripts/profile_gpu.py)
  Reason: similar names (97%)
  Doc 1: Collects and formats benchmark results.
  Doc 2: Results from a single benchmark run.

- **DeviceType** (jax_spice/analysis/mna.py)
  ↔ **Device** (jax_spice/devices/base.py)
  Reason: similar names (75%)
  Doc 1: Enumeration of supported device types for vectorized evaluation
  Doc 2: Base protocol for all device models

- **DeviceInfo** (jax_spice/analysis/mna.py)
  ↔ **Device** (jax_spice/devices/base.py)
  Reason: similar names (75%)
  Doc 1: Information about a device instance for simulation
  Doc 2: Base protocol for all device models

- **CircuitData** (jax_spice/analysis/transient/_mna.py)
  ↔ **Circuit** (jax_spice/netlist/circuit.py)
  Reason: similar names (78%)
  Doc 1: Pre-compiled circuit data for JIT-compatible simulation
  Doc 2: Top-level circuit containing all definitions

- **TransientSetup** (jax_spice/analysis/transient/base.py)
  ↔ **TransientResult** (jax_spice/simulator.py)
  Reason: similar names (76%)
  Doc 1: Cached transient simulation setup data.
  Doc 2: Result of a transient simulation.

- **PythonLoopStrategy** (jax_spice/analysis/transient/python_loop.py)
  ↔ **ScanStrategy** (jax_spice/analysis/transient/scan.py)
  Reason: similar docstrings (69%)
  Doc 1: Transient analysis using Python for-loop with JIT-compiled NR solver.
  Doc 2: Transient analysis using lax.scan for fully JIT-compiled simulation.

- **VerilogADevice** (jax_spice/devices/verilog_a.py)
  ↔ **CompiledVAModel** (tests/test_vacask_sim_parser.py)
  Reason: similar docstrings (66%)
  Doc 1: A device model compiled from Verilog-A using OpenVAF
  Doc 2: A Verilog-A model compiled to JAX via openvaf_jax

- **CompiledDevice** (openvaf-py/openvaf_jax.py)
  ↔ **CompiledModel** (openvaf-py/tests/conftest.py)
  Reason: similar docstrings (73%)
  Doc 1: A compiled Verilog-A device with JAX evaluation function
  Doc 2: Wrapper for a compiled Verilog-A model with JAX function

- **CompiledDevice** (openvaf-py/openvaf_jax.py)
  ↔ **CompiledPDKModel** (openvaf-py/tests_pdk/pdk_utils.py)
  Reason: similar docstrings (71%)
  Doc 1: A compiled Verilog-A device with JAX evaluation function
  Doc 2: Wrapper for a compiled PDK Verilog-A model with JAX function

- **CompiledModel** (openvaf-py/tests/conftest.py)
  ↔ **CompiledPDKModel** (openvaf-py/tests_pdk/pdk_utils.py)
  Reason: similar names (90%), similar docstrings (97%)
  Doc 1: Wrapper for a compiled Verilog-A model with JAX function
  Doc 2: Wrapper for a compiled PDK Verilog-A model with JAX function

- **CompiledModel** (openvaf-py/tests/conftest.py)
  ↔ **CompiledVAModel** (tests/test_vacask_sim_parser.py)
  Reason: similar names (93%)
  Doc 1: Wrapper for a compiled Verilog-A model with JAX function
  Doc 2: A Verilog-A model compiled to JAX via openvaf_jax

- **ParamInfo** (openvaf-py/tests/snap_parser.py)
  ↔ **OsdiParamInfo** (openvaf-py/src/lib.rs)
  Reason: similar names (82%)
  Doc 1: Parameter metadata from snapshot.

- **NodeInfo** (openvaf-py/tests/snap_parser.py)
  ↔ **OsdiNodeInfo** (openvaf-py/src/lib.rs)
  Reason: similar names (80%)
  Doc 1: Node metadata from snapshot.

- **CompiledPDKModel** (openvaf-py/tests_pdk/pdk_utils.py)
  ↔ **CompiledVAModel** (tests/test_vacask_sim_parser.py)
  Reason: similar names (84%)
  Doc 1: Wrapper for a compiled PDK Verilog-A model with JAX function
  Doc 2: A Verilog-A model compiled to JAX via openvaf_jax

- **BenchmarkConfig** (scripts/compare_vacask.py)
  ↔ **BenchmarkSpec** (tests/test_vacask_benchmarks.py)
  Reason: similar docstrings (66%)
  Doc 1: Configuration for a benchmark.
  Doc 2: Specification for a benchmark comparison test.

- **BenchmarkResult** (scripts/profile_cpu.py)
  ↔ **BenchmarkResult** (scripts/profile_gpu.py)
  Reason: similar names (100%), similar docstrings (99%)
  Doc 1: Results from a single benchmark run
  Doc 2: Results from a single benchmark run.

## ⚠️ Potentially Similar Functions

These functions may be duplicates:

- **get_benchmark_sim_file** (benchmarks/test_benchmarks.py:34)
  ↔ **get_benchmark_sim** (tests/test_vacask_benchmarks.py:31)
  Reason: similar names (88%)
  Doc 1: Get the sim file path for a benchmark.
  Doc 2: Get path to benchmark .sim file

- **dc_operating_point** (jax_spice/analysis/dc.py:23)
  ↔ **newton_solve** (jax_spice/analysis/solver.py:56)
  Reason: similar docstrings (66%)
  Doc 1: Find DC operating point using Newton-Raphson iteration.
  Doc 2: Solve nonlinear system using Newton-Raphson iteration.

- **is_gpu_available** (jax_spice/analysis/gpu_backend.py:29)
  ↔ **is_spineax_available** (jax_spice/analysis/spineax_solver.py:36)
  Reason: similar names (75%)
  Doc 1: Check if a GPU backend is available.
  Doc 2: Check if Spineax is available for use.

- **source_stepping** (jax_spice/analysis/homotopy.py:243)
  ↔ **source_stepping_solve** (jax_spice/analysis/solver.py:373)
  Reason: similar names (85%)
  Doc 1: VACASK-style adaptive source stepping with GMIN fallback.
  Doc 2: JIT-compiled source stepping using lax.scan.

- **capacitor** (jax_spice/devices/capacitor.py:128)
  ↔ **capacitor_eval** (tests/test_gpu_backend.py:38)
  Reason: similar names (82%)
  Doc 1: Functional capacitor model for charge calculation
  Doc 2: Capacitor evaluation function.

- **capacitor** (jax_spice/devices/capacitor.py:128)
  ↔ **capacitor_eval** (tests/test_transient.py:34)
  Reason: similar names (82%)
  Doc 1: Functional capacitor model for charge calculation
  Doc 2: Capacitor evaluation function

- **capacitor** (jax_spice/devices/capacitor.py:128)
  ↔ **capacitor_eval** (tests/test_vacask_suite.py:365)
  Reason: similar names (82%)
  Doc 1: Functional capacitor model for charge calculation
  Doc 2: Capacitor evaluation function.

- **mosfet_batch** (jax_spice/devices/mosfet_simple.py:358)
  ↔ **resistor_batch** (jax_spice/devices/resistor.py:124)
  Reason: similar docstrings (87%)
  Doc 1: Vectorized MOSFET evaluation for batch processing.
  Doc 2: Vectorized resistor evaluation for batch processing

- **mosfet_batch** (jax_spice/devices/mosfet_simple.py:358)
  ↔ **vsource_batch** (jax_spice/devices/vsource.py:247)
  Reason: similar docstrings (86%)
  Doc 1: Vectorized MOSFET evaluation for batch processing.
  Doc 2: Vectorized voltage source evaluation for batch processing

- **mosfet_batch** (jax_spice/devices/mosfet_simple.py:358)
  ↔ **isource_batch** (jax_spice/devices/vsource.py:281)
  Reason: similar docstrings (84%)
  Doc 1: Vectorized MOSFET evaluation for batch processing.
  Doc 2: Vectorized current source evaluation for batch processing

- **resistor** (jax_spice/devices/resistor.py:101)
  ↔ **resistor_model** (openvaf-py/tests/conftest.py:148)
  Reason: similar names (76%), similar docstrings (75%)
  Doc 1: Functional resistor model
  Doc 2: Compiled resistor model

- **resistor** (jax_spice/devices/resistor.py:101)
  ↔ **resistor_eval** (tests/test_gpu_backend.py:22)
  Reason: similar names (80%)
  Doc 1: Functional resistor model
  Doc 2: Resistor evaluation function.

- **resistor** (jax_spice/devices/resistor.py:101)
  ↔ **resistor_eval** (tests/test_transient.py:18)
  Reason: similar names (80%)
  Doc 1: Functional resistor model
  Doc 2: Resistor evaluation function

- **resistor** (jax_spice/devices/resistor.py:101)
  ↔ **resistor_eval** (tests/test_vacask_suite.py:236)
  Reason: similar names (80%)
  Doc 1: Functional resistor model
  Doc 2: Resistor evaluation function matching VACASK resistor.va

- **resistor_batch** (jax_spice/devices/resistor.py:124)
  ↔ **vsource_batch** (jax_spice/devices/vsource.py:247)
  Reason: similar docstrings (83%)
  Doc 1: Vectorized resistor evaluation for batch processing
  Doc 2: Vectorized voltage source evaluation for batch processing

- **resistor_batch** (jax_spice/devices/resistor.py:124)
  ↔ **isource_batch** (jax_spice/devices/vsource.py:281)
  Reason: similar docstrings (89%)
  Doc 1: Vectorized resistor evaluation for batch processing
  Doc 2: Vectorized current source evaluation for batch processing

- **compile_va** (jax_spice/devices/verilog_a.py:287)
  ↔ **parse_netlist** (jax_spice/netlist/parser.py:497)
  Reason: similar docstrings (68%)
  Doc 1: Convenience function to compile a Verilog-A file
  Doc 2: Convenience function to parse a VACASK netlist

- **compile_va** (jax_spice/devices/verilog_a.py:287)
  ↔ **compile_va** (openvaf-py/openvaf_jax.py:2458)
  Reason: similar names (100%)
  Doc 1: Convenience function to compile a Verilog-A file
  Doc 2: Compile a Verilog-A file to a JAX-compatible device

- **compile_va** (jax_spice/devices/verilog_a.py:287)
  ↔ **compile_va** (openvaf-py/src/lib.rs:958)
  Reason: similar names (100%)
  Doc 1: Convenience function to compile a Verilog-A file

- **pulse_voltage** (jax_spice/devices/vsource.py:132)
  ↔ **vt** (openvaf-py/tests/test_osdi_evaluation.py:35)
  Reason: similar docstrings (67%)
  Doc 1: Calculate pulse voltage at time t
  Doc 2: Calculate thermal voltage Vt = kT/q.

- **vsource_batch** (jax_spice/devices/vsource.py:247)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar docstrings (66%)
  Doc 1: Vectorized voltage source evaluation for batch processing
  Doc 2: Create a voltage source evaluation function

- **compile_va** (openvaf-py/openvaf_jax.py:2458)
  ↔ **compile_va** (openvaf-py/src/lib.rs:958)
  Reason: similar names (100%)
  Doc 1: Compile a Verilog-A file to a JAX-compatible device

- **compile_model** (openvaf-py/tests/conftest.py:125)
  ↔ **compile_pdk_model_fixture** (openvaf-py/tests_pdk/conftest.py:37)
  Reason: similar docstrings (95%)
  Doc 1: Factory fixture to compile VA models
  Doc 2: Factory fixture to compile PDK VA models

- **compile_model** (openvaf-py/tests/conftest.py:125)
  ↔ **compile_pdk_model** (openvaf-py/tests_pdk/pdk_utils.py:95)
  Reason: similar names (89%)
  Doc 1: Factory fixture to compile VA models
  Doc 2: Compile a PDK VA model with path sanitization

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  Reason: similar docstrings (70%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled HICUM L2 model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  Reason: similar docstrings (71%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled MEXTRAM model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  Reason: similar docstrings (76%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled ASMHEMT model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  Reason: similar docstrings (77%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled HiSIM2 model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  Reason: similar docstrings (76%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled HiSIMHV model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  Reason: similar docstrings (79%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled BSIM3 model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  Reason: similar docstrings (79%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled BSIM4 model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  Reason: similar docstrings (79%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled BSIM6 model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  Reason: similar docstrings (74%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled BSIMBULK model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  Reason: similar docstrings (76%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled BSIMCMG model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  Reason: similar docstrings (84%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled BSIMSOI model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (73%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled PSP102 model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (73%)
  Doc 1: Compiled resistor model
  Doc 2: Compiled PSP103 model

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **resistor_eval** (tests/test_gpu_backend.py:22)
  Reason: similar names (80%)
  Doc 1: Compiled resistor model
  Doc 2: Resistor evaluation function.

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **resistor_eval** (tests/test_transient.py:18)
  Reason: similar names (80%)
  Doc 1: Compiled resistor model
  Doc 2: Resistor evaluation function

- **resistor_model** (openvaf-py/tests/conftest.py:148)
  ↔ **resistor_eval** (tests/test_vacask_suite.py:236)
  Reason: similar names (80%)
  Doc 1: Compiled resistor model
  Doc 2: Resistor evaluation function matching VACASK resistor.va

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  Reason: similar docstrings (74%)
  Doc 1: Compiled diode model
  Doc 2: Compiled HICUM L2 model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  Reason: similar docstrings (76%)
  Doc 1: Compiled diode model
  Doc 2: Compiled MEXTRAM model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  Reason: similar docstrings (76%)
  Doc 1: Compiled diode model
  Doc 2: Compiled ASMHEMT model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  Reason: similar docstrings (78%)
  Doc 1: Compiled diode model
  Doc 2: Compiled HiSIM2 model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  Reason: similar docstrings (76%)
  Doc 1: Compiled diode model
  Doc 2: Compiled HiSIMHV model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  Reason: similar docstrings (80%)
  Doc 1: Compiled diode model
  Doc 2: Compiled BSIM3 model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  Reason: similar docstrings (80%)
  Doc 1: Compiled diode model
  Doc 2: Compiled BSIM4 model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  Reason: similar docstrings (80%)
  Doc 1: Compiled diode model
  Doc 2: Compiled BSIM6 model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  Reason: similar docstrings (74%)
  Doc 1: Compiled diode model
  Doc 2: Compiled BSIMBULK model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  Reason: similar docstrings (76%)
  Doc 1: Compiled diode model
  Doc 2: Compiled BSIMCMG model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  Reason: similar docstrings (81%)
  Doc 1: Compiled diode model
  Doc 2: Compiled BSIMSOI model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (73%)
  Doc 1: Compiled diode model
  Doc 2: Compiled PSP102 model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (73%)
  Doc 1: Compiled diode model
  Doc 2: Compiled PSP103 model

- **diode_model** (openvaf-py/tests/conftest.py:154)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (68%)
  Doc 1: Compiled diode model
  Doc 2: Compiled JUNCAP200 model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  Reason: similar docstrings (77%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled HICUM L2 model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  Reason: similar docstrings (74%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled MEXTRAM model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  Reason: similar docstrings (74%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled ASMHEMT model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  Reason: similar docstrings (71%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled HiSIM2 model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  Reason: similar docstrings (70%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled HiSIMHV model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  Reason: similar docstrings (73%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled BSIM3 model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  Reason: similar docstrings (73%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled BSIM4 model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  Reason: similar docstrings (73%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled BSIM6 model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  Reason: similar docstrings (68%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled BSIMBULK model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  Reason: similar docstrings (74%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled BSIMCMG model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  Reason: similar docstrings (74%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled BSIMSOI model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (67%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled PSP102 model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (67%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled PSP103 model

- **diode_cmc_model** (openvaf-py/tests/conftest.py:160)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (67%)
  Doc 1: Compiled CMC diode model
  Doc 2: Compiled JUNCAP200 model

- **isrc_model** (openvaf-py/tests/conftest.py:166)
  ↔ **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  Reason: similar docstrings (69%)
  Doc 1: Compiled current source model
  Doc 2: Compiled HICUM L2 model

- **isrc_model** (openvaf-py/tests/conftest.py:166)
  ↔ **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  Reason: similar docstrings (67%)
  Doc 1: Compiled current source model
  Doc 2: Compiled ASMHEMT model

- **isrc_model** (openvaf-py/tests/conftest.py:166)
  ↔ **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  Reason: similar docstrings (65%)
  Doc 1: Compiled current source model
  Doc 2: Compiled BSIM3 model

- **isrc_model** (openvaf-py/tests/conftest.py:166)
  ↔ **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  Reason: similar docstrings (65%)
  Doc 1: Compiled current source model
  Doc 2: Compiled BSIM4 model

- **isrc_model** (openvaf-py/tests/conftest.py:166)
  ↔ **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  Reason: similar docstrings (65%)
  Doc 1: Compiled current source model
  Doc 2: Compiled BSIM6 model

- **isrc_model** (openvaf-py/tests/conftest.py:166)
  ↔ **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  Reason: similar docstrings (67%)
  Doc 1: Compiled current source model
  Doc 2: Compiled BSIMSOI model

- **vccs_model** (openvaf-py/tests/conftest.py:172)
  ↔ **mvsg_model** (openvaf-py/tests/test_hemt_models.py:15)
  Reason: similar names (78%)
  Doc 1: Compiled VCCS model
  Doc 2: Compiled MVSG model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  Reason: similar docstrings (76%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled ASMHEMT model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  Reason: similar names (76%), similar docstrings (86%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled HiSIM2 model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  Reason: similar docstrings (80%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled HiSIMHV model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  Reason: similar docstrings (79%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled BSIM3 model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  Reason: similar docstrings (79%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled BSIM4 model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  Reason: similar docstrings (79%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled BSIM6 model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  Reason: similar docstrings (78%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled BSIMBULK model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  Reason: similar docstrings (80%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled BSIMCMG model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  Reason: similar docstrings (76%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled BSIMSOI model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (73%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled PSP102 model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (68%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled PSP103 model

- **hicum_model** (openvaf-py/tests/test_bjt_models.py:9)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (72%)
  Doc 1: Compiled HICUM L2 model
  Doc 2: Compiled JUNCAP200 model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  Reason: similar docstrings (82%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled ASMHEMT model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  Reason: similar docstrings (74%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled HiSIM2 model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  Reason: similar docstrings (73%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled HiSIMHV model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  Reason: similar docstrings (76%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled BSIM3 model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  Reason: similar docstrings (76%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled BSIM4 model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  Reason: similar docstrings (76%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled BSIM6 model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  Reason: similar docstrings (71%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled BSIMBULK model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  Reason: similar docstrings (77%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled BSIMCMG model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  Reason: similar docstrings (73%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled BSIMSOI model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (70%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled PSP102 model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (70%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled PSP103 model

- **mextram_model** (openvaf-py/tests/test_bjt_models.py:15)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (70%)
  Doc 1: Compiled MEXTRAM model
  Doc 2: Compiled JUNCAP200 model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  Reason: similar docstrings (79%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled HiSIM2 model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  Reason: similar docstrings (82%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled HiSIMHV model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  Reason: similar docstrings (81%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled BSIM3 model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  Reason: similar docstrings (81%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled BSIM4 model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  Reason: similar docstrings (81%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled BSIM6 model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  Reason: similar docstrings (76%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled BSIMBULK model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  Reason: similar docstrings (82%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled BSIMCMG model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  Reason: similar docstrings (77%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled BSIMSOI model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (74%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled PSP102 model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (74%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled PSP103 model

- **asmhemt_model** (openvaf-py/tests/test_hemt_models.py:9)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (70%)
  Doc 1: Compiled ASMHEMT model
  Doc 2: Compiled JUNCAP200 model

- **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  ↔ **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  Reason: similar names (76%), similar docstrings (88%)
  Doc 1: Compiled HiSIM2 model
  Doc 2: Compiled BSIM3 model

- **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  ↔ **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  Reason: similar names (76%), similar docstrings (88%)
  Doc 1: Compiled HiSIM2 model
  Doc 2: Compiled BSIM4 model

- **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  ↔ **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  Reason: similar names (76%), similar docstrings (88%)
  Doc 1: Compiled HiSIM2 model
  Doc 2: Compiled BSIM6 model

- **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  ↔ **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  Reason: similar docstrings (82%)
  Doc 1: Compiled HiSIM2 model
  Doc 2: Compiled BSIMBULK model

- **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  ↔ **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  Reason: similar docstrings (84%)
  Doc 1: Compiled HiSIM2 model
  Doc 2: Compiled BSIMCMG model

- **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  ↔ **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  Reason: similar docstrings (84%)
  Doc 1: Compiled HiSIM2 model
  Doc 2: Compiled BSIMSOI model

- **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (81%)
  Doc 1: Compiled HiSIM2 model
  Doc 2: Compiled PSP102 model

- **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (76%)
  Doc 1: Compiled HiSIM2 model
  Doc 2: Compiled PSP103 model

- **hisim2_model** (openvaf-py/tests/test_hisim_models.py:9)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (71%)
  Doc 1: Compiled HiSIM2 model
  Doc 2: Compiled JUNCAP200 model

- **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  ↔ **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  Reason: similar docstrings (86%)
  Doc 1: Compiled HiSIMHV model
  Doc 2: Compiled BSIM3 model

- **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  ↔ **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  Reason: similar docstrings (86%)
  Doc 1: Compiled HiSIMHV model
  Doc 2: Compiled BSIM4 model

- **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  ↔ **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  Reason: similar docstrings (86%)
  Doc 1: Compiled HiSIMHV model
  Doc 2: Compiled BSIM6 model

- **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  ↔ **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  Reason: similar docstrings (80%)
  Doc 1: Compiled HiSIMHV model
  Doc 2: Compiled BSIMBULK model

- **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  ↔ **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  Reason: similar docstrings (82%)
  Doc 1: Compiled HiSIMHV model
  Doc 2: Compiled BSIMCMG model

- **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  ↔ **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  Reason: similar docstrings (82%)
  Doc 1: Compiled HiSIMHV model
  Doc 2: Compiled BSIMSOI model

- **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (74%)
  Doc 1: Compiled HiSIMHV model
  Doc 2: Compiled PSP102 model

- **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (74%)
  Doc 1: Compiled HiSIMHV model
  Doc 2: Compiled PSP103 model

- **hisimhv_model** (openvaf-py/tests/test_hisim_models.py:15)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (65%)
  Doc 1: Compiled HiSIMHV model
  Doc 2: Compiled JUNCAP200 model

- **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (78%)
  Doc 1: Compiled BSIM3 model
  Doc 2: Compiled PSP102 model

- **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (83%)
  Doc 1: Compiled BSIM3 model
  Doc 2: Compiled PSP103 model

- **bsim3_model** (openvaf-py/tests/test_mosfet_models.py:15)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (68%)
  Doc 1: Compiled BSIM3 model
  Doc 2: Compiled JUNCAP200 model

- **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (78%)
  Doc 1: Compiled BSIM4 model
  Doc 2: Compiled PSP102 model

- **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (78%)
  Doc 1: Compiled BSIM4 model
  Doc 2: Compiled PSP103 model

- **bsim4_model** (openvaf-py/tests/test_mosfet_models.py:21)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (68%)
  Doc 1: Compiled BSIM4 model
  Doc 2: Compiled JUNCAP200 model

- **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (78%)
  Doc 1: Compiled BSIM6 model
  Doc 2: Compiled PSP102 model

- **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (78%)
  Doc 1: Compiled BSIM6 model
  Doc 2: Compiled PSP103 model

- **bsim6_model** (openvaf-py/tests/test_mosfet_models.py:27)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (68%)
  Doc 1: Compiled BSIM6 model
  Doc 2: Compiled JUNCAP200 model

- **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (73%)
  Doc 1: Compiled BSIMBULK model
  Doc 2: Compiled PSP102 model

- **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (73%)
  Doc 1: Compiled BSIMBULK model
  Doc 2: Compiled PSP103 model

- **bsimbulk_model** (openvaf-py/tests/test_mosfet_models.py:33)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (68%)
  Doc 1: Compiled BSIMBULK model
  Doc 2: Compiled JUNCAP200 model

- **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (74%)
  Doc 1: Compiled BSIMCMG model
  Doc 2: Compiled PSP102 model

- **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (74%)
  Doc 1: Compiled BSIMCMG model
  Doc 2: Compiled PSP103 model

- **bsimcmg_model** (openvaf-py/tests/test_mosfet_models.py:39)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (70%)
  Doc 1: Compiled BSIMCMG model
  Doc 2: Compiled JUNCAP200 model

- **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  ↔ **psp102_model** (openvaf-py/tests/test_psp_models.py:9)
  Reason: similar docstrings (74%)
  Doc 1: Compiled BSIMSOI model
  Doc 2: Compiled PSP102 model

- **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  ↔ **psp103_model** (openvaf-py/tests/test_psp_models.py:15)
  Reason: similar docstrings (74%)
  Doc 1: Compiled BSIMSOI model
  Doc 2: Compiled PSP103 model

- **bsimsoi_model** (openvaf-py/tests/test_mosfet_models.py:45)
  ↔ **juncap_model** (openvaf-py/tests/test_psp_models.py:21)
  Reason: similar docstrings (65%)
  Doc 1: Compiled BSIMSOI model
  Doc 2: Compiled JUNCAP200 model

- **get_module** (openvaf-py/tests/test_osdi_evaluation.py:27)
  ↔ **get_osdi_descriptor** (openvaf-py/tests/test_osdi_metadata.py:33)
  Reason: similar docstrings (78%)
  Doc 1: Compile a Verilog-A file and return the module.
  Doc 2: Compile a Verilog-A file and return its OSDI descriptor.

- **pytest_configure** (openvaf-py/tests_pdk/conftest.py:17)
  ↔ **pytest_configure** (tests/conftest.py:47)
  Reason: similar names (100%)
  Doc 1: Register custom markers
  Doc 2: Pytest hook that runs before test collection.

- **compile_pdk_model_fixture** (openvaf-py/tests_pdk/conftest.py:37)
  ↔ **compile_pdk_model** (openvaf-py/tests_pdk/pdk_utils.py:95)
  Reason: similar names (81%)
  Doc 1: Factory fixture to compile PDK VA models
  Doc 2: Compile a PDK VA model with path sanitization

- **parse_value_with_suffix** (scripts/compare_results.py:20)
  ↔ **parse_si_value** (tests/test_vacask_suite.py:515)
  Reason: similar docstrings (70%)
  Doc 1: Parse a numeric value that may have an SI suffix (e.g., '1u' -> 1e-6).
  Doc 2: Parse a value with SI suffix (e.g., '2k' -> 2000).

- **run_jaxspice** (scripts/compare_results.py:114)
  ↔ **run_jax_spice** (scripts/compare_vacask.py:219)
  Reason: similar names (100%), similar docstrings (67%)
  Doc 1: Run JAX-SPICE and return final voltages.
  Doc 2: Run JAX-SPICE and return (time_per_step_ms, wall_time_s, stats).

- **find_vacask_binary** (scripts/compare_vacask.py:123)
  ↔ **find_vacask_binary** (tests/test_vacask_benchmarks.py:684)
  Reason: similar names (100%)
  Doc 1: Find the VACASK binary (returns absolute path).
  Doc 2: Find VACASK simulator binary.

- **main** (scripts/compare_vacask.py:303)
  ↔ **main** (scripts/nsys_profile_target.py:36)
  Reason: similar names (100%)

- **main** (scripts/compare_vacask.py:303)
  ↔ **main** (scripts/profile_cpu.py:166)
  Reason: similar names (100%)

- **main** (scripts/compare_vacask.py:303)
  ↔ **main** (scripts/profile_gpu.py:452)
  Reason: similar names (100%)

- **main** (scripts/compare_vacask.py:303)
  ↔ **main** (scripts/profile_gpu_cloudrun.py:61)
  Reason: similar names (100%)

- **main** (scripts/compare_vacask.py:303)
  ↔ **main** (scripts/profile_nsys_cloudrun.py:64)
  Reason: similar names (100%)

- **main** (scripts/compare_vacask.py:303)
  ↔ **main** (scripts/run_gpu_tests.py:39)
  Reason: similar names (100%)

- **main** (scripts/compare_vacask.py:303)
  ↔ **main** (scripts/view_traces.py:244)
  Reason: similar names (100%)

- **main** (scripts/nsys_profile_target.py:36)
  ↔ **main** (scripts/profile_cpu.py:166)
  Reason: similar names (100%)

- **main** (scripts/nsys_profile_target.py:36)
  ↔ **main** (scripts/profile_gpu.py:452)
  Reason: similar names (100%)

- **main** (scripts/nsys_profile_target.py:36)
  ↔ **main** (scripts/profile_gpu_cloudrun.py:61)
  Reason: similar names (100%)

- **main** (scripts/nsys_profile_target.py:36)
  ↔ **main** (scripts/profile_nsys_cloudrun.py:64)
  Reason: similar names (100%)

- **main** (scripts/nsys_profile_target.py:36)
  ↔ **main** (scripts/run_gpu_tests.py:39)
  Reason: similar names (100%)

- **main** (scripts/nsys_profile_target.py:36)
  ↔ **main** (scripts/view_traces.py:244)
  Reason: similar names (100%)

- **get_vacask_benchmarks** (scripts/profile_cpu.py:64)
  ↔ **get_vacask_benchmarks** (scripts/profile_gpu.py:333)
  Reason: similar names (100%), similar docstrings (100%)
  Doc 1: Get list of VACASK benchmark .sim files
  Doc 2: Get list of VACASK benchmark .sim files

- **get_vacask_benchmarks** (scripts/profile_cpu.py:64)
  ↔ **get_benchmark_sim** (tests/test_vacask_benchmarks.py:31)
  Reason: similar names (76%), similar docstrings (77%)
  Doc 1: Get list of VACASK benchmark .sim files
  Doc 2: Get path to benchmark .sim file

- **run_benchmark** (scripts/profile_cpu.py:81)
  ↔ **run_single_benchmark** (scripts/profile_gpu.py:350)
  Reason: similar names (80%)
  Doc 1: Run a single benchmark configuration.
  Doc 2: Run a single benchmark in subprocess mode and output JSON result.

- **main** (scripts/profile_cpu.py:166)
  ↔ **main** (scripts/profile_gpu.py:452)
  Reason: similar names (100%)

- **main** (scripts/profile_cpu.py:166)
  ↔ **main** (scripts/profile_gpu_cloudrun.py:61)
  Reason: similar names (100%)

- **main** (scripts/profile_cpu.py:166)
  ↔ **main** (scripts/profile_nsys_cloudrun.py:64)
  Reason: similar names (100%)

- **main** (scripts/profile_cpu.py:166)
  ↔ **main** (scripts/run_gpu_tests.py:39)
  Reason: similar names (100%)

- **main** (scripts/profile_cpu.py:166)
  ↔ **main** (scripts/view_traces.py:244)
  Reason: similar names (100%)

- **get_vacask_benchmarks** (scripts/profile_gpu.py:333)
  ↔ **get_benchmark_sim** (tests/test_vacask_benchmarks.py:31)
  Reason: similar names (76%), similar docstrings (77%)
  Doc 1: Get list of VACASK benchmark .sim files
  Doc 2: Get path to benchmark .sim file

- **main** (scripts/profile_gpu.py:452)
  ↔ **main** (scripts/profile_gpu_cloudrun.py:61)
  Reason: similar names (100%)

- **main** (scripts/profile_gpu.py:452)
  ↔ **main** (scripts/profile_nsys_cloudrun.py:64)
  Reason: similar names (100%)

- **main** (scripts/profile_gpu.py:452)
  ↔ **main** (scripts/run_gpu_tests.py:39)
  Reason: similar names (100%)

- **main** (scripts/profile_gpu.py:452)
  ↔ **main** (scripts/view_traces.py:244)
  Reason: similar names (100%)

- **run_cmd** (scripts/profile_gpu_cloudrun.py:46)
  ↔ **run_cmd** (scripts/profile_nsys_cloudrun.py:45)
  Reason: similar names (100%), similar docstrings (100%)
  Doc 1: Run a command and optionally capture output.
  Doc 2: Run a command and optionally capture output.

- **run_cmd** (scripts/profile_gpu_cloudrun.py:46)
  ↔ **run_command** (scripts/run_gpu_tests.py:26)
  Reason: similar names (75%), similar docstrings (65%)
  Doc 1: Run a command and optionally capture output.
  Doc 2: Run a command and return the result.

- **main** (scripts/profile_gpu_cloudrun.py:61)
  ↔ **main** (scripts/profile_nsys_cloudrun.py:64)
  Reason: similar names (100%)

- **main** (scripts/profile_gpu_cloudrun.py:61)
  ↔ **main** (scripts/run_gpu_tests.py:39)
  Reason: similar names (100%)

- **main** (scripts/profile_gpu_cloudrun.py:61)
  ↔ **main** (scripts/view_traces.py:244)
  Reason: similar names (100%)

- **run_cmd** (scripts/profile_nsys_cloudrun.py:45)
  ↔ **run_command** (scripts/run_gpu_tests.py:26)
  Reason: similar names (75%), similar docstrings (65%)
  Doc 1: Run a command and optionally capture output.
  Doc 2: Run a command and return the result.

- **main** (scripts/profile_nsys_cloudrun.py:64)
  ↔ **main** (scripts/run_gpu_tests.py:39)
  Reason: similar names (100%)

- **main** (scripts/profile_nsys_cloudrun.py:64)
  ↔ **main** (scripts/view_traces.py:244)
  Reason: similar names (100%)

- **main** (scripts/run_gpu_tests.py:39)
  ↔ **main** (scripts/view_traces.py:244)
  Reason: similar names (100%)

- **list_trace_files** (scripts/view_traces.py:41)
  ↔ **discover_sim_files** (tests/test_vacask_suite.py:52)
  Reason: similar docstrings (70%)
  Doc 1: List all trace files in a directory.
  Doc 2: Find all .sim test files in VACASK test directory.

- **resistor_eval** (tests/test_gpu_backend.py:22)
  ↔ **resistor_eval** (tests/test_transient.py:18)
  Reason: similar names (100%), similar docstrings (98%)
  Doc 1: Resistor evaluation function.
  Doc 2: Resistor evaluation function

- **resistor_eval** (tests/test_gpu_backend.py:22)
  ↔ **capacitor_eval** (tests/test_transient.py:34)
  Reason: similar docstrings (83%)
  Doc 1: Resistor evaluation function.
  Doc 2: Capacitor evaluation function

- **resistor_eval** (tests/test_gpu_backend.py:22)
  ↔ **resistor_eval** (tests/test_vacask_suite.py:236)
  Reason: similar names (100%), similar docstrings (68%)
  Doc 1: Resistor evaluation function.
  Doc 2: Resistor evaluation function matching VACASK resistor.va

- **resistor_eval** (tests/test_gpu_backend.py:22)
  ↔ **isource_eval** (tests/test_vacask_suite.py:286)
  Reason: similar docstrings (81%)
  Doc 1: Resistor evaluation function.
  Doc 2: Current source evaluation function.

- **resistor_eval** (tests/test_gpu_backend.py:22)
  ↔ **inductor_eval** (tests/test_vacask_suite.py:343)
  Reason: similar docstrings (86%)
  Doc 1: Resistor evaluation function.
  Doc 2: Inductor evaluation function.

- **resistor_eval** (tests/test_gpu_backend.py:22)
  ↔ **capacitor_eval** (tests/test_vacask_suite.py:365)
  Reason: similar docstrings (85%)
  Doc 1: Resistor evaluation function.
  Doc 2: Capacitor evaluation function.

- **resistor_eval** (tests/test_gpu_backend.py:22)
  ↔ **make_resistor_eval** (tests/test_vectorized_mna.py:25)
  Reason: similar names (86%), similar docstrings (85%)
  Doc 1: Resistor evaluation function.
  Doc 2: Create a resistor evaluation function

- **resistor_eval** (tests/test_gpu_backend.py:22)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar docstrings (69%)
  Doc 1: Resistor evaluation function.
  Doc 2: Create a voltage source evaluation function

- **capacitor_eval** (tests/test_gpu_backend.py:38)
  ↔ **resistor_eval** (tests/test_transient.py:18)
  Reason: similar docstrings (83%)
  Doc 1: Capacitor evaluation function.
  Doc 2: Resistor evaluation function

- **capacitor_eval** (tests/test_gpu_backend.py:38)
  ↔ **capacitor_eval** (tests/test_transient.py:34)
  Reason: similar names (100%), similar docstrings (98%)
  Doc 1: Capacitor evaluation function.
  Doc 2: Capacitor evaluation function

- **capacitor_eval** (tests/test_gpu_backend.py:38)
  ↔ **isource_eval** (tests/test_vacask_suite.py:286)
  Reason: similar docstrings (71%)
  Doc 1: Capacitor evaluation function.
  Doc 2: Current source evaluation function.

- **capacitor_eval** (tests/test_gpu_backend.py:38)
  ↔ **inductor_eval** (tests/test_vacask_suite.py:343)
  Reason: similar docstrings (85%)
  Doc 1: Capacitor evaluation function.
  Doc 2: Inductor evaluation function.

- **capacitor_eval** (tests/test_gpu_backend.py:38)
  ↔ **capacitor_eval** (tests/test_vacask_suite.py:365)
  Reason: similar names (100%), similar docstrings (100%)
  Doc 1: Capacitor evaluation function.
  Doc 2: Capacitor evaluation function.

- **capacitor_eval** (tests/test_gpu_backend.py:38)
  ↔ **make_resistor_eval** (tests/test_vectorized_mna.py:25)
  Reason: similar docstrings (81%)
  Doc 1: Capacitor evaluation function.
  Doc 2: Create a resistor evaluation function

- **capacitor_eval** (tests/test_gpu_backend.py:38)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar docstrings (66%)
  Doc 1: Capacitor evaluation function.
  Doc 2: Create a voltage source evaluation function

- **vsource_eval** (tests/test_gpu_backend.py:60)
  ↔ **vsource_eval** (tests/test_transient.py:56)
  Reason: similar names (100%), similar docstrings (91%)
  Doc 1: Voltage source evaluation function (DC only).
  Doc 2: Voltage source evaluation function (DC only for now)

- **vsource_eval** (tests/test_gpu_backend.py:60)
  ↔ **vsource_eval** (tests/test_vacask_suite.py:256)
  Reason: similar names (100%), similar docstrings (70%)
  Doc 1: Voltage source evaluation function (DC only).
  Doc 2: Voltage source evaluation function using large conductance method.

- **vsource_eval** (tests/test_gpu_backend.py:60)
  ↔ **isource_eval** (tests/test_vacask_suite.py:286)
  Reason: similar names (91%), similar docstrings (72%)
  Doc 1: Voltage source evaluation function (DC only).
  Doc 2: Current source evaluation function.

- **vsource_eval** (tests/test_gpu_backend.py:60)
  ↔ **pulse_vsource_eval** (tests/test_vacask_suite.py:299)
  Reason: similar names (81%)
  Doc 1: Voltage source evaluation function (DC only).
  Doc 2: Pulse voltage source for transient analysis.

- **vsource_eval** (tests/test_gpu_backend.py:60)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar names (85%), similar docstrings (77%)
  Doc 1: Voltage source evaluation function (DC only).
  Doc 2: Create a voltage source evaluation function

- **resistor_eval** (tests/test_transient.py:18)
  ↔ **resistor_eval** (tests/test_vacask_suite.py:236)
  Reason: similar names (100%), similar docstrings (67%)
  Doc 1: Resistor evaluation function
  Doc 2: Resistor evaluation function matching VACASK resistor.va

- **resistor_eval** (tests/test_transient.py:18)
  ↔ **isource_eval** (tests/test_vacask_suite.py:286)
  Reason: similar docstrings (79%)
  Doc 1: Resistor evaluation function
  Doc 2: Current source evaluation function.

- **resistor_eval** (tests/test_transient.py:18)
  ↔ **inductor_eval** (tests/test_vacask_suite.py:343)
  Reason: similar docstrings (84%)
  Doc 1: Resistor evaluation function
  Doc 2: Inductor evaluation function.

- **resistor_eval** (tests/test_transient.py:18)
  ↔ **capacitor_eval** (tests/test_vacask_suite.py:365)
  Reason: similar docstrings (83%)
  Doc 1: Resistor evaluation function
  Doc 2: Capacitor evaluation function.

- **resistor_eval** (tests/test_transient.py:18)
  ↔ **make_resistor_eval** (tests/test_vectorized_mna.py:25)
  Reason: similar names (86%), similar docstrings (86%)
  Doc 1: Resistor evaluation function
  Doc 2: Create a resistor evaluation function

- **resistor_eval** (tests/test_transient.py:18)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar docstrings (70%)
  Doc 1: Resistor evaluation function
  Doc 2: Create a voltage source evaluation function

- **capacitor_eval** (tests/test_transient.py:34)
  ↔ **isource_eval** (tests/test_vacask_suite.py:286)
  Reason: similar docstrings (69%)
  Doc 1: Capacitor evaluation function
  Doc 2: Current source evaluation function.

- **capacitor_eval** (tests/test_transient.py:34)
  ↔ **inductor_eval** (tests/test_vacask_suite.py:343)
  Reason: similar docstrings (83%)
  Doc 1: Capacitor evaluation function
  Doc 2: Inductor evaluation function.

- **capacitor_eval** (tests/test_transient.py:34)
  ↔ **capacitor_eval** (tests/test_vacask_suite.py:365)
  Reason: similar names (100%), similar docstrings (98%)
  Doc 1: Capacitor evaluation function
  Doc 2: Capacitor evaluation function.

- **capacitor_eval** (tests/test_transient.py:34)
  ↔ **make_resistor_eval** (tests/test_vectorized_mna.py:25)
  Reason: similar docstrings (82%)
  Doc 1: Capacitor evaluation function
  Doc 2: Create a resistor evaluation function

- **capacitor_eval** (tests/test_transient.py:34)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar docstrings (67%)
  Doc 1: Capacitor evaluation function
  Doc 2: Create a voltage source evaluation function

- **vsource_eval** (tests/test_transient.py:56)
  ↔ **vsource_eval** (tests/test_vacask_suite.py:256)
  Reason: similar names (100%), similar docstrings (68%)
  Doc 1: Voltage source evaluation function (DC only for now)
  Doc 2: Voltage source evaluation function using large conductance method.

- **vsource_eval** (tests/test_transient.py:56)
  ↔ **isource_eval** (tests/test_vacask_suite.py:286)
  Reason: similar names (91%)
  Doc 1: Voltage source evaluation function (DC only for now)
  Doc 2: Current source evaluation function.

- **vsource_eval** (tests/test_transient.py:56)
  ↔ **pulse_vsource_eval** (tests/test_vacask_suite.py:299)
  Reason: similar names (81%)
  Doc 1: Voltage source evaluation function (DC only for now)
  Doc 2: Pulse voltage source for transient analysis.

- **vsource_eval** (tests/test_transient.py:56)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar names (85%), similar docstrings (72%)
  Doc 1: Voltage source evaluation function (DC only for now)
  Doc 2: Create a voltage source evaluation function

- **parse_embedded_python** (tests/test_vacask_jax.py:33)
  ↔ **parse_embedded_python** (tests/test_vacask_suite.py:59)
  Reason: similar names (100%), similar docstrings (100%)
  Doc 1: Extract expected values from embedded Python test script.
  Doc 2: Extract expected values from embedded Python test script.

- **parse_si_value** (tests/test_vacask_jax.py:70)
  ↔ **parse_si_value** (tests/test_vacask_suite.py:515)
  Reason: similar names (100%)
  Doc 1: Parse SI-suffixed value like '2k' -> 2000.0
  Doc 2: Parse a value with SI suffix (e.g., '2k' -> 2000).

- **resistor_eval** (tests/test_vacask_suite.py:236)
  ↔ **make_resistor_eval** (tests/test_vectorized_mna.py:25)
  Reason: similar names (86%)
  Doc 1: Resistor evaluation function matching VACASK resistor.va
  Doc 2: Create a resistor evaluation function

- **vsource_eval** (tests/test_vacask_suite.py:256)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar names (85%)
  Doc 1: Voltage source evaluation function using large conductance method.
  Doc 2: Create a voltage source evaluation function

- **isource_eval** (tests/test_vacask_suite.py:286)
  ↔ **make_resistor_eval** (tests/test_vectorized_mna.py:25)
  Reason: similar docstrings (78%)
  Doc 1: Current source evaluation function.
  Doc 2: Create a resistor evaluation function

- **isource_eval** (tests/test_vacask_suite.py:286)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar names (77%), similar docstrings (79%)
  Doc 1: Current source evaluation function.
  Doc 2: Create a voltage source evaluation function

- **pulse_vsource_eval** (tests/test_vacask_suite.py:299)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar names (77%)
  Doc 1: Pulse voltage source for transient analysis.
  Doc 2: Create a voltage source evaluation function

- **inductor_eval** (tests/test_vacask_suite.py:343)
  ↔ **make_resistor_eval** (tests/test_vectorized_mna.py:25)
  Reason: similar docstrings (73%)
  Doc 1: Inductor evaluation function.
  Doc 2: Create a resistor evaluation function

- **capacitor_eval** (tests/test_vacask_suite.py:365)
  ↔ **make_resistor_eval** (tests/test_vectorized_mna.py:25)
  Reason: similar docstrings (81%)
  Doc 1: Capacitor evaluation function.
  Doc 2: Create a resistor evaluation function

- **capacitor_eval** (tests/test_vacask_suite.py:365)
  ↔ **make_vsource_eval** (tests/test_vectorized_mna.py:44)
  Reason: similar docstrings (66%)
  Doc 1: Capacitor evaluation function.
  Doc 2: Create a voltage source evaluation function

## 📝 Documentation Opportunities

Adding docstrings helps both humans and AI understand your code:

**Undocumented classes:**
- Token (jax_spice/netlist/parser.py:18)
- OsdiParamInfo (openvaf-py/src/lib.rs:30)
- OsdiNodeInfo (openvaf-py/src/lib.rs:41)
- OsdiJacobianInfo (openvaf-py/src/lib.rs:50)
- OsdiNoiseInfo (openvaf-py/src/lib.rs:58)
- VaModule (openvaf-py/src/lib.rs:66)

**Undocumented functions:**
- main (scripts/compare_vacask.py:303)
- main (scripts/nsys_profile_target.py:36)
- main (scripts/profile_cpu.py:166)
- main (scripts/profile_gpu.py:452)
- main (scripts/profile_gpu_cloudrun.py:61)
- main (scripts/profile_nsys_cloudrun.py:64)
- main (scripts/run_gpu_tests.py:39)
- main (scripts/view_traces.py:244)
- compile_va (openvaf-py/src/lib.rs:958)
- openvaf_py (openvaf-py/src/lib.rs:1391)

## Code Structure

### benchmarks/test_benchmarks.py

**class BenchmarkResults**
  Collects and formats benchmark results.
  - add(self, name: str, nodes: int, devices: int, openvaf: int, timesteps: int, time_s: float, solver: str, passed: bool, error: str) ❌
  - to_markdown(self) -> str
      Generate markdown report.

**class TestVACASKBenchmarks**
  Test all VACASK benchmark circuits.
  - test_benchmark(self, benchmark_name, benchmark_results)
      Run a VACASK benchmark and verify correctness.

**get_benchmark_sim_file(name: str) -> Path**
  Get the sim file path for a benchmark.

**write_github_summary(content: str)**
  Write content to GitHub step summary if available.

**benchmark_results()**
  Fixture to collect and report benchmark results.

### jax_spice/analysis/context.py

**class AnalysisContext**
  Context passed to device evaluations during analysis
  - is_dc(self) -> bool
      Check if this is a DC analysis
  - is_transient(self) -> bool
      Check if this is a transient analysis

### jax_spice/analysis/dc.py

**dc_operating_point(system: MNASystem, initial_guess: Optional[Array], max_iterations: int, abstol: float, reltol: float, damping: float, vdd: float, init_supplies: bool, backend: Optional[str]) -> Tuple[Array, Dict]**
  Find DC operating point using Newton-Raphson iteration.

### jax_spice/analysis/gpu_backend.py

**class BackendConfig**
  Configuration for simulation backend selection.

**is_gpu_available() -> bool**
  Check if a GPU backend is available.

**get_gpu_devices() -> list**
  Get list of available GPU devices.

**select_backend(num_nodes: int, config: Optional[BackendConfig]) -> str**
  Select optimal backend based on circuit size and availability.

**get_device(backend: str) -> jax.Device**
  Get JAX device for the selected backend.

**get_default_dtype(backend: str)**
  Get default dtype for the selected backend.

**backend_info() -> dict**
  Get information about available backends.

### jax_spice/analysis/homotopy.py

**class HomotopyConfig**
  Configuration for homotopy algorithms.

**class HomotopyResult**
  Result from a homotopy algorithm.

**gmin_stepping(build_residual_fn: Callable[[float, float], Callable], build_jacobian_fn: Callable[[float, float], Callable], V_init: Array, config: HomotopyConfig, nr_config: NRConfig, mode: str) -> HomotopyResult**
  VACASK-style adaptive GMIN stepping.

**source_stepping(build_residual_fn: Callable[[float, float, float], Callable], build_jacobian_fn: Callable[[float, float, float], Callable], V_init: Array, config: HomotopyConfig, nr_config: NRConfig) -> HomotopyResult**
  VACASK-style adaptive source stepping with GMIN fallback.

**run_homotopy_chain(build_residual_fn: Callable, build_jacobian_fn: Callable, V_init: Array, config: HomotopyConfig, nr_config: NRConfig) -> HomotopyResult**
  Run VACASK-style homotopy chain: gdev -> gshunt -> src.

### jax_spice/analysis/mna.py

**class DeviceType**
  Enumeration of supported device types for vectorized evaluation

**class VectorizedDeviceGroup**
  Group of devices of the same type for vectorized evaluation
  - n_terminals(self) -> int
      Number of terminals per device

**class DeviceInfo**
  Information about a device instance for simulation

**class MNASystem**
  MNA system for circuit simulation
  - from_circuit(cls, circuit: Circuit, top_subckt: str, model_registry: Dict[str, Tuple[Any, Dict]]) -> 'MNASystem'
      Create MNA system from parsed circuit
  - build_jacobian_and_residual(self, voltages: Array, context: AnalysisContext) -> Tuple[Array, Array]
      Build Jacobian matrix and residual vector
  - get_node_voltage(self, solution: Array, node_name: str) -> float
      Get voltage at a named node from solution vector
  - full_voltage_vector(self, solution: Array) -> Array
      Convert solution to full voltage vector including ground
  - build_sparse_jacobian_and_residual(self, voltages: Array, context: AnalysisContext) -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike, Tuple[int, int]], ArrayLike]
      Build Jacobian matrix and residual vector in sparse CSR format
  - build_vectorized_jacobian_and_residual(self, voltages: Array, context: AnalysisContext) -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike, Tuple[int, int]], ArrayLike]
      Build Jacobian matrix and residual using vectorized device evaluation
  - build_device_groups(self, vdd: float) -> None
      Build vectorized device groups from the devices list
  - build_gpu_residual_fn(self, vdd: float, gmin: float) -> Callable[[Array], Array]
      Build pure JAX residual function for GPU execution.
  - build_parameterized_residual_fn(self, gmin: float) -> Callable[[Array, float], Array]
      Build parameterized residual function for source stepping.
  - build_gpu_system_fns(self, vdd: float, gmin: float) -> Tuple[Callable[[Array], Array], Callable[[Array], Array]]
      Build both residual and Jacobian functions for GPU execution.
  - build_transient_residual_fn(self, gmin: float) -> Callable[[Array, Array, float], Array]
      Build vectorized transient residual function for GPU execution.
  - build_sparsity_pattern(self) -> Tuple[Array, Array]
      Build sparsity pattern (row, col indices) for Jacobian.

**eval_param_simple(value, vdd: float, defaults: dict)**
  Simple parameter evaluation for common cases.

### jax_spice/analysis/mna_gpu.py

**stamp_2terminal_residual_gpu(residual: Array, node_p: Array, node_n: Array, I_batch: Array, ground_node: int) -> Array**
  Stamp 2-terminal device currents into residual using GPU scatter.

**stamp_4terminal_residual_gpu(residual: Array, node_d: Array, node_g: Array, node_s: Array, node_b: Array, Ids: Array, ground_node: int) -> Array**
  Stamp 4-terminal MOSFET currents into residual using GPU scatter.

**stamp_vsource_residual_gpu(residual: Array, node_p: Array, node_n: Array, V_batch: Array, I_branch: Array, branch_indices: Array, V_nodes: Array, vdd_scale: float, ground_node: int) -> Array**
  Stamp voltage source contributions to residual.

**stamp_gmin_residual_gpu(residual: Array, V: Array, gmin: float, ground_node: int) -> Array**
  Add GMIN contribution to all nodes (except ground).

**stamp_gshunt_residual_gpu(residual: Array, V: Array, gshunt: float, ground_node: int) -> Array**
  Add GSHUNT contribution to residual for all voltage nodes.

**stamp_gshunt_jacobian_gpu(jacobian: Array, gshunt: float, num_nodes: int, ground_node: int) -> Array**
  Add GSHUNT contribution to Jacobian diagonals.

**build_mosfet_params_from_group(group, temperature: float) -> Dict[str, Array]**
  Build parameter dictionary for mosfet_batch from VectorizedDeviceGroup.

**build_resistor_params_from_group(group) -> Tuple[Array, Array, Array]**
  Build parameter arrays for resistor_batch from VectorizedDeviceGroup.

### jax_spice/analysis/solver.py

**class NRConfig**
  Configuration for Newton-Raphson solver.

**class NRResult**
  Result from Newton-Raphson solver.

**newton_solve(residual_fn: Callable[[jax.Array], jax.Array], jacobian_fn: Callable[[jax.Array], jax.Array], V_init: jax.Array, config: NRConfig | None) -> NRResult**
  Solve nonlinear system using Newton-Raphson iteration.

**newton_solve_with_system(build_system_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array]], V_init: jax.Array, config: NRConfig | None) -> NRResult**
  Solve using a combined system builder function.

**newton_solve_parameterized(residual_fn: Callable[[jax.Array, float], jax.Array], V_init: jax.Array, param: float, config: NRConfig | None) -> NRResult**
  Newton-Raphson solver for parameterized residual functions.

**source_stepping_solve(residual_fn: Callable[[jax.Array, float], jax.Array], V_init: jax.Array, vdd_steps: jax.Array, config: NRConfig | None) -> Tuple[jax.Array, jax.Array, jax.Array]**
  JIT-compiled source stepping using lax.scan.

### jax_spice/analysis/sparse.py

**sparse_solve_csr(data: Array, indices: Array, indptr: Array, b: Array, shape: Tuple[int, int]) -> Array**
  Solve sparse linear system Ax = b with CSR format matrix

**build_csr_arrays(rows: ArrayLike, cols: ArrayLike, values: ArrayLike, shape: Tuple[int, int]) -> Tuple[Array, Array, Array]**
  Convert COO triplets to CSR format arrays using pure JAX.

**build_csc_arrays(rows: Array, cols: Array, values: Array, shape: Tuple[int, int]) -> Tuple[Array, Array, Array]**
  Convert COO triplets to CSC format arrays using pure JAX.

**sparse_solve(data: Array, indices: Array, indptr: Array, b: Array, shape: Tuple[int, int]) -> Array**
  Legacy alias for sparse_solve_csr

**build_bcoo_from_coo(rows: Array, cols: Array, values: Array, shape: Tuple[int, int]) -> 'jax.experimental.sparse.BCOO'**
  Build BCOO sparse matrix from COO triplets.

**sparse_solve_bcoo(bcoo_matrix: 'jax.experimental.sparse.BCOO', b: Array) -> Array**
  Solve sparse linear system Ax = b with BCOO format matrix.

**dense_to_sparse_gpu(dense_matrix: Array, threshold: float) -> 'jax.experimental.sparse.BCOO'**
  Convert dense matrix to BCOO sparse format on GPU.

**sparse_solve_from_dense_gpu(dense_matrix: Array, b: Array) -> Array**
  Solve Ax = b where A is provided as dense but solved as sparse.

### jax_spice/analysis/spineax_solver.py

**class SpineaxSolver**
  Sparse solver with cached symbolic factorization using Spineax/cuDSS.
  - solve(self, data: Array, b: Array) -> Tuple[Array, dict]
      Solve Ax = b using cached symbolic factorization.

**is_spineax_available() -> bool**
  Check if Spineax is available for use.

**create_spineax_solver(indptr: Array, indices: Array, n: int, device_id: int) -> Optional[SpineaxSolver]**
  Create a Spineax solver if available, else return None.

**sparse_solve_with_spineax(data: Array, indices: Array, indptr: Array, b: Array, solver: Optional[SpineaxSolver]) -> Array**
  Solve sparse system, using Spineax if solver provided, else JAX spsolve.

### jax_spice/analysis/transient/_mna.py

**class CircuitData**
  Pre-compiled circuit data for JIT-compatible simulation

**transient_analysis_jit(system: MNASystem, t_stop: float, t_step: float, t_start: float, initial_conditions: Optional[Dict[str, float]], max_iterations: int, abstol: float, reltol: float, backend: Optional[str]) -> Tuple[Array, Array, Dict]**
  Run JIT-compiled transient analysis

**transient_analysis_vectorized(system: MNASystem, t_stop: float, t_step: float, t_start: float, initial_conditions: Optional[Dict[str, float]], max_iterations: int, abstol: float, reltol: float, gmin: float, backend: Optional[str], use_sparse: bool, gmres_maxiter: int, gmres_tol: float) -> Tuple[Array, Array, Dict]**
  GPU-optimized transient analysis using vectorized device evaluation.

**transient_analysis(system: MNASystem, t_stop: float, t_step: float, t_start: float, initial_conditions: Optional[Dict[str, float]], max_iterations: int, abstol: float, reltol: float, save_all: bool, use_jit: bool) -> Tuple[Array, Array, Dict]**
  Run transient analysis

### jax_spice/analysis/transient/base.py

**class TransientSetup**
  Cached transient simulation setup data.

**class TransientStrategy**
  Abstract base class for transient analysis strategies.
  - name(self) -> str
      Human-readable strategy name for logging.
  - ensure_setup(self) -> TransientSetup
      Ensure transient setup is initialized, using cache if available.
  - ensure_solver(self) -> Callable
      Ensure NR solver is initialized, using cache if available.
  - run(self, t_stop: float, dt: float, max_steps: int) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]
      Run transient analysis.

### jax_spice/analysis/transient/python_loop.py

**class PythonLoopStrategy**
  Transient analysis using Python for-loop with JIT-compiled NR solver.
  - run(self, t_stop: float, dt: float, max_steps: int) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]
      Run transient analysis with Python for-loop.

### jax_spice/analysis/transient/scan.py

**class ScanStrategy**
  Transient analysis using lax.scan for fully JIT-compiled simulation.
  - run(self, t_stop: float, dt: float, max_steps: int) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]
      Run transient analysis with lax.scan.

### jax_spice/benchmarks/runner.py

**class VACASKBenchmarkRunner**
  Generic runner for VACASK benchmark circuits.
  - clear_cache(self)
      Clear all cached data to free memory between benchmarks.
  - parse_spice_number(self, s: str) -> float
      Parse SPICE number with suffix (e.g., 1u, 100n, 1.5k)
  - parse(self)
      Parse the sim file and extract circuit information.
  - to_mna_system(self) -> MNASystem
      Convert parsed devices to MNASystem for production analysis.
  - run_transient(self, t_stop: Optional[float], dt: Optional[float], max_steps: int, use_sparse: Optional[bool], backend: Optional[str], use_scan: bool, use_while_loop: bool, profile_config: Optional['ProfileConfig']) -> Tuple[jax.Array, Dict[int, jax.Array], Dict]
      Run transient analysis.

### jax_spice/devices/base.py

**class DeviceStamps**
  Device contribution to circuit equations

**class Device**
  Base protocol for all device models
  - evaluate(self, voltages: Dict[str, float], params: Optional[Dict[str, float]], context: Optional['AnalysisContext']) -> DeviceStamps
      Evaluate device at given terminal voltages

**class LinearDevice**
  Helper class for linear devices (resistors, capacitors, etc.)
  - two_terminal(value: float, V1: float, V2: float, terminal_names: Tuple[str, str]) -> DeviceStamps
      Generic two-terminal linear element

### jax_spice/devices/capacitor.py

**class Capacitor**
  Two-terminal capacitor device
  - evaluate(self, voltages: Dict[str, float], params: Optional[Dict[str, float]], context: Optional['AnalysisContext']) -> DeviceStamps
      Evaluate capacitor at given terminal voltages

**capacitor(Vp: Array, Vn: Array, C: Array) -> Tuple[Array, Array]**
  Functional capacitor model for charge calculation

**capacitor_companion(V: Array, V_prev: Array, C: Array, dt: Array) -> Tuple[Array, Array, Array]**
  Backward Euler companion model for capacitor

### jax_spice/devices/mosfet_simple.py

**class MOSFETParams**
  MOSFET model parameters
  - Cox(self)
      Oxide capacitance per unit area (F/m^2)
  - Vt(self)
      Thermal voltage (V)
  - beta(self)
      Transconductance parameter before mobility degradation

**class MOSFETSimple**
  Simplified MOSFET device model with automatic differentiation
  - evaluate(self, voltages: Dict[str, float], context: Optional['AnalysisContext']) -> DeviceStamps
      Evaluate MOSFET at given terminal voltages

**mosfet_ids(Vgs: float, Vds: float, Vbs: float, params: MOSFETParams) -> float**
  MOSFET drain current

**mosfet_batch(V_batch: Array, params: Dict[str, Array]) -> Tuple[Array, Array, Array, Array]**
  Vectorized MOSFET evaluation for batch processing.

### jax_spice/devices/resistor.py

**class Resistor**
  Two-terminal resistor device
  - evaluate(self, voltages: Dict[str, float], params: Optional[Dict[str, float]], context: Optional['AnalysisContext']) -> DeviceStamps
      Evaluate resistor at given terminal voltages

**resistor(Vp: Array, Vn: Array, R: Array) -> Tuple[Array, Array]**
  Functional resistor model

**resistor_batch(V_batch: Array, R_batch: Array) -> Tuple[Array, Array]**
  Vectorized resistor evaluation for batch processing

### jax_spice/devices/verilog_a.py

**class VerilogADevice**
  A device model compiled from Verilog-A using OpenVAF
  - from_va_file(cls, va_path: str, default_params: Optional[Dict[str, float]], allow_analog_in_cond: bool) -> 'VerilogADevice'
      Create a device from a Verilog-A file
  - set_parameters(self, **params)
      Set device parameters
  - get_parameter_info(self) -> List[Tuple[str, str, float]]
      Get information about all parameters
  - build_inputs(self, voltages: Dict[str, float], params: Optional[Dict[str, float]], temperature: float) -> List[float]
      Build input array for the eval function
  - eval(self, voltages: Dict[str, float], params: Optional[Dict[str, float]], temperature: float) -> Tuple[Dict, Dict]
      Evaluate the device at given voltages
  - eval_with_interpreter(self, voltages: Dict[str, float], params: Optional[Dict[str, float]], temperature: float) -> Tuple[Any, Any]
      Evaluate using the MIR interpreter (for validation)
  - get_stamps(self, node_indices: Dict[str, int], voltages: Dict[str, float], params: Optional[Dict[str, float]], temperature: float) -> Tuple[Dict[Tuple[int, int], float], Dict[int, float]]
      Get conductance matrix stamps and RHS contributions

**compile_va(va_path: str, allow_analog_in_cond: bool, **default_params) -> VerilogADevice**
  Convenience function to compile a Verilog-A file

### jax_spice/devices/vsource.py

**class VoltageSource**
  Independent voltage source with various waveform types
  - get_voltage(self, time: Optional[float]) -> float
      Get voltage at specified time
  - evaluate(self, voltages: Dict[str, float], params: Optional[Dict[str, float]], context: Optional['AnalysisContext']) -> DeviceStamps
      Evaluate voltage source at given terminal voltages

**class CurrentSource**
  Independent current source
  - evaluate(self, voltages: Dict[str, float], params: Optional[Dict[str, float]], context: Optional['AnalysisContext']) -> DeviceStamps
      Evaluate current source

**pulse_voltage(t: float, pulse_params: Tuple[float, ...]) -> float**
  Calculate pulse voltage at time t

**pulse_voltage_jax(t: Array, v0: Array, v1: Array, td: Array, tr: Array, tf: Array, pw: Array, per: Array) -> Array**
  JAX-compatible pulse voltage calculation

**vsource_batch(V_batch: Array, V_target: Array, G_BIG: float) -> Tuple[Array, Array]**
  Vectorized voltage source evaluation for batch processing

**isource_batch(I_target: Array) -> Array**
  Vectorized current source evaluation for batch processing

### jax_spice/logging.py

**class FlushingHandler**
  StreamHandler that flushes after every emit (for Cloud Run log visibility).
  - emit(self, record) ❌

**class MemoryLoggingHandler**
  StreamHandler that prepends memory stats and flushes after every emit.
  - emit(self, record) ❌

**class PerfCounterHandler**
  StreamHandler that prepends time.perf_counter() and flushes after every emit.
  - emit(self, record) ❌

**enable_performance_logging(with_memory: bool, with_perf_counter: bool)**
  Enable DEBUG level logging with immediate flush for performance tracing.

**set_log_level(level: int)**
  Set the logging level.

### jax_spice/netlist/circuit.py

**class Model**
  Model definition mapping name to device module

**class Instance**
  Device or subcircuit instance

**class Subcircuit**
  Subcircuit definition

**class Circuit**
  Top-level circuit containing all definitions
  - flatten(self, top_subckt: str) -> Tuple[List[Instance], Dict[str, int]]
      Flatten hierarchy starting from given subcircuit
  - stats(self) -> Dict[str, int]
      Return statistics about the circuit

### jax_spice/netlist/parser.py

**class Token** ❌

**class Lexer**
  Simple lexer for VACASK netlist format

**class Parser**
  Recursive descent parser for VACASK format
  - current(self) -> Token ❌
  - peek(self, offset: int) -> Token ❌
  - advance(self) -> Token ❌
  - skip_newlines(self) ❌
  - expect(self, type_: str) -> Token ❌
  - parse(self) -> Circuit ❌
  - statement(self) ❌
  - load_stmt(self) ❌
  - include_stmt(self) ❌
  - global_stmt(self) ❌
  - ground_stmt(self) ❌
  - model_stmt(self) ❌
  - parameters_stmt(self) ❌
  - subckt_def(self) ❌
  - instance_stmt(self) ❌
  - param_list(self) -> Dict[str, str] ❌
  - param_list_multiline(self) -> Dict[str, str]
      Parse parameter list that can span multiple lines
  - param_value(self) -> str
      Parse a parameter value (may be string, name, value expression)
  - control_block(self)
      Skip control block
  - embed_block(self)
      Skip embed block
  - directive_block(self)
      Skip @if/@elseif/@else/@endif conditional block

**class VACASKParser**
  Parser for VACASK netlist format
  - parse(self, text: str, base_path: Optional[Path]) -> Circuit
      Parse VACASK netlist text
  - parse_file(self, filename: Union[str, Path]) -> Circuit
      Parse VACASK netlist from file, handling includes

**parse_netlist(source: Union[str, Path]) -> Circuit**
  Convenience function to parse a VACASK netlist

### jax_spice/profiling.py

**class ProfileConfig**
  Configuration for profiling.
  - enabled(self) -> bool
      Return True if any profiling is enabled.

**class ProfileTimer**
  Timer that integrates with profiling for measuring code sections.
  - elapsed_ms(self) -> float
      Elapsed time in milliseconds.
  - start(self) -> 'ProfileTimer'
      Start timing and profiling.
  - stop(self) -> 'ProfileTimer'
      Stop timing and profiling.

**get_config() -> ProfileConfig**
  Get the global profiling configuration.

**set_config(config: ProfileConfig) -> None**
  Set the global profiling configuration.

**enable_profiling(jax: bool, cuda: bool, trace_dir: Optional[str]) -> None**
  Enable profiling globally.

**disable_profiling() -> None**
  Disable all profiling globally.

**profile_section(name: str, config: Optional[ProfileConfig])**
  Context manager for profiling a code section.

**profile(func: Optional[F]) -> Union[F, Callable[[F], F]]**
  Decorator for profiling a function.

### jax_spice/simulator.py

**class TransientResult**
  Result of a transient simulation.
  - num_steps(self) -> int
      Number of timesteps in the simulation.
  - voltage(self, node: Union[int, str]) -> Array
      Get voltage waveform at a specific node.

**class Simulator**
  JAX-SPICE circuit simulator.
  - parse(self) -> 'Simulator'
      Parse the circuit file.
  - circuit_path(self) -> Path
      Path to the circuit file.
  - num_nodes(self) -> int
      Number of circuit nodes (excluding ground).
  - node_names(self) -> Dict[str, int]
      Mapping of node names to indices.
  - analysis_params(self) -> Dict[str, Any]
      Analysis parameters from the circuit file (dt, stop time, etc.).
  - devices(self) -> list
      List of parsed devices with model info.
  - is_warmed_up(self) -> bool
      Whether JIT warmup has been performed.
  - warmup(self, t_stop: float, dt: float) -> 'Simulator'
      Warmup JIT compilation.
  - transient(self, t_stop: float, dt: float) -> TransientResult
      Run transient simulation.

### openvaf-py/compare_jax_interpreter.py

**compare_test(V, R, temperature, tnom, zeta, mfactor)**
  Compare JAX vs interpreter results

### openvaf-py/openvaf_jax.py

**class CompiledDevice**
  A compiled Verilog-A device with JAX evaluation function

**class OpenVAFToJAX**
  Translates OpenVAF MIR to JAX functions
  - from_file(cls, va_path: str) -> 'OpenVAFToJAX'
      Create translator from a Verilog-A file
  - translate(self) -> Callable
      Generate a JAX function from the MIR
  - translate_array(self) -> Tuple[Callable, Dict]
      Generate a JAX function that returns arrays (vmap-compatible)
  - translate_init_array(self) -> Tuple[Callable, Dict]
      Generate a standalone vmappable init function.
  - translate_eval_array_with_cache(self) -> Tuple[Callable, Dict]
      Generate a vmappable eval function that takes cache as input.
  - get_parameter_info(self) -> List[Tuple[str, str, str]]
      Get parameter information
  - get_generated_code(self) -> str
      Get the generated JAX code as a string

**compile_va(va_path: str) -> CompiledDevice**
  Compile a Verilog-A file to a JAX-compatible device

### openvaf-py/src/lib.rs

**class OsdiParamInfo** ❌

**class OsdiNodeInfo** ❌

**class OsdiJacobianInfo** ❌

**class OsdiNoiseInfo** ❌

**class VaModule** ❌
  - get_residual_structure(&self) (Vec<usize>, Vec<u32>, Vec<u32>)
      Get residual structure as (row_indices, resist_var_indices, react_var_indices)
  - get_jacobian_structure(&self) (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)
      Get Jacobian structure as (row_indices, col_indices, resist_var_indices, react_var_indices)
  - get_all_func_params(&self) Vec<(u32, u32)>
      Get all Param-defined values in the function
  - get_mir(&self, literals: Vec<String>) String
      Get MIR function as string for debugging
  - get_num_func_calls(&self) usize
      Get number of function calls (built-in functions) in the MIR
  - get_cache_mapping(&self) Vec<(u32, u32)>
      Get cache mapping as list of (init_value_idx, eval_param_idx)
  - get_param_defaults(&self) HashMap<String, f64>
      Get parameter defaults extracted from Verilog-A source
  - get_str_constants(&self) HashMap<String, String>
      Get resolved string constant values
  - get_osdi_descriptor(&self) std::collections::HashMap<String, pyo3::PyObject>
      Get OSDI-compatible descriptor metadata
  - get_mir_instructions(&self) std::collections::HashMap<String, pyo3::PyObject>
      Export MIR instructions for JAX translation
  - get_init_mir_instructions(&self) std::collections::HashMap<String, pyo3::PyObject>
      Export init function MIR instructions for JAX translation
  - get_dae_system(&self) std::collections::HashMap<String, pyo3::PyObject>
      Export DAE system (residuals and Jacobian) for JAX translation
  - run_init_eval(&self, params: std::collections::HashMap<String, f64>) PyResult<(Vec<(f64, f64)>, Vec<(u32, u32, f64, f64)>)>
      Run init function and then eval function
  - stub_callback(state: &mut InterpreterState, _args: &[Value], rets: &[Value], _data: *mut c_void) ❌
  - evaluate(&self, params: std::collections::HashMap<String, f64>) PyResult<std::collections::HashMap<String, Vec<(f64, f64)>>>
      Evaluate the module with given parameter values
  - evaluate_full(&self, params: std::collections::HashMap<String, f64>, extra_params: Option<Vec<f64>>) PyResult<(Vec<(f64, f64)>, Vec<(u32, u32, f64, f64)>)>
      Evaluate and return full results as nested structure
  - stub_callback(state: &mut InterpreterState, _args: &[Value], rets: &[Value], _data: *mut c_void) ❌

**compile_va(path: &str, allow_analog_in_cond: bool, allow_builtin_primitives: bool) PyResult<Vec<VaModule>>** ❌

**openvaf_py(_py: Python, m: &PyModule) PyResult<()>** ❌

### openvaf-py/test_all_models.py

**count_by_kind(param_kinds)**
  Count parameters by kind

### openvaf-py/test_jax_diode.py

**test_diode(V_diode, temperature)**
  Test diode at given forward voltage

### openvaf-py/test_jax_resistor.py

**test_resistor(V, R, temperature, tnom, zeta, mfactor)**
  Test resistor with given parameters

### openvaf-py/test_pdk_models.py

**find_va_files(base_path)**
  Find all .va files under a path

**test_model(va_path, allow_analog_in_cond)**
  Test compiling a single model

**test_pdk(name, base_path, allow_analog_in_cond)**
  Test all models in a PDK

### openvaf-py/tests/conftest.py

**class CompiledModel**
  Wrapper for a compiled Verilog-A model with JAX function
  - name(self) -> str ❌
  - nodes(self) -> List[str] ❌
  - param_names(self) -> List[str] ❌
  - param_kinds(self) -> List[str] ❌
  - build_default_inputs(self) -> List[float]
      Build input array with sensible defaults
  - evaluate(self, inputs: List[float]) -> Tuple[Dict, Dict]
      Evaluate the JAX function
  - run_interpreter(self, params: Dict[str, float]) -> Tuple[List, List]
      Run the MIR interpreter

**compile_model()**
  Factory fixture to compile VA models

**resistor_model(compile_model) -> CompiledModel**
  Compiled resistor model

**diode_model(compile_model) -> CompiledModel**
  Compiled diode model

**diode_cmc_model(compile_model) -> CompiledModel**
  Compiled CMC diode model

**isrc_model(compile_model) -> CompiledModel**
  Compiled current source model

**vccs_model(compile_model) -> CompiledModel**
  Compiled VCCS model

**cccs_model(compile_model) -> CompiledModel**
  Compiled CCCS model

**assert_allclose(actual, expected, rtol, atol, err_msg)**
  Assert that actual and expected are close within tolerances

**build_param_dict(model: CompiledModel, inputs: List[float]) -> Dict[str, float]**
  Build parameter dictionary from input list

### openvaf-py/tests/snap_parser.py

**class ParamInfo**
  Parameter metadata from snapshot.
  - is_instance(self) -> bool
      Check if this is an instance parameter (vs model parameter).
  - is_model(self) -> bool
      Check if this is a model parameter.

**class NodeInfo**
  Node metadata from snapshot.

**class JacobianEntry**
  Jacobian entry metadata from snapshot.
  - has_resist(self) -> bool ❌
  - has_react(self) -> bool ❌
  - resist_const(self) -> bool ❌
  - react_const(self) -> bool ❌

**class CollapsiblePair**
  Collapsible node pair from snapshot.

**class NoiseSource**
  Noise source from snapshot.

**class SnapshotData**
  Parsed OSDI snapshot data.

**parse_flags(flags_str: str) -> int**
  Parse ParameterFlags or JacobianFlags string to integer.

**parse_snap_file(path: str | Path) -> SnapshotData**
  Parse an OpenVAF .snap file into structured data.

**load_snap_file(model_name: str, snap_dir: str | Path | None) -> SnapshotData**
  Load a snapshot file by model name.

### openvaf-py/tests/test_bjt_models.py

**class TestHICUM**
  Test HICUM Level 2 BJT model
  - test_compilation(self, hicum_model: CompiledModel)
      HICUM model compiles without error
  - test_valid_output(self, hicum_model: CompiledModel)
      HICUM produces valid outputs
  - test_has_jacobian(self, hicum_model: CompiledModel)
      HICUM has jacobian entries
  - test_complexity(self, hicum_model: CompiledModel)
      HICUM is a complex model

**class TestMEXTRAM**
  Test MEXTRAM BJT model
  - test_compilation(self, mextram_model: CompiledModel)
      MEXTRAM model compiles without error
  - test_valid_output(self, mextram_model: CompiledModel)
      MEXTRAM produces valid outputs
  - test_has_jacobian(self, mextram_model: CompiledModel)
      MEXTRAM has jacobian entries
  - test_complexity(self, mextram_model: CompiledModel)
      MEXTRAM is a complex model

**class TestBJTBehavior**
  Test physical behavior of BJT models
  - test_has_multiple_nodes(self, request, fixture_name)
      BJT model has multiple terminal nodes

**hicum_model(compile_model) -> CompiledModel**
  Compiled HICUM L2 model

**mextram_model(compile_model) -> CompiledModel**
  Compiled MEXTRAM model

### openvaf-py/tests/test_hemt_models.py

**class TestASMHEMT**
  Test ASMHEMT GaN HEMT model
  - test_compilation(self, asmhemt_model: CompiledModel)
      ASMHEMT model compiles without error
  - test_valid_output(self, asmhemt_model: CompiledModel)
      ASMHEMT produces valid outputs
  - test_has_jacobian(self, asmhemt_model: CompiledModel)
      ASMHEMT has jacobian entries

**class TestMVSG**
  Test MVSG GaN HEMT model
  - test_compilation(self, mvsg_model: CompiledModel)
      MVSG model compiles without error
  - test_valid_output(self, mvsg_model: CompiledModel)
      MVSG produces valid outputs
  - test_has_jacobian(self, mvsg_model: CompiledModel)
      MVSG has jacobian entries

**class TestHEMTBehavior**
  Test physical behavior of HEMT models
  - test_has_multiple_nodes(self, request, fixture_name)
      HEMT model has multiple terminal nodes

**asmhemt_model(compile_model) -> CompiledModel**
  Compiled ASMHEMT model

**mvsg_model(compile_model) -> CompiledModel**
  Compiled MVSG model

### openvaf-py/tests/test_hisim_models.py

**class TestHiSIM2**
  Test HiSIM2 MOSFET model
  - test_compilation(self, hisim2_model: CompiledModel)
      HiSIM2 model compiles without error
  - test_valid_output(self, hisim2_model: CompiledModel)
      HiSIM2 produces valid outputs
  - test_has_jacobian(self, hisim2_model: CompiledModel)
      HiSIM2 has jacobian entries

**class TestHiSIMHV**
  Test HiSIMHV high-voltage MOSFET model
  - test_compilation(self, hisimhv_model: CompiledModel)
      HiSIMHV model compiles without error
  - test_valid_output(self, hisimhv_model: CompiledModel)
      HiSIMHV produces valid outputs
  - test_complexity(self, hisimhv_model: CompiledModel)
      HiSIMHV is a complex model

**hisim2_model(compile_model) -> CompiledModel**
  Compiled HiSIM2 model

**hisimhv_model(compile_model) -> CompiledModel**
  Compiled HiSIMHV model

### openvaf-py/tests/test_jax_analytical.py

**class TestResistorAnalytical**
  Test resistor model against Ohm's law
  - resistor(self, compile_model) ❌
  - test_ohms_law_current(self, resistor, voltage, resistance)
      I = V/R
  - test_ohms_law_conductance(self, resistor, voltage, resistance)
      dI/dV = 1/R (Jacobian)
  - test_temperature_scaling(self, resistor, temperature, zeta)
      R_eff = R * (T/Tnom)^zeta

**class TestCurrentSourceAnalytical**
  Test current source model
  - isrc(self, compile_model) ❌
  - test_current_source_compiles(self, isrc)
      Current source compiles and produces output

### openvaf-py/tests/test_jax_equivalence.py

**class TestResistorEquivalence**
  Tests for resistor JAX translation equivalence.
  - resistor_translator(self)
      Load resistor translator.
  - test_resistor_translation_compiles(self, resistor_translator)
      Test that resistor JAX translation compiles without error.
  - test_resistor_input_output_shapes(self, resistor_translator)
      Test that resistor JAX function has correct input/output shapes.
  - test_resistor_jax_outputs_finite(self, resistor_translator)
      Test that resistor JAX function produces finite outputs.

**class TestDiodeEquivalence**
  Tests for diode JAX translation equivalence.
  - diode_translator(self)
      Load diode translator.
  - test_diode_translation_compiles(self, diode_translator)
      Test that diode JAX translation compiles without error.
  - test_diode_zero_bias_output(self, diode_translator)
      Test diode at zero bias produces near-zero current.

**class TestCapacitorEquivalence**
  Tests for capacitor JAX translation equivalence.
  - capacitor_translator(self)
      Load capacitor translator.
  - test_capacitor_translation_compiles(self, capacitor_translator)
      Test that capacitor JAX translation compiles without error.
  - test_capacitor_has_reactive_outputs(self, capacitor_translator)
      Test that capacitor produces reactive (charge) outputs.

**class TestJITCompilation**
  Tests for JAX JIT compilation correctness.
  - resistor_translator(self)
      Load resistor translator.
  - test_jit_compilation_succeeds(self, resistor_translator)
      Test that JAX JIT compilation succeeds.
  - test_jit_multiple_inputs(self, resistor_translator)
      Test JIT function with different inputs.

**class TestVmapBatching**
  Tests for JAX vmap batching correctness.
  - resistor_translator(self)
      Load resistor translator.
  - test_vmap_batching_succeeds(self, resistor_translator)
      Test that vmap batching works correctly.
  - test_vmap_consistency_with_loop(self, resistor_translator)
      Test that vmap produces same results as explicit loop.

**get_translator(va_path: Path) -> OpenVAFToJAX**
  Create a JAX translator from a Verilog-A file.

### openvaf-py/tests/test_jax_interpreter.py

**class TestJaxVsInterpreter**
  Compare JAX function output against MIR interpreter for all models
  - test_model_compiles(self, compile_model, model_name, model_path)
      Model compiles to JAX without error
  - test_simple_model_produces_valid_output(self, compile_model, model_name, model_path)
      Simple JAX function produces non-NaN outputs
  - test_working_complex_model_produces_valid_output(self, compile_model, model_name, model_path)
      Working complex JAX function produces non-NaN resist outputs
  - test_failing_complex_model_produces_valid_output(self, compile_model, model_name, model_path)
      Failing complex JAX function produces non-NaN resist outputs

**class TestResistorJaxInterpreter**
  Detailed comparison for resistor model
  - test_resistor_residuals_match(self, resistor_model: CompiledModel, voltage, resistance, temperature, tnom, zeta, mfactor)
      Compare JAX vs interpreter residuals for resistor
  - test_resistor_jacobian_match(self, resistor_model: CompiledModel, voltage, resistance, temperature, tnom, zeta, mfactor)
      Compare JAX vs interpreter jacobian for resistor

**class TestModelNodeCounts**
  Verify models have reasonable node counts
  - test_has_nodes(self, compile_model, model_name, model_path)
      Model has at least 2 nodes
  - test_two_terminal_devices(self, compile_model, model_name, model_path)
      Two-terminal devices have 2 nodes
  - test_four_terminal_devices(self, compile_model, model_name, model_path)
      Controlled sources have at least 4 terminals

**class TestModelComplexity**
  Test that complex models compile and produce outputs
  - test_complex_model_compiles(self, compile_model, model_name, model_path)
      Complex model compiles to JAX
  - test_working_complex_model_outputs(self, compile_model, model_name, model_path)
      Working complex model produces finite outputs
  - test_failing_complex_model_outputs(self, compile_model, model_name, model_path)
      Failing complex model produces finite outputs

### openvaf-py/tests/test_jax_vs_osdi.py

**class TestJaxCompilation**
  Verify all models compile to JAX without error
  - test_compiles(self, compile_model, model_name, model_path)
      Model compiles to JAX function

**class TestJaxProducesValidOutput**
  Verify JAX functions produce valid (non-NaN) output
  - test_simple_model_output(self, compile_model, model_name, model_path)
      Simple models produce valid output with default inputs

**class TestResistorDetailed**
  Detailed tests for resistor model with various parameter combinations
  - resistor(self, compile_model) ❌
  - test_ohms_law(self, resistor, voltage, resistance)
      Resistor follows Ohm's law: I = V/R
  - test_temperature_dependence(self, resistor, temperature, tnom, zeta)
      Resistor temperature dependence matches between JAX and interpreter

**class TestDiodeDetailed**
  Detailed tests for diode model
  - diode(self, compile_model) ❌
  - test_diode_compiles(self, diode)
      Diode model compiles to JAX
  - test_diode_output_shape(self, diode)
      Diode produces expected output structure

**class TestEkvMosfet**
  Tests for EKV MOSFET model
  - ekv(self, compile_model) ❌
  - test_ekv_compiles_and_evaluates(self, ekv)
      EKV model compiles and produces valid output
  - test_ekv_matches_interpreter(self, ekv)
      EKV JAX output matches interpreter

### openvaf-py/tests/test_mosfet_models.py

**class TestEKV**
  Test EKV MOSFET model
  - test_compilation(self, ekv_model: CompiledModel)
      EKV model compiles without error
  - test_valid_output(self, ekv_model: CompiledModel)
      EKV produces valid outputs

**class TestBSIM3**
  Test BSIM3 MOSFET model
  - test_compilation(self, bsim3_model: CompiledModel)
      BSIM3 model compiles without error
  - test_valid_output(self, bsim3_model: CompiledModel)
      BSIM3 produces valid outputs
  - test_jacobian_valid(self, bsim3_model: CompiledModel)
      BSIM3 jacobian is valid

**class TestBSIM4**
  Test BSIM4 MOSFET model
  - test_compilation(self, bsim4_model: CompiledModel)
      BSIM4 model compiles without error
  - test_valid_output(self, bsim4_model: CompiledModel)
      BSIM4 produces valid outputs
  - test_has_many_jacobian_entries(self, bsim4_model: CompiledModel)
      BSIM4 has substantial jacobian (complex model)

**class TestBSIM6**
  Test BSIM6 MOSFET model
  - test_compilation(self, bsim6_model: CompiledModel)
      BSIM6 model compiles without error
  - test_valid_output(self, bsim6_model: CompiledModel)
      BSIM6 produces valid outputs

**class TestBSIMBULK**
  Test BSIMBULK MOSFET model
  - test_compilation(self, bsimbulk_model: CompiledModel)
      BSIMBULK model compiles without error
  - test_valid_output(self, bsimbulk_model: CompiledModel)
      BSIMBULK produces valid outputs

**class TestBSIMCMG**
  Test BSIMCMG FinFET model
  - test_compilation(self, bsimcmg_model: CompiledModel)
      BSIMCMG model compiles without error
  - test_valid_output(self, bsimcmg_model: CompiledModel)
      BSIMCMG produces valid outputs

**class TestBSIMSOI**
  Test BSIMSOI SOI MOSFET model
  - test_compilation(self, bsimsoi_model: CompiledModel)
      BSIMSOI model compiles without error
  - test_valid_output(self, bsimsoi_model: CompiledModel)
      BSIMSOI produces valid outputs

**class TestMOSFETBehavior**
  Test physical behavior of MOSFET models
  - test_has_multiple_nodes(self, request, fixture_name)
      MOSFET model has multiple terminal nodes

**ekv_model(compile_model) -> CompiledModel**
  Compiled EKV model

**bsim3_model(compile_model) -> CompiledModel**
  Compiled BSIM3 model

**bsim4_model(compile_model) -> CompiledModel**
  Compiled BSIM4 model

**bsim6_model(compile_model) -> CompiledModel**
  Compiled BSIM6 model

**bsimbulk_model(compile_model) -> CompiledModel**
  Compiled BSIMBULK model

**bsimcmg_model(compile_model) -> CompiledModel**
  Compiled BSIMCMG model

**bsimsoi_model(compile_model) -> CompiledModel**
  Compiled BSIMSOI model

### openvaf-py/tests/test_osdi_evaluation.py

**class TestResistorEvaluation**
  Tests for resistor device evaluation correctness.
  - resistor_module(self)
      Load resistor module.
  - test_resistor_residual_ohms_law(self, resistor_module)
      Test that resistor follows Ohm's law: I = V/R.
  - test_resistor_jacobian_conductance(self, resistor_module)
      Test that resistor has proper Jacobian structure.

**class TestDiodeEvaluation**
  Tests for diode device evaluation correctness.
  - diode_module(self)
      Load diode module.
  - test_diode_forward_bias_physics(self, diode_module)
      Test diode follows Shockley equation in forward bias.
  - test_diode_jacobian_nonlinear(self, diode_module)
      Test that diode Jacobian is NOT constant (nonlinear device).

**class TestCapacitorEvaluation**
  Tests for capacitor device evaluation correctness.
  - capacitor_module(self)
      Load capacitor module.
  - test_capacitor_has_reactive_contribution(self, capacitor_module)
      Test that capacitor has reactive (charge storage) contribution.

**class TestParameterDefaults**
  Tests for parameter default value extraction.
  - test_resistor_has_default_resistance(self)
      Test that resistor has extractable default resistance value.
  - test_diode_has_default_is(self)
      Test that diode has extractable default Is value.

**class TestNodeCollapse**
  Tests for node collapse functionality.
  - test_diode_has_collapsible_pairs(self)
      Test that diode with internal node has collapsible pairs.

**class TestNoiseSourceExtraction**
  Tests for noise source metadata extraction.
  - test_diode_has_noise_sources(self)
      Test that diode has shot noise sources.

**class TestVCCSEvaluation**
  Tests for voltage-controlled current source evaluation.
  - vccs_module(self)
      Load VCCS module.
  - test_vccs_has_linear_gain(self, vccs_module)
      Test that VCCS has constant (linear) gain.

**get_module(va_path: Path) -> openvaf_py.VaModule**
  Compile a Verilog-A file and return the module.

**vt(temp: float) -> float**
  Calculate thermal voltage Vt = kT/q.

### openvaf-py/tests/test_osdi_metadata.py

**class TestParameterMetadata**
  Tests for parameter metadata correctness.
  - test_parameter_names_match(self, model_name: str, va_path: Path)
      Verify parameter names match OSDI reference (excluding built-ins).
  - test_parameter_units_match(self, model_name: str, va_path: Path)
      Verify parameter units match OSDI reference.
  - test_parameter_descriptions_match(self, model_name: str, va_path: Path)
      Verify parameter descriptions match OSDI reference.
  - test_parameter_instance_flag_match(self, model_name: str, va_path: Path)
      Verify parameter instance/model flags match OSDI reference.

**class TestNodeMetadata**
  Tests for node metadata correctness.
  - test_terminal_count_matches(self, model_name: str, va_path: Path)
      Verify terminal count matches OSDI reference.
  - test_node_count_matches(self, model_name: str, va_path: Path)
      Verify total node count matches OSDI reference.
  - test_node_names_match(self, model_name: str, va_path: Path)
      Verify node names match OSDI reference.

**class TestJacobianMetadata**
  Tests for Jacobian structure correctness.
  - test_jacobian_count_matches(self, model_name: str, va_path: Path)
      Verify Jacobian entry count matches OSDI reference.
  - test_jacobian_sparsity_pattern_matches(self, model_name: str, va_path: Path)
      Verify Jacobian (row, col) sparsity pattern matches OSDI reference.
  - test_jacobian_flags_match(self, model_name: str, va_path: Path)
      Verify Jacobian entry flags match OSDI reference.

**class TestCollapsiblePairs**
  Tests for node collapse metadata.
  - test_collapsible_count_matches(self, model_name: str, va_path: Path)
      Verify collapsible pair count matches OSDI reference.

**class TestMiscMetadata**
  Tests for miscellaneous metadata fields.
  - test_num_states_matches(self, model_name: str, va_path: Path)
      Verify num_states matches OSDI reference.
  - test_has_bound_step_matches(self, model_name: str, va_path: Path)
      Verify has_bound_step matches OSDI reference.

**get_osdi_descriptor(va_path: Path) -> dict**
  Compile a Verilog-A file and return its OSDI descriptor.

### openvaf-py/tests/test_osdi_methodology.py

**class TestDiodeLimOSDI**
  Test diode_lim model using OpenVAF's exact test methodology
  - diode_lim(self)
      Compile diode_lim model
  - diode_jax(self, diode_lim)
      Create JAX function from diode_lim model
  - test_model_compiles(self, diode_lim)
      Model compiles successfully
  - test_model_has_expected_parameters(self, diode_lim)
      Model has is and cj0 parameters
  - test_vcrit_calculation(self)
      Verify vcrit calculation matches OpenVAF
  - test_zero_bias_current(self, diode_lim)
      At Vd=0, Id should be essentially zero
  - test_forward_bias_current(self, diode_lim)
      Test forward bias: Id should increase exponentially
  - test_conductance_matches_derivative(self, diode_lim)
      Verify gd = dId/dVd matches analytical formula

**class TestSimpleResistorOSDI**
  Test resistor model with OpenVAF methodology
  - resistor(self)
      Compile resistor model
  - resistor_jax(self, resistor)
      Create JAX function from resistor model
  - test_jax_vs_interpreter_residual(self, resistor, resistor_jax, voltage, resistance)
      JAX output matches interpreter for resistor
  - test_jax_vs_interpreter_jacobian(self, resistor, resistor_jax, voltage, resistance)
      JAX Jacobian matches interpreter for resistor

**class TestDAEEquations**
  Test DAE equation formulation as in OpenVAF's check_dae_equations
  - test_dae_formulation_concept(self)
      Verify understanding of DAE formulation

**class TestSPICEEquations**
  Test SPICE equation formulation as in OpenVAF's check_spice_equations
  - test_spice_formulation_concept(self)
      Verify understanding of SPICE formulation

**id_func(vd: float) -> float**
  Diode current: Id = Is * (exp(Vd/Vt) - 1)

**gd_func(vd: float) -> float**
  Diode conductance: gd = dId/dVd = Is/Vt * exp(Vd/Vt)

**cj_func(vd: float) -> float**
  Junction charge: Qd = Cj0 * Vd (simplified model)

### openvaf-py/tests/test_psp_models.py

**class TestPSP102**
  Test PSP102 MOSFET model
  - test_compilation(self, psp102_model: CompiledModel)
      PSP102 model compiles without error
  - test_valid_output(self, psp102_model: CompiledModel)
      PSP102 produces valid outputs
  - test_has_jacobian(self, psp102_model: CompiledModel)
      PSP102 has jacobian entries

**class TestPSP103**
  Test PSP103 MOSFET model (latest PSP version)
  - test_compilation(self, psp103_model: CompiledModel)
      PSP103 model compiles without error
  - test_valid_output(self, psp103_model: CompiledModel)
      PSP103 produces valid outputs
  - test_complexity(self, psp103_model: CompiledModel)
      PSP103 is a complex model

**class TestJUNCAP**
  Test JUNCAP200 junction capacitance model
  - test_compilation(self, juncap_model: CompiledModel)
      JUNCAP200 model compiles without error
  - test_valid_output(self, juncap_model: CompiledModel)
      JUNCAP200 produces valid outputs
  - test_is_two_terminal(self, juncap_model: CompiledModel)
      JUNCAP is a two-terminal device

**psp102_model(compile_model) -> CompiledModel**
  Compiled PSP102 model

**psp103_model(compile_model) -> CompiledModel**
  Compiled PSP103 model

**juncap_model(compile_model) -> CompiledModel**
  Compiled JUNCAP200 model

### openvaf-py/tests/test_simple_models.py

**class TestResistor**
  Test resistor model: I = V/R where R = R0 * (T/Tnom)^zeta
  - test_ohms_law(self, resistor_model: CompiledModel, voltage, resistance)
      Test basic Ohm's law: I = V/R
  - test_conductance(self, resistor_model: CompiledModel, voltage, resistance)
      Test Jacobian entry: G = 1/R
  - test_temperature_coefficient(self, resistor_model: CompiledModel, temperature, tnom, zeta)
      Test R(T) = R0 * (T/Tnom)^zeta
  - test_multiplier(self, resistor_model: CompiledModel, mfactor)
      Test multiplier factor

**class TestDiode**
  Test diode model: I = Is * (exp(V/(n*Vt)) - 1)
  - test_compilation(self, diode_model: CompiledModel)
      Diode model compiles without error
  - test_zero_bias(self, diode_model: CompiledModel)
      At zero bias, current should be near zero
  - test_forward_bias_increases_current(self, diode_model: CompiledModel)
      Forward bias should increase current exponentially

**class TestCurrentSource**
  Test ideal current source model
  - test_compilation(self, isrc_model: CompiledModel)
      Current source compiles without error
  - test_constant_current(self, isrc_model: CompiledModel)
      Current should be constant regardless of voltage

**class TestVCCS**
  Test voltage-controlled current source
  - test_compilation(self, vccs_model: CompiledModel)
      VCCS compiles without error
  - test_output_valid(self, vccs_model: CompiledModel)
      VCCS produces valid outputs

**class TestCCCS**
  Test current-controlled current source
  - test_compilation(self, cccs_model: CompiledModel)
      CCCS compiles without error
  - test_output_valid(self, cccs_model: CompiledModel)
      CCCS produces valid outputs

### openvaf-py/tests/test_va_features.py

**class TestCollapsableResistor**
  Test conditional contributions: if (R > minr) I<+V/R else V<+0.
  - diode_model(self, compile_model)
      Compile diode model which has CollapsableR pattern.
  - test_resistive_mode(self, diode_model)
      Test diode with rs > 0 (resistor mode, I<+V/R contribution).
  - test_collapsed_mode(self, diode_model)
      Test diode with rs = 0 (collapsed mode, V<+0 contribution).

**class TestPSP103NoiseCorrelationNode**
  Test PSP103 NOI (noise correlation) internal node handling.
  - psp103_model(self, compile_model)
      Compile PSP103 model.
  - test_has_noi_node(self, psp103_model)
      PSP103 should have NOI as an internal node.
  - test_has_vnoi_voltage_input(self, psp103_model)
      PSP103 should have V(NOI) as a voltage input parameter.
  - test_has_branch_current_input(self, psp103_model)
      PSP103 should have I(NOII) as a current input parameter.
  - test_noi_zero_voltage_residuals(self, psp103_model)
      With V(NOI)=0, all residuals should be finite and reasonable.
  - test_noi_nonzero_voltage_stability(self, psp103_model)
      With V(NOI)=0.6V, residuals should still be bounded.

**class TestCollapsiblePairs**
  Test node collapse information from OpenVAF.
  - psp103_module(self)
      Get raw PSP103 module.
  - test_has_collapsible_pairs(self, psp103_module)
      PSP103 should report collapsible node pairs.
  - test_noi_not_collapsible(self, psp103_module)
      NOI node should NOT be in collapsible pairs.

**class TestLargeConductance**
  Test numerical stability with large conductance values (> 1e20).
  - resistor_model(self, compile_model)
      Compile resistor model.
  - test_small_resistance_large_conductance(self, resistor_model)
      Test resistor with very small R (large G = 1/R).

**class TestInternalNodeAllocation**
  Test that internal nodes are properly allocated in models.
  - diode_module(self)
      Get raw diode module.
  - test_diode_has_internal_node(self, diode_module)
      Diode should have CI as internal node.
  - test_diode_residual_count_matches_nodes(self, diode_module)
      Number of residuals should match number of nodes.

### openvaf-py/tests_pdk/conftest.py

**pytest_configure(config)**
  Register custom markers

**pytest_collection_modifyitems(config, items)**
  Skip PDK tests if PDK not available

**compile_pdk_model_fixture()**
  Factory fixture to compile PDK VA models

### openvaf-py/tests_pdk/pdk_utils.py

**class CompiledPDKModel**
  Wrapper for a compiled PDK Verilog-A model with JAX function
  - name(self) -> str ❌
  - nodes(self) ❌
  - param_names(self) ❌
  - param_kinds(self) ❌
  - build_default_inputs(self)
      Build input array with sensible defaults

**get_pdk_path(env_var: str) -> Optional[Path]**
  Get PDK path from environment variable

**sanitize_pdk_path(msg: str) -> str**
  Remove PDK paths from error messages to avoid leaking in CI logs

**compile_pdk_model(model_path: Path, allow_analog_in_cond: bool) -> CompiledPDKModel**
  Compile a PDK VA model with path sanitization

### openvaf-py/tests_pdk/test_gf130.py

**class TestGF130Compilation**
  Test that GF130 models compile to JAX
  - test_pdk_path_exists(self)
      GF130 PDK path is valid
  - test_has_va_files(self)
      GF130 PDK contains .va files
  - test_model_count(self)
      GF130 has expected number of models
  - test_all_models_compile(self)
      GF130 .va files compile to JAX (excluding known failures)

**class TestGF130ModelProperties**
  Test properties of compiled GF130 models
  - sample_model(self) -> CompiledPDKModel
      Get a sample compiled GF130 model for testing
  - test_model_has_nodes(self, sample_model)
      Compiled model has nodes
  - test_model_has_params(self, sample_model)
      Compiled model has parameters
  - test_model_has_jax_function(self, sample_model)
      Compiled model has JAX function

**class TestGF130JAXExecution**
  Test JAX execution of GF130 models
  - compiled_models(self)
      Compile a few sample GF130 models
  - test_models_produce_output(self, compiled_models)
      GF130 models produce valid JAX output

### scripts/compare_results.py

**parse_value_with_suffix(value_str: str) -> float**
  Parse a numeric value that may have an SI suffix (e.g., '1u' -> 1e-6).

**run_vacask_short_tran(sim_path: Path, vacask_bin: Path, steps: int) -> dict**
  Run VACASK with a short transient and get final voltages.

**run_jaxspice(sim_path: Path, steps: int) -> dict**
  Run JAX-SPICE and return final voltages.

**compare_benchmarks()**
  Compare all benchmarks.

### scripts/compare_vacask.py

**class BenchmarkConfig**
  Configuration for a benchmark.

**find_vacask_binary() -> Optional[Path]**
  Find the VACASK binary (returns absolute path).

**run_vacask(config: BenchmarkConfig, num_steps: int) -> Optional[Tuple[float, float]]**
  Run VACASK and return (time_per_step_ms, wall_time_s).

**run_jax_spice(config: BenchmarkConfig, num_steps: int, use_scan: bool, use_sparse: bool, profile_config: Optional[ProfileConfig], profile_full: bool) -> Tuple[float, float, Dict]**
  Run JAX-SPICE and return (time_per_step_ms, wall_time_s, stats).

**main()** ❌

### scripts/nsys_profile_target.py

**main()** ❌

### scripts/profile_cpu.py

**class BenchmarkResult**
  Results from a single benchmark run

**log(msg, end)**
  Print with flush for real-time output

**get_vacask_benchmarks(names: Optional[List[str]]) -> List[Tuple[str, Path]]**
  Get list of VACASK benchmark .sim files

**run_benchmark(sim_path: Path, name: str, use_sparse: bool, num_steps: int, use_scan: bool) -> BenchmarkResult**
  Run a single benchmark configuration.

**main()** ❌

### scripts/profile_gpu.py

**class BenchmarkResult**
  Results from a single benchmark run.

**class GPUProfiler**
  Profiles VACASK benchmark circuits for CPU/GPU performance analysis
  - start(self)
      Mark the start of profiling
  - stop(self)
      Mark the end of profiling
  - total_time_s(self) -> float ❌
  - run_benchmark(self, sim_path: Path, name: str, use_sparse: bool, num_steps: int, trace_ctx, full: bool, warmup_steps: int) -> BenchmarkResult
      Run a single benchmark configuration.
  - generate_report(self) -> str
      Generate a markdown report for GitHub Actions

**get_vacask_benchmarks(names: Optional[List[str]]) -> List[Tuple[str, Path]]**
  Get list of VACASK benchmark .sim files

**run_single_benchmark(args)**
  Run a single benchmark in subprocess mode and output JSON result.

**run_benchmark_subprocess(name: str, solver: str, timesteps: int, warmup_steps: int, full: bool, cpu: bool) -> Optional[BenchmarkResult]**
  Run a benchmark in a separate subprocess to ensure memory cleanup.

**main()** ❌

### scripts/profile_gpu_cloudrun.py

**run_cmd(cmd: list[str], check: bool, capture: bool) -> subprocess.CompletedProcess**
  Run a command and optionally capture output.

**main()** ❌

### scripts/profile_nsys_cloudrun.py

**run_cmd(cmd: list[str], check: bool, capture: bool) -> subprocess.CompletedProcess**
  Run a command and optionally capture output.

**main()** ❌

### scripts/run_gpu_tests.py

**run_command(cmd: list[str], check: bool) -> subprocess.CompletedProcess**
  Run a command and return the result.

**main()** ❌

### scripts/view_traces.py

**list_trace_files(trace_dir: Path) -> list[Path]**
  List all trace files in a directory.

**get_gcs_path_for_trace(trace_file: Path, cache_dir: Path) -> str | None**
  Get the GCS path for a cached trace file.

**generate_signed_url(gcs_object_path: str, duration: str) -> str | None**
  Generate a signed URL for a GCS object.

**open_perfetto_with_url(trace_url: str) -> None**
  Open Perfetto UI with a trace URL.

**download_from_github(run_id: str | None) -> tuple[Path, str | None]**
  Download traces from GitHub workflow artifact.

**download_from_gcs(gcs_path: str) -> tuple[Path, str | None]**
  Download traces from GCS to cache directory.

**main()** ❌

### tests/conftest.py

**pytest_configure(config)**
  Pytest hook that runs before test collection.

### tests/test_gpu_backend.py

**class TestBackendSelection**
  Tests for automatic backend selection.
  - test_small_circuit_uses_cpu(self)
      Circuits below threshold should use CPU.
  - test_medium_circuit_uses_cpu_without_gpu(self)
      Circuits above threshold but without GPU available should use CPU.
  - test_force_cpu_backend(self)
      Force CPU should always return CPU.
  - test_force_gpu_without_gpu_falls_back(self)
      Force GPU without GPU available should fall back to CPU.
  - test_custom_threshold(self)
      Custom threshold should be respected.

**class TestDeviceSelection**
  Tests for device selection.
  - test_get_cpu_device(self)
      Should be able to get CPU device.
  - test_get_gpu_device_when_available(self)
      Should get GPU device if available.
  - test_get_gpu_device_when_unavailable_raises(self)
      Should raise error if GPU requested but not available.

**class TestDtype**
  Tests for dtype selection.
  - test_default_dtype_is_float64_on_cpu(self)
      CPU should use float64 by default.
  - test_metal_uses_float32(self)
      Metal backend should use float32.

**class TestBackendInfo**
  Tests for backend_info utility.
  - test_backend_info_returns_dict(self)
      Should return dict with expected keys.
  - test_backend_info_threshold(self)
      Should report default threshold.

**class TestGPUDCSolver**
  Tests for GPU-native DC solver.
  - test_dc_gpu_simple_resistor_divider(self)
      Test GPU DC solver with simple resistor divider.
  - test_dc_gpu_matches_cpu(self)
      GPU and CPU solvers should give same results.

**class TestTransientBackend**
  Tests for transient analysis with backend selection.
  - test_transient_jit_with_backend_param(self)
      Transient analysis should accept backend parameter.

**resistor_eval(voltages, params, context)**
  Resistor evaluation function.

**capacitor_eval(voltages, params, context)**
  Capacitor evaluation function.

**vsource_eval(voltages, params, context)**
  Voltage source evaluation function (DC only).

### tests/test_homotopy.py

**class TestHomotopyConfig**
  Tests for HomotopyConfig dataclass.
  - test_default_config(self)
      Test default configuration values match VACASK defaults.
  - test_custom_config(self)
      Test custom configuration.

**class TestHomotopyResult**
  Tests for HomotopyResult dataclass.
  - test_result_fields(self)
      Test result dataclass fields.

**class TestSimpleCircuits**
  Tests with simple circuits to validate homotopy algorithms.
  - test_gmin_stepping_resistor_divider(self)
      Test GMIN stepping with a simple resistor voltage divider.
  - test_source_stepping_resistor(self)
      Test source stepping with a simple resistor circuit.
  - test_homotopy_chain_simple_circuit(self)
      Test the full homotopy chain with a simple circuit.

**class TestDifficultCircuits**
  Tests with circuits that are difficult to converge without homotopy.
  - test_near_singular_circuit(self)
      Test a circuit with near-singular Jacobian at initial guess.

**class TestAdaptiveStepAdjustment**
  Tests for adaptive step adjustment in homotopy algorithms.
  - test_gmin_factor_increases_on_fast_convergence(self)
      Verify factor increases when convergence is fast.
  - test_source_step_scales_adaptively(self)
      Verify source step adapts based on convergence.

### tests/test_transient.py

**class TestDCOperatingPoint**
  Test DC operating point analysis
  - test_voltage_divider(self)
      Test simple voltage divider

**class TestTransientRC**
  Test transient analysis on RC circuits
  - test_rc_charging(self)
      Test RC circuit charging from 0V to 5V

**class TestMNASystem**
  Test MNA system construction
  - test_create_system(self)
      Test creating MNA system
  - test_add_device(self)
      Test adding devices to system

**resistor_eval(voltages, params, context)**
  Resistor evaluation function

**capacitor_eval(voltages, params, context)**
  Capacitor evaluation function

**vsource_eval(voltages, params, context)**
  Voltage source evaluation function (DC only for now)

### tests/test_vacask_benchmarks.py

**class TestRCBenchmark**
  Test RC circuit benchmark (resistor + capacitor)
  - sim_path(self)
      Get RC benchmark sim path
  - test_parse(self, sim_path)
      Test RC benchmark parses correctly
  - test_transient_dense(self, sim_path)
      Test RC transient with dense solver
  - test_transient_sparse(self, sim_path)
      Test RC transient with sparse solver
  - test_rc_time_constant(self, sim_path)
      Verify RC time constant behavior

**class TestGraetzBenchmark**
  Test Graetz bridge benchmark (full-wave rectifier with diodes)
  - sim_path(self)
      Get Graetz benchmark sim path
  - test_parse(self, sim_path)
      Test Graetz benchmark parses correctly
  - test_transient_dense(self, sim_path)
      Test Graetz transient with dense solver
  - test_transient_sparse(self, sim_path)
      Test Graetz transient with sparse solver

**class TestMulBenchmark**
  Test multiplier circuit benchmark
  - sim_path(self)
      Get mul benchmark sim path
  - test_parse(self, sim_path)
      Test mul benchmark parses correctly
  - test_transient_dense(self, sim_path)
      Test mul transient with dense solver

**class TestRingBenchmark**
  Test ring oscillator benchmark (PSP103 MOSFETs)
  - sim_path(self)
      Get ring benchmark sim path
  - test_parse(self, sim_path)
      Test ring benchmark parses correctly
  - test_transient_dense(self, sim_path)
      Test ring transient with dense solver
  - test_transient_sparse(self, sim_path)
      Test ring transient with sparse solver

**class TestC6288Benchmark**
  Test c6288 large benchmark (sparse solver only)
  - sim_path(self)
      Get c6288 benchmark sim path
  - test_parse(self, sim_path)
      Test c6288 benchmark parses correctly
  - test_transient_sparse(self, sim_path)
      Test c6288 transient with sparse solver.

**class TestNodeCountComparison**
  Test that JAX-SPICE node counts match VACASK.
  - find_vacask_binary() -> Path | None
      Find VACASK simulator binary.
  - get_vacask_node_count(vacask_bin: Path, benchmark: str, timeout: int) -> int
      Run VACASK on benchmark and extract node count from 'print stats'.
  - vacask_bin(self)
      Get VACASK binary path, skip if not available.
  - test_node_count_matches_vacask(self, vacask_bin, benchmark, xfail_reason)
      Compare JAX-SPICE node count with VACASK for benchmarks.
  - test_c6288_node_count(self, vacask_bin)
      Test c6288 node count - this is the main target for node collapse fix.

**class TestNodeCollapseStandalone**
  Test node collapse without requiring VACASK binary.
  - test_c6288_node_collapse_reduces_count(self)
      Test that node collapse significantly reduces c6288 node count.
  - test_ring_node_collapse(self)
      Test that node collapse is applied to ring benchmark.

**class BenchmarkSpec**
  Specification for a benchmark comparison test.

**class TestVACASKResultComparison**
  Compare JAX-SPICE simulation results against VACASK reference.
  - vacask_bin(self)
      Get VACASK binary path, skip if not available.
  - test_transient_matches_vacask(self, vacask_bin, benchmark_name)
      Parametrized test comparing JAX-SPICE to VACASK for each benchmark.

**get_benchmark_sim(name: str) -> Path**
  Get path to benchmark .sim file

**find_vacask_binary() -> Path | None**
  Find VACASK simulator binary.

**run_vacask_simulation(vacask_bin: Path, sim_path: Path, t_stop: float, dt: float) -> dict**
  Run VACASK and parse the .raw file output.

**compare_waveforms(vacask_time: np.ndarray, vacask_voltage: np.ndarray, jax_times: np.ndarray, jax_voltage: np.ndarray) -> dict**
  Compare two voltage waveforms.

### tests/test_vacask_jax.py

**class TestResistorSim**
  Tests based on vendor/VACASK/test/test_resistor.sim
  - sim_data(self)
      Parse the sim file and compile the model
  - test_parses_correctly(self, sim_data)
      Sim file parses without error
  - test_resistor_current(self, sim_data)
      Resistor produces correct current for V=1V, R=2k
  - test_mfactor_total_current(self, sim_data)
      With mfactor=3, total current = 3 * V/R = 1.5mA

**class TestDiodeSim**
  Tests based on vendor/VACASK/test/test_diode.sim
  - sim_data(self)
      Parse the sim file and compile the model
  - test_parses_correctly(self, sim_data)
      Sim file parses without error
  - test_model_params(self, sim_data)
      Model parameters are parsed correctly
  - test_diode_compiles(self, sim_data)
      Diode model compiles to JAX function

**class TestCapacitorSim**
  Tests based on vendor/VACASK/test/test_capacitor.sim
  - sim_data(self) ❌
  - test_parses_correctly(self, sim_data)
      Sim file parses without error

**class TestInductorSim**
  Tests based on vendor/VACASK/test/test_inductor.sim
  - sim_data(self) ❌
  - test_parses_correctly(self, sim_data)
      Sim file parses without error

**class TestOpSim**
  Tests based on vendor/VACASK/test/test_op.sim
  - test_parses_correctly(self)
      Sim file parses without error

**parse_embedded_python(sim_path: Path) -> dict**
  Extract expected values from embedded Python test script.

**parse_si_value(s: str) -> float**
  Parse SI-suffixed value like '2k' -> 2000.0

### tests/test_vacask_sim_parser.py

**class CompiledVAModel**
  A Verilog-A model compiled to JAX via openvaf_jax
  - name(self) -> str ❌
  - nodes(self) ❌
  - param_names(self) ❌
  - param_kinds(self) ❌
  - build_inputs(self, voltages: dict, params: dict, temperature: float) -> list
      Build input array for the JAX function
  - evaluate(self, inputs: list)
      Evaluate the JAX function

**class TestParseVACASKSimFiles**
  Test that we can parse VACASK .sim test files
  - test_parse_test_resistor(self)
      Parse test_resistor.sim
  - test_parse_test_diode(self)
      Parse test_diode.sim

**class TestOsdiToVaMapping**
  Test that we can map OSDI filenames to VA sources
  - test_resistor_mapping(self)
      resistor.osdi maps to resistor.va
  - test_capacitor_mapping(self)
      capacitor.osdi maps to capacitor.va
  - test_diode_mapping(self)
      diode.osdi maps to diode.va

**class TestCompileVACASKModels**
  Test that VACASK models compile with openvaf_jax
  - test_compile_resistor(self)
      Compile resistor.va to JAX
  - test_compile_capacitor(self)
      Compile capacitor.va to JAX
  - test_compile_diode(self)
      Compile diode.va to JAX

**class TestVACASKResistorSim**
  Test that replicates VACASK test_resistor.sim using openvaf_jax
  - resistor_model(self)
      Compile VACASK resistor model
  - test_resistor_current(self, resistor_model)
      Test resistor current calculation matches VACASK expected values
  - test_resistor_with_mfactor(self, resistor_model)
      Test resistor with mfactor=3 (parallel instances)
  - test_resistor_jacobian(self, resistor_model)
      Test that Jacobian (conductance) is correct

**class TestFullVACASKTestResistor**
  Full integration test parsing and running test_resistor.sim
  - test_parse_compile_and_evaluate(self)
      Parse test_resistor.sim, compile models, and evaluate

**osdi_to_va_path(osdi_name: str) -> Path**
  Map an OSDI filename to its Verilog-A source file

### tests/test_vacask_suite.py

**class TestVACASKDiscovery**
  Test that we can discover and parse VACASK test files.
  - test_discover_sim_files(self)
      Should find VACASK .sim test files.
  - test_categorize_all_tests(self)
      Categorize all discovered tests.

**class TestVACASKParsing**
  Test that we can parse all VACASK .sim files.
  - test_parse_netlist(self, sim_file)
      Each sim file should parse without error.
  - test_extract_expectations(self, sim_file)
      Should extract expected values from embedded Python.

**class TestVACASKOperatingPoint**
  Run DC operating point tests from VACASK suite.
  - test_test_op(self)
      Test test_op.sim - resistor voltage divider.
  - test_test_resistor(self)
      Test test_resistor.sim - basic resistor with mfactor.
  - test_test_ctlsrc(self)
      Test test_ctlsrc.sim - controlled sources (VCCS, VCVS).
  - test_test_capacitor_op(self)
      Test test_capacitor.sim DC operating point.
  - test_test_visrc(self)
      Test test_visrc.sim - voltage and current sources with mfactor.
  - test_test_inductor_op(self)
      Test test_inductor.sim DC operating point.

**class TestVACASKTransient**
  Run transient analysis tests from VACASK suite.
  - test_test_tran(self)
      Test test_tran.sim - RC transient response.

**class TestVACASKSweep**
  Run DC sweep tests from VACASK suite.
  - test_diode_dc_sweep(self)
      Test DC sweep of a simple diode circuit.

**class TestVACASKSubcircuit**
  Test subcircuit support.
  - test_subcircuit_parsing(self)
      Test that we can parse subcircuits from VACASK files.
  - test_subcircuit_instantiation(self)
      Test instantiating a subcircuit in a circuit.

**discover_benchmark_dirs() -> List[Path]**
  Find all benchmark directories with runme.sim files.

**discover_sim_files() -> List[Path]**
  Find all .sim test files in VACASK test directory.

**parse_embedded_python(sim_path: Path) -> Dict[str, Any]**
  Extract expected values from embedded Python test script.

**parse_analysis_commands(sim_path: Path) -> List[Dict]**
  Extract analysis commands from control block.

**get_required_models(sim_path: Path) -> List[str]**
  Extract model types required by the sim file.

**categorize_test(sim_path: Path) -> Tuple[str, List[str]]**
  Categorize a test and return (category, skip_reasons).

**get_test_ids()**
  Generate test IDs from sim file names.

**resistor_eval(voltages, params, context)**
  Resistor evaluation function matching VACASK resistor.va

**vsource_eval(voltages, params, context)**
  Voltage source evaluation function using large conductance method.

**isource_eval(voltages, params, context)**
  Current source evaluation function.

**pulse_vsource_eval(voltages, params, context, time)**
  Pulse voltage source for transient analysis.

**inductor_eval(voltages, params, context)**
  Inductor evaluation function.

**capacitor_eval(voltages, params, context)**
  Capacitor evaluation function.

**diode_eval(voltages, params, context)**
  Diode evaluation function implementing Shockley equation.

**vccs_eval(voltages, params, context)**
  Voltage-Controlled Current Source.

**vcvs_eval(voltages, params, context)**
  Voltage-Controlled Voltage Source.

**parse_si_value(s: str) -> float**
  Parse a value with SI suffix (e.g., '2k' -> 2000).

### tests/test_vectorized_mna.py

**class TestVectorizedBatchFunctions**
  Test the individual batch functions
  - test_resistor_batch(self)
      Test resistor_batch produces correct I = V/R
  - test_vsource_batch(self)
      Test vsource_batch produces correct enforcement current
  - test_resistor_batch_minimum_resistance(self)
      Test that tiny resistances are clamped for numerical stability

**class TestDeviceGrouping**
  Test device grouping logic
  - test_group_by_type(self)
      Test that devices are correctly grouped by type
  - test_node_indices_shape(self)
      Test that node_indices array has correct shape

**class TestVectorizedVsScalar**
  Compare vectorized and scalar implementations
  - test_simple_circuit(self)
      Test that vectorized produces same results as scalar for simple circuit
  - test_multiple_voltage_sources(self)
      Test with multiple voltage sources
  - test_ground_terminal_handling(self)
      Test that ground terminal stamps are handled correctly

**make_resistor_eval()**
  Create a resistor evaluation function

**make_vsource_eval(V_dc)**
  Create a voltage source evaluation function
