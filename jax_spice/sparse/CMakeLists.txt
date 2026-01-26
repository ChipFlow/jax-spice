cmake_minimum_required(VERSION 3.18...3.29)
project(umfpack_jax_cpp LANGUAGES CXX)

# C++17 required for XLA FFI headers
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and nanobind
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

# Find nanobind - used for Python bindings (lighter than pybind11)
# If not found via CMake, use pip-installed version
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Find SuiteSparse/UMFPACK
# On macOS with Homebrew: brew install suite-sparse
# On Ubuntu/Debian: apt install libsuitesparse-dev
find_package(PkgConfig REQUIRED)

# Set PKG_CONFIG_PATH for macOS Homebrew
if(APPLE)
    set(ENV{PKG_CONFIG_PATH} "/opt/homebrew/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()

pkg_check_modules(UMFPACK REQUIRED IMPORTED_TARGET umfpack)

# Get jaxlib include path for XLA FFI headers
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
        "import jaxlib; import os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE JAXLIB_INCLUDE_DIR
)
message(STATUS "jaxlib include dir: ${JAXLIB_INCLUDE_DIR}")

if(NOT EXISTS "${JAXLIB_INCLUDE_DIR}/xla/ffi/api/ffi.h")
    message(FATAL_ERROR "XLA FFI headers not found at ${JAXLIB_INCLUDE_DIR}")
endif()

# Build the Python extension module
nanobind_add_module(umfpack_jax_cpp STABLE_ABI
    umfpack_ffi.cpp
)

target_include_directories(umfpack_jax_cpp PRIVATE
    ${JAXLIB_INCLUDE_DIR}
)

target_link_libraries(umfpack_jax_cpp PRIVATE
    PkgConfig::UMFPACK
)

# Install target
install(TARGETS umfpack_jax_cpp LIBRARY DESTINATION .)
