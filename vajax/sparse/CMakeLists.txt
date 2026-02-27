cmake_minimum_required(VERSION 3.18...3.29)
project(umfpack_jax_cpp LANGUAGES C CXX)

# C++17 required for XLA FFI headers
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and nanobind
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

# If not found via CMake, use pip-installed version
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# =============================================================================
# Find SuiteSparse/UMFPACK
# =============================================================================
# Primary: FetchContent pins SuiteSparse v7.8.3 for reproducible builds.
#          Uses system BLAS (Accelerate on macOS, openblas on Linux).
# Fallback: Set UMFPACK_SYSTEM=1 to use system SuiteSparse libraries instead.

if(DEFINED ENV{UMFPACK_SYSTEM} OR UMFPACK_SYSTEM)
    # =========================================================================
    # System library fallback (pkg-config / manual search)
    # =========================================================================
    message(STATUS "UMFPACK_SYSTEM set: using system SuiteSparse libraries")

    # When UMFPACK_STATIC is set, prefer static libraries for self-contained wheels
    if(DEFINED ENV{UMFPACK_STATIC} OR UMFPACK_STATIC)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
        message(STATUS "Static linking mode: preferring .a libraries")
    endif()

    # Try CMake find_package with CMAKE_PREFIX_PATH
    find_package(UMFPACK QUIET CONFIG)
    find_package(AMD QUIET CONFIG)
    find_package(SuiteSparse_config QUIET CONFIG)
    find_package(CHOLMOD QUIET CONFIG)
    find_package(COLAMD QUIET CONFIG)
    find_package(CAMD QUIET CONFIG)
    find_package(CCOLAMD QUIET CONFIG)

    if(UMFPACK_FOUND AND AMD_FOUND AND SuiteSparse_config_FOUND)
        message(STATUS "Found SuiteSparse via CMake config (CMAKE_PREFIX_PATH)")
        add_library(SuiteSparse::All INTERFACE IMPORTED)
        target_link_libraries(SuiteSparse::All INTERFACE
            SuiteSparse::UMFPACK
            SuiteSparse::AMD
            SuiteSparse::SuiteSparse_config
        )
        if(TARGET SuiteSparse::CHOLMOD)
            target_link_libraries(SuiteSparse::All INTERFACE SuiteSparse::CHOLMOD)
        endif()
        if(TARGET SuiteSparse::COLAMD)
            target_link_libraries(SuiteSparse::All INTERFACE SuiteSparse::COLAMD)
        endif()
        if(TARGET SuiteSparse::CAMD)
            target_link_libraries(SuiteSparse::All INTERFACE SuiteSparse::CAMD)
        endif()
        if(TARGET SuiteSparse::CCOLAMD)
            target_link_libraries(SuiteSparse::All INTERFACE SuiteSparse::CCOLAMD)
        endif()
        if(DEFINED ENV{UMFPACK_STATIC} OR UMFPACK_STATIC)
            find_package(BLAS REQUIRED)
            find_package(LAPACK REQUIRED)
            target_link_libraries(SuiteSparse::All INTERFACE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
        endif()
        set(_UMFPACK_TARGET SuiteSparse::All)
    else()
        # pkg-config (macOS Homebrew)
        find_package(PkgConfig REQUIRED)
        if(APPLE)
            set(ENV{PKG_CONFIG_PATH} "/opt/homebrew/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
        endif()
        pkg_check_modules(UMFPACK_PKG IMPORTED_TARGET umfpack)

        if(UMFPACK_PKG_FOUND)
            message(STATUS "Found UMFPACK via pkg-config")
            set(_UMFPACK_TARGET PkgConfig::UMFPACK_PKG)
        else()
            # Manual search (Ubuntu/Debian)
            find_path(UMFPACK_INCLUDE_DIR umfpack.h
                PATHS /usr/include /usr/local/include /usr/include/suitesparse
                PATH_SUFFIXES suitesparse
            )
            find_library(UMFPACK_LIBRARY umfpack
                PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
            )

            if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
                message(STATUS "Found UMFPACK via manual search: ${UMFPACK_LIBRARY}")
                find_library(AMD_LIBRARY amd
                    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
                )
                find_library(SUITESPARSE_CONFIG_LIBRARY suitesparseconfig
                    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
                )

                add_library(UMFPACK_manual INTERFACE IMPORTED)
                target_include_directories(UMFPACK_manual INTERFACE ${UMFPACK_INCLUDE_DIR})
                target_link_libraries(UMFPACK_manual INTERFACE
                    ${UMFPACK_LIBRARY}
                    ${AMD_LIBRARY}
                    ${SUITESPARSE_CONFIG_LIBRARY}
                )
                set(_UMFPACK_TARGET UMFPACK_manual)
            else()
                message(FATAL_ERROR
                    "UMFPACK not found. Install libsuitesparse-dev (Ubuntu), suite-sparse (Homebrew), "
                    "or unset UMFPACK_SYSTEM to use FetchContent.")
            endif()
        endif()
    endif()
else()
    # =========================================================================
    # FetchContent: build pinned SuiteSparse v7.8.3 from source
    # Uses system BLAS (Accelerate on macOS, openblas on Linux)
    # =========================================================================
    include(FetchContent)

    message(STATUS "Using FetchContent to build SuiteSparse v7.8.3")

    # SuiteSparse needs old cmake_minimum_required compat with CMake 4.x
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

    # Find system BLAS (Accelerate on macOS, openblas on Linux)
    find_package(BLAS REQUIRED)
    message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")

    FetchContent_Declare(
        SuiteSparse
        URL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.8.3.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # SuiteSparse build options: only UMFPACK and its dependencies
    set(SUITESPARSE_ENABLE_PROJECTS "umfpack;amd;colamd;camd;ccolamd;suitesparse_config" CACHE STRING "" FORCE)
    set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "" FORCE)
    set(SUITESPARSE_USE_CUDA OFF CACHE BOOL "" FORCE)
    set(UMFPACK_USE_CHOLMOD OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(SUITESPARSE_DEMOS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(SuiteSparse)

    set(_UMFPACK_TARGET SuiteSparse::UMFPACK)
endif()

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
    ${_UMFPACK_TARGET}
)

# Install target
install(TARGETS umfpack_jax_cpp LIBRARY DESTINATION .)
