#!/bin/bash
# Build VACASK simulator from vendored source
#
# Prerequisites (macOS with Homebrew):
#   brew install cmake ninja bison flex suite-sparse boost tomlplusplus llvm@18
#
# Prerequisites (Debian/Ubuntu):
#   sudo apt-get install cmake ninja-build bison flex libsuitesparse-dev libtoml++-dev
#   # Boost 1.88 may need to be built from source - see vendor/VACASK/README.md
#
# OpenVAF-reloaded compiler:
#   Run scripts/build_openvaf.sh first, or
#   Download from https://fides.fe.uni-lj.si/openvaf/download/ (OSDI 0.4 version)
#
# KNOWN ISSUES (macOS):
#   - VACASK requires C++20 and macOS 14+ deployment target for std::format support
#   - VACASK C++ code has forward declaration issues (PTBlockSequence incomplete type)
#     that prevent the simulator from building on macOS with Apple Clang
#   - The OSDI model compilation using OpenVAF works successfully
#   - Full VACASK build requires upstream fixes for macOS compatibility

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VACASK_SRC="$PROJECT_ROOT/vendor/VACASK"
BUILD_DIR="$VACASK_SRC/build"

# Check if VACASK source exists
if [ ! -d "$VACASK_SRC" ]; then
    echo "Error: VACASK source not found at $VACASK_SRC"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Detect platform
case "$(uname -s)" in
    Darwin*)
        PLATFORM="macOS"
        # macOS with Homebrew
        BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"

        # Check for required tools
        for tool in cmake ninja bison; do
            if ! command -v $tool &> /dev/null; then
                echo "Error: $tool not found. Install with: brew install $tool"
                exit 1
            fi
        done

        # Flex needs to be from Homebrew for include files
        FLEX_PREFIX="$BREW_PREFIX/opt/flex"
        if [ ! -d "$FLEX_PREFIX" ]; then
            echo "Error: Homebrew flex not found. Install with: brew install flex"
            exit 1
        fi
        FLEX_INCLUDE_DIR="$FLEX_PREFIX/include"

        # Bison needs to be from Homebrew (system bison is too old)
        BISON_PREFIX="$BREW_PREFIX/opt/bison"
        if [ ! -d "$BISON_PREFIX" ]; then
            echo "Error: Homebrew bison not found. Install with: brew install bison"
            exit 1
        fi
        export PATH="$BISON_PREFIX/bin:$PATH"

        # LLVM 18 location for OpenVAF runtime
        LLVM_PREFIX="$BREW_PREFIX/opt/llvm@18"
        if [ ! -d "$LLVM_PREFIX" ]; then
            echo "Error: LLVM 18 not found. Install with: brew install llvm@18"
            exit 1
        fi
        export LLVM_SYS_181_PREFIX="$LLVM_PREFIX"

        # SuiteSparse location
        if [ -d "$BREW_PREFIX/opt/suite-sparse" ]; then
            SUITESPARSE_DIR="$BREW_PREFIX/opt/suite-sparse"
        elif [ -d "$BREW_PREFIX/opt/suitesparse" ]; then
            SUITESPARSE_DIR="$BREW_PREFIX/opt/suitesparse"
        else
            echo "Error: SuiteSparse not found. Install with: brew install suite-sparse"
            exit 1
        fi

        # Boost location
        BOOST_ROOT="$BREW_PREFIX/opt/boost"
        if [ ! -d "$BOOST_ROOT" ]; then
            echo "Error: Boost not found. Install with: brew install boost"
            exit 1
        fi

        # toml++ location
        TOMLPP_DIR="$BREW_PREFIX/opt/tomlplusplus"
        if [ ! -d "$TOMLPP_DIR" ]; then
            echo "Warning: toml++ not found at $TOMLPP_DIR"
            echo "Install with: brew install tomlplusplus"
            TOMLPP_DIR=""
        fi
        ;;
    Linux*)
        PLATFORM="Linux"
        # Check for required tools
        for tool in cmake bison flex; do
            if ! command -v $tool &> /dev/null; then
                echo "Error: $tool not found"
                exit 1
            fi
        done

        # Use system paths
        SUITESPARSE_DIR="/usr"
        BOOST_ROOT=""  # Let CMake find it
        TOMLPP_DIR=""
        ;;
    *)
        echo "Unsupported platform: $(uname -s)"
        exit 1
        ;;
esac

# Find OpenVAF-reloaded compiler - prefer our built version
OPENVAF_COMPILER=""

# First check our built version
LOCAL_OPENVAF="$PROJECT_ROOT/openvaf-py/vendor/OpenVAF/target/release/openvaf-r"
if [ -x "$LOCAL_OPENVAF" ]; then
    OPENVAF_COMPILER="$LOCAL_OPENVAF"
fi

# Then check system PATH
if [ -z "$OPENVAF_COMPILER" ]; then
    for name in openvaf-r openvaf; do
        if command -v $name &> /dev/null; then
            OPENVAF_COMPILER="$(command -v $name)"
            break
        fi
    done
fi

if [ -z "$OPENVAF_COMPILER" ]; then
    echo "Error: OpenVAF-reloaded compiler not found"
    echo ""
    echo "Build it first with:"
    echo "  bash scripts/build_openvaf.sh"
    echo ""
    echo "Or download from https://fides.fe.uni-lj.si/openvaf/download/"
    exit 1
fi

echo "Found OpenVAF compiler: $OPENVAF_COMPILER"
OPENVAF_DIR="$(dirname "$OPENVAF_COMPILER")"

# Create build directory
mkdir -p "$BUILD_DIR"

# On macOS, we need to patch CMakeLists.txt because it doesn't detect Darwin
# and tries to use Linux-specific Boost settings (requiring Boost 1.88 with system component)
if [ "$PLATFORM" = "macOS" ]; then
    echo ""
    echo "Applying macOS patch to CMakeLists.txt..."

    # Patch the CMakeLists.txt to handle macOS/Darwin properly
    # The original only checks for Windows vs "else" (Linux)
    CMAKELISTS="$VACASK_SRC/CMakeLists.txt"

    # Check if already patched
    if ! grep -q "Darwin" "$CMAKELISTS"; then
        # Create backup
        cp "$CMAKELISTS" "$CMAKELISTS.bak"

        # Patch: Add Darwin/macOS handling before the Linux else block
        # Replace the Linux Boost handling with macOS-compatible version
        sed -i.tmp 's/else()/elseif (\${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")\
    # macOS with Homebrew\
    set(SIM_PLATFORM "Darwin")\
    set(SIM_ARCH "arm64")\
    set(PROGRAM_LIB_DIR "lib\/\${PROGRAM_NAME}")\
    find_package(Boost REQUIRED COMPONENTS filesystem)\
    include_directories("\${Boost_INCLUDE_DIRS}")\
    message(STATUS "Boost include: \${Boost_INCLUDE_DIRS}")\
else()/' "$CMAKELISTS"

        # Also need to remove the strict Boost 1.88 requirement for the remaining else (Linux)
        # and the system component which may not exist
        sed -i.tmp 's/find_package(Boost 1.88 REQUIRED COMPONENTS filesystem process system)/find_package(Boost REQUIRED COMPONENTS filesystem)/' "$CMAKELISTS"

        # Clean up temp files
        rm -f "$CMAKELISTS.tmp"

        echo "Patched $CMAKELISTS for macOS support"
    fi

    # Also patch packaging.cmake to handle Darwin (which doesn't need .deb packaging)
    PACKAGING_CMAKE="$VACASK_SRC/packaging.cmake"
    if ! grep -q "Darwin" "$PACKAGING_CMAKE"; then
        cp "$PACKAGING_CMAKE" "$PACKAGING_CMAKE.bak"

        # Add Darwin/macOS handling - use ZIP packaging like Windows
        sed -i.tmp 's/if(\${SIM_PLATFORM} STREQUAL "Windows")/if(\${SIM_PLATFORM} STREQUAL "Windows" OR \${SIM_PLATFORM} STREQUAL "Darwin")/' "$PACKAGING_CMAKE"

        rm -f "$PACKAGING_CMAKE.tmp"
        echo "Patched $PACKAGING_CMAKE for macOS support"
    else
        echo "CMakeLists.txt already patched for Darwin"
    fi
fi

# Configure
echo ""
echo "Configuring VACASK build..."
echo "  Source: $VACASK_SRC"
echo "  Build:  $BUILD_DIR"
echo "  Platform: $PLATFORM"
echo "  OpenVAF: $OPENVAF_DIR"

CMAKE_ARGS=(
    -G Ninja
    -S "$VACASK_SRC"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DSuiteSparse_DIR="$SUITESPARSE_DIR"
    -DOPENVAF_DIR="$OPENVAF_DIR"
    -DCMAKE_CXX_STANDARD=20
    -DCMAKE_CXX_STANDARD_REQUIRED=ON
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
)

if [ -n "$BOOST_ROOT" ]; then
    CMAKE_ARGS+=("-DBoost_ROOT=$BOOST_ROOT")
fi

if [ -n "$TOMLPP_DIR" ]; then
    CMAKE_ARGS+=("-DTOMLPP_DIR=$TOMLPP_DIR")
fi

# For macOS, we need to work around VACASK's Boost version requirement
if [ "$PLATFORM" = "macOS" ]; then
    # Tell CMake to not be strict about Boost version
    CMAKE_ARGS+=("-DBoost_NO_SYSTEM_PATHS=FALSE")
    CMAKE_ARGS+=("-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
    # Add flex include directory (system flex doesn't have headers)
    if [ -n "$FLEX_INCLUDE_DIR" ]; then
        CMAKE_ARGS+=("-DFLEX_INCLUDE_DIR=$FLEX_INCLUDE_DIR")
    fi
    # Use Homebrew bison (system bison is too old for -Wcounterexamples)
    if [ -n "$BISON_PREFIX" ]; then
        CMAKE_ARGS+=("-DBISON_EXECUTABLE=$BISON_PREFIX/bin/bison")
    fi
fi

echo ""
echo "Running: cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}" 2>&1 || {
    echo ""
    echo "CMake configuration failed."
    echo ""
    if [ "$PLATFORM" = "macOS" ]; then
        echo "macOS-specific issues:"
        echo "  - VACASK requires Boost 1.88 but Homebrew may have newer version"
        echo "  - The VACASK CMakeLists.txt may need patching for macOS support"
        echo ""
        echo "You may need to:"
        echo "  1. Edit vendor/VACASK/CMakeLists.txt to relax Boost version requirement"
        echo "  2. Or build Boost 1.88 from source"
    fi
    echo ""
    echo "Common issues:"
    echo "  - Boost with filesystem/process components required"
    echo "  - SuiteSparse/KLU required (brew install suite-sparse)"
    echo "  - OpenVAF-reloaded compiler required"
    echo ""
    exit 1
}

# Build
echo ""
echo "Building VACASK..."
cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)" || {
    echo ""
    echo "Build failed."
    exit 1
}

# Check result
VACASK_BIN="$BUILD_DIR/simulator/vacask"
if [ -x "$VACASK_BIN" ]; then
    echo ""
    echo "Build successful!"
    echo "VACASK binary: $VACASK_BIN"
    echo ""
    echo "To add to PATH:"
    echo "  export PATH=\"$BUILD_DIR/simulator:\$PATH\""
else
    echo ""
    echo "Build completed but vacask binary not found at expected location."
    echo "Check $BUILD_DIR for output."
fi
