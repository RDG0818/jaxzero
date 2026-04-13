cmake_minimum_required(VERSION 3.18)
project(replay_buffer_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# pybind11 (required) — installed via: pip install pybind11
# Locate the cmake config dir from the active Python environment so this
# works inside conda/virtualenv without manually setting CMAKE_PREFIX_PATH.
# ---------------------------------------------------------------------------
if(NOT DEFINED Python_EXECUTABLE OR Python_EXECUTABLE STREQUAL "")
    find_package(Python COMPONENTS Interpreter REQUIRED)
endif()
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_ret
)
if(NOT _pybind11_ret EQUAL 0)
    message(FATAL_ERROR "Could not locate pybind11 cmake dir. Run: pip install pybind11")
endif()
message(STATUS "pybind11 cmake dir: ${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED HINTS "${PYBIND11_CMAKE_DIR}")

# ---------------------------------------------------------------------------
# NumPy include directory (required for buffer-protocol types in bindings)
# ---------------------------------------------------------------------------

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _numpy_ret
)
if(NOT _numpy_ret EQUAL 0)
    message(FATAL_ERROR "Could not determine NumPy include directory. "
                        "Is NumPy installed in the active Python environment?")
endif()
message(STATUS "NumPy include dir: ${NUMPY_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# CUDA (optional) — enables cudaMallocHost for pinned output buffers.
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(HAS_CUDA TRUE)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} found — pinned memory enabled.")
else()
    set(HAS_CUDA FALSE)
    message(STATUS "CUDA not found — output buffers will use regular malloc.")
endif()

# ---------------------------------------------------------------------------
# Build the extension module
# ---------------------------------------------------------------------------
pybind11_add_module(_replay_buffer_cpp
    csrc/replay_buffer/replay_buffer.cpp
    csrc/replay_buffer/bindings.cpp
)

target_include_directories(_replay_buffer_cpp PRIVATE
    csrc/replay_buffer/
    ${NUMPY_INCLUDE_DIR}
)

target_compile_options(_replay_buffer_cpp PRIVATE
    -O3
    -march=native       # enables AVX2 for vectorised memcpy in gather()
    -ffast-math         # safe for the arithmetic here; no NaN special-casing needed
    -Wall
    -Wextra
)

if(HAS_CUDA)
    target_compile_definitions(_replay_buffer_cpp PRIVATE HAS_CUDA=1)
    target_link_libraries(_replay_buffer_cpp PRIVATE CUDA::cudart)
endif()

# Place the .so directly in the repo root so `import _replay_buffer_cpp` works
# from any script that has the repo on sys.path (Ray actors inherit this).
set_target_properties(_replay_buffer_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
