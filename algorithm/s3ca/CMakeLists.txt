cmake_minimum_required(VERSION 3.18)
project(s3ca_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# FFTW3 single-precision. The package is named differently on different
# platforms, so try the most common configurations.
find_library(FFTW3F_LIB
    NAMES fftw3f libfftw3f libfftw3f-3
    REQUIRED
)
find_path(FFTW3F_INCLUDE_DIR fftw3.h REQUIRED)

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(S3CA_SOURCES
    ${SRC_DIR}/autossca.cpp
    ${SRC_DIR}/computefourier.cc
    ${SRC_DIR}/fftw.cc
    ${SRC_DIR}/filters.cc
    ${SRC_DIR}/parameters.cc
    ${SRC_DIR}/utils.cc
)

# ---------------------------------------------------------------------------
# Build the Python extension. Target name must match PYBIND11_MODULE.
# ---------------------------------------------------------------------------
pybind11_add_module(_native
    ${SRC_DIR}/python_bridge.cpp
    ${S3CA_SOURCES}
)

target_include_directories(_native PRIVATE
    ${SRC_DIR}
    ${FFTW3F_INCLUDE_DIR}
)

target_link_libraries(_native PRIVATE ${FFTW3F_LIB})

if(MSVC)
    target_compile_definitions(_native PRIVATE _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(_native PRIVATE /O2 /bigobj)
else()
    target_compile_options(_native PRIVATE -O3)
endif()

# scikit-build-core installs into the python package directory
install(TARGETS _native LIBRARY DESTINATION algorithm/s3ca)