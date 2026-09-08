cmake_minimum_required(VERSION 3.20)

if(APPLE)
  set(
    CMAKE_OSX_DEPLOYMENT_TARGET "11.0"
    CACHE STRING "Minimum supported macOS version" FORCE
  )
endif()

project(tensordev_native_cpu LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(FATAL_ERROR
    "tensordev-native-cpu currently supports Linux and macOS only")
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "from jax import ffi; print(ffi.include_dir())"
  RESULT_VARIABLE TENSORDEV_JAX_INCLUDE_RESULT
  OUTPUT_VARIABLE TENSORDEV_JAX_INCLUDE_DIR
  ERROR_VARIABLE TENSORDEV_JAX_INCLUDE_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT TENSORDEV_JAX_INCLUDE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Could not locate the JAX XLA FFI headers: "
    "${TENSORDEV_JAX_INCLUDE_ERROR}")
endif()

if(NOT EXISTS "${TENSORDEV_JAX_INCLUDE_DIR}/xla/ffi/api/ffi.h")
  message(FATAL_ERROR
    "JAX reported an invalid FFI include directory: "
    "${TENSORDEV_JAX_INCLUDE_DIR}")
endif()

add_library(_tensordev_native_cpu SHARED cpp/kernels.cc)
target_compile_features(_tensordev_native_cpu PRIVATE cxx_std_17)
target_include_directories(
  _tensordev_native_cpu
  PRIVATE "${TENSORDEV_JAX_INCLUDE_DIR}"
)

set_target_properties(
  _tensordev_native_cpu
  PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
)

install(
  TARGETS _tensordev_native_cpu
  LIBRARY DESTINATION tensordev_native_cpu
  RUNTIME DESTINATION tensordev_native_cpu
)
