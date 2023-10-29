cmake_minimum_required(VERSION 3.20 FATAL_ERROR)


set(CMAKE_C_COMPILER clang-16)
set(CMAKE_CXX_COMPILER clang++-16)
add_link_options($<$<CONFIG:RELEASE>:-s>)
project(lightning LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") # works

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(LLVM_ROOT /usr/local/llvm)
find_package(LLVM 17 REQUIRED)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Found LLVNMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_DIR ${LLVM_ROOT}/lib/cmake/mlir/)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)




set(NO_RTTI "-fno-rtti")
# add_definitions(${NO_RTTI})



include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})

llvm_map_components_to_libnames(llvm_libs core Support)
add_subdirectory(brainfuck-mlir)