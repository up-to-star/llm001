cmake_minimum_required(VERSION 3.20)

set(CMAKE_INSTALL_PREFIX "/home/cyj/study_space/llm001/ai-compiler/mlir/mlir-learn/install")
project(mlir-toy LANGUAGES C CXX)

cmake_policy(SET CMP0116 NEW)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(ENV{PATH} "/home/cyj/study_space/llm001/ai-compiler/mlir/mlir-learn/install/bin:\$ENV{PATH}")

add_compile_options(-g)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

add_executable(mlir-toy main.cpp)

add_subdirectory(include)
add_subdirectory(src)

target_link_libraries(mlir-toy PRIVATE
    MLIRLuminaDialect
    MLIRIR
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
)
