cmake_minimum_required(VERSION 3.20)
project(mlir_l1feas)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
set(CMAKE_CXX_STANDARD 17)

add_mlir_tool(mlir-l1feas L1Feas.cpp)
llvm_update_compile_flags(mlir-l1feas)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-l1feas PRIVATE
  ${dialect_libs}
  MLIRAffineAnalysis
  MLIRParser
  MLIRPass
  MLIROptLib
)
