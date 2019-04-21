cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

# Find modules.
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/public
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules_CUDA_fix)

function(copy_header file)
  configure_file(${file} ${CMAKE_BINARY_DIR}/include/rpc/${file} COPYONLY)
endfunction()

set(RPC_SRCS
  Request.cpp
  Response.cpp
  Server.cpp
  )

add_library(rpc STATIC ${RPC_SRCS})
set_property(TARGET rpc PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET rpc PROPERTY CXX_STANDARD 11)
target_compile_options(rpc PUBLIC
  -Wall
  -Wextra
  -Wno-unused-parameter
  -Wno-missing-field-initializers
  -Wno-write-strings
  -Wno-unknown-pragmas
  )

target_include_directories(rpc PUBLIC
  ${CMAKE_BINARY_DIR}/aten/src # provides "ATen/TypeExtendedInterface.h" to ATen.h
  )

# For <c10d/...>
target_include_directories(rpc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# For torch/csrc/utils/hash.h
target_include_directories(rpc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

add_dependencies(rpc caffe2)
add_dependencies(rpc torch)


copy_header(Request.hpp)
copy_header(Response.hpp)
copy_header(Server.hpp)

set(RPC_LIBS caffe2)
list(APPEND RPC_LIBS torch)


target_link_libraries(rpc PUBLIC ${RPC_LIBS})

install(TARGETS rpc DESTINATION lib)

option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

# Install all header files that were prepared in the build directory
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/ DESTINATION include)
