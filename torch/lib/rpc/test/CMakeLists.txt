
function(rpc_add_test test_src)
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} "${test_src}")
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(${test_name} pthread ${ARGN})
  target_compile_options(${test_name} PRIVATE -Wno-error)
  add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endfunction()

rpc_add_test(ServerTest.cpp rpc)
