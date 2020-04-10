#list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}")
#list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/include")
#list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/operator/include")


# macro for adding test
#macro (tengine_test name)
#    add_executable (test_${name} ${name}.cpp)

#    target_include_directories (test_${name} PRIVATE ${test_inlucde_path})
#    target_link_libraries (test_${name} tengine)

#    add_test (test_${name} test_${name})
#endmacro()


#tengine_test(elu)

list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}")
list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/include")
list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/operator/include")
list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/tests/include")

function(create_target file_list)
  STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" fname ${file_list})
  add_executable(${fname} ${file_list})
  target_include_directories (${fname} PRIVATE ${test_inlucde_path})
  target_link_libraries(${fname} tengine)
  
  add_test(NAME ${fname} COMMAND ${fname})
  install(TARGETS ${fname} DESTINATION tests/unit/)
endfunction()


FILE(GLOB file_list "*.cpp" "*.c")

foreach(file ${file_list})
	create_target(${file})
endforeach()

