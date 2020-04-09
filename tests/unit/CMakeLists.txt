list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}")
list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/include")
list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/operator/include")


# macro for adding test
macro (tengine_test name)
    add_executable (test_${name} ${name}.cpp)

    target_include_directories (test_${name} PRIVATE ${test_inlucde_path})
    target_link_libraries (test_${name} tengine)

    add_test (test_${name} test_${name})
endmacro()


tengine_test(elu)
