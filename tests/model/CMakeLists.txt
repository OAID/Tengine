list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/include")
list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/tests/include")
list (APPEND test_inlucde_path "${CMAKE_SOURCE_DIR}/tests/model/include")


# macro for adding test
macro (tengine_mode_test name)
    add_executable (test_${name} ${name}.cpp model_test.cpp)

    target_include_directories (test_${name} PRIVATE ${test_inlucde_path})
    target_link_libraries (test_${name} tengine)

    add_test (test_${name} test_${name} ${name}.json)
endmacro()


# add each test building task
tengine_mode_test(mobilenet)
