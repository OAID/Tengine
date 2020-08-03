# macro for adding test
macro (tengine_example name file)
    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    target_link_libraries(${name} ${CMAKE_PROJECT_NAME})
    install (TARGETS ${name} DESTINATION bin)
    add_test(NAME ${name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmark COMMAND ${name} )
endmacro()

# add benchmark
tengine_example(tm_benchmark      tm_benchmark.c)
