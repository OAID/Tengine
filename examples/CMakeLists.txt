# macro for adding test
macro (tengine_example name file)
    add_executable(${name}
        "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
        "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")

    target_link_libraries(${name} ${CMAKE_PROJECT_NAME})
    include_directories("${PROJECT_SOURCE_DIR}/examples/common")
    install (TARGETS ${name} DESTINATION bin)
endmacro()

# add examples
tengine_example(tm_classification           tm_classification.c)
tengine_example(tm_classification_uint8     tm_classification_uint8.c)
tengine_example(tm_mobilenet_ssd            tm_mobilenet_ssd.c)
tengine_example(tm_mobilenet_ssd_uint8      tm_mobilenet_ssd_uint8.cpp)
tengine_example(tm_retinaface               tm_retinaface.cpp)
tengine_example(tm_yolov3_tiny              tm_yolov3_tiny.cpp)

