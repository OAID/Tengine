# add examples with metal
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
tengine_example(tm_yolov3_uint8             tm_yolov3_uint8.cpp)
tengine_example(tm_yolov4                   tm_yolov4.cpp)
tengine_example(tm_landmark                 tm_landmark.cpp)
tengine_example(tm_landmark_uint8           tm_landmark_uint8.cpp)


# add examples with opencv
if (${TENGINE_TARGET_PROCESSOR} MATCHES "X86")
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    if(OpenCV_FOUND)
        # macro for adding examples
        macro (tengine_example_cv name file)
            add_executable(${name}
                    "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")
            target_link_libraries(${name} ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
        endmacro()

        tengine_example_cv(tm_yolact        tm_yolact.cpp)
    else()
        message(WARNING "OpenCV not found, some examples won't be built")
    endif()
endif()
