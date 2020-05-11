if(OPENCV_EXAMPLE)
	set( CLASSIFICATION_CV classification_cv )
if(ANDROID)
    include_directories(${OpenCV_DIR}/include)
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${CLASSIFICATION_CV} classification_cv.cpp common.cpp)
target_link_libraries(${CLASSIFICATION_CV} tengine ${OpenCV_LIBS})

install(TARGETS ${CLASSIFICATION_CV} DESTINATION examples)
endif()

# macro for adding example
macro (tengine_example name)
    add_executable (${name} ${name}.cpp common.cpp)
    target_link_libraries(${name} tengine)
    install(TARGETS ${name} DESTINATION examples)
endmacro()

tengine_example(faster_rcnn_old_api)
tengine_example(faster_rcnn)
tengine_example(classification_old_api)
tengine_example(classification)
tengine_example(mobilenet_ssd_old_api)
tengine_example(mobilenet_ssd)
tengine_example(mobilenetv2_yolov3)
tengine_example(test_insight_face)
tengine_example(yufacedetect)


install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/synset_words.txt DESTINATION examples)
