if(OPENCV_EXAMPLE)
    if(ANDROID)
        include_directories(${OpenCV_DIR}/include)
    endif()

    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    add_executable(${CLASSIFICATION_CV} classification_cv.cpp common.cpp)
    add_executable(ncnn_squeezenet ncnn_squeezenet.cpp)
    add_executable(ncnn_mobilenetssd ncnn_mobilenetssd.cpp)
    add_executable(mobilenet_ssd_cv mobilenet_ssd_cv.cpp common.cpp)
    add_executable(ncnn_mbv2yolov3 ncnn_mbv2yolov3.cpp)
    add_executable(ncnn_yolact ncnn_yolact.cpp)

    target_link_libraries(${CLASSIFICATION_CV} tengine ${OpenCV_LIBS})
    target_link_libraries(ncnn_squeezenet tengine ${OpenCV_LIBS})
    target_link_libraries(ncnn_mobilenetssd tengine ${OpenCV_LIBS})
    target_link_libraries(mobilenet_ssd_cv tengine ${OpenCV_LIBS})
    target_link_libraries(ncnn_mbv2yolov3 tengine ${OpenCV_LIBS})
    target_link_libraries(ncnn_yolact tengine ${OpenCV_LIBS})

    install(TARGETS ${CLASSIFICATION_CV} DESTINATION examples)
    install(TARGETS ncnn_squeezenet DESTINATION examples)
    install(TARGETS ncnn_mobilenetssd DESTINATION examples)
    install(TARGETS ncnn_mbv2yolov3 DESTINATION examples)
    install(TARGETS ncnn_yolact DESTINATION examples)
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
tengine_example(mobilenetv2_yolov3_old_api)
tengine_example(retinaface_old_api)
tengine_example(yufacedetect_old_api)


install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/synset_words.txt DESTINATION examples)
