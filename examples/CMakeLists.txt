set( CLASSIFICATION classification)
set( CLASSIFICATION_CV classification_cv)
set( CLASSIFICATION_OLD classification_old_api)

set( MOBILENET_SSD mobilenet_ssd )
set( MOBILENET_SSD_OLD mobilenet_ssd_old_api)

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

    target_link_libraries(${CLASSIFICATION_CV} tengine ${OpenCV_LIBS})
    target_link_libraries(ncnn_squeezenet tengine ${OpenCV_LIBS})
    target_link_libraries(ncnn_mobilenetssd tengine ${OpenCV_LIBS})
    target_link_libraries(mobilenet_ssd_cv tengine ${OpenCV_LIBS})

    install(TARGETS ${CLASSIFICATION_CV} DESTINATION examples)
    install(TARGETS ncnn_squeezenet DESTINATION examples)
    install(TARGETS ncnn_mobilenetssd DESTINATION examples)
endif()

add_executable(${CLASSIFICATION_OLD} classification_old_api.cpp common.cpp)

target_link_libraries(${CLASSIFICATION_OLD} tengine)
add_executable(${CLASSIFICATION} classification.cpp common.cpp)
target_link_libraries(${CLASSIFICATION} tengine)

add_executable(${MOBILENET_SSD_OLD} mobilenet_ssd_old_api.cpp common.cpp)
target_link_libraries(${MOBILENET_SSD_OLD} tengine)
add_executable(${MOBILENET_SSD} mobilenet_ssd.cpp common.cpp)
target_link_libraries(${MOBILENET_SSD} tengine)

install(TARGETS ${CLASSIFICATION} DESTINATION examples)
install(TARGETS ${MOBILENET_SSD} DESTINATION examples)
install(TARGETS ${CLASSIFICATION_OLD} DESTINATION examples)
install(TARGETS ${MOBILENET_SSD_OLD} DESTINATION examples)
install(TARGETS ${NCNN_MOBILENET} DESTINATION examples)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/synset_words.txt DESTINATION examples)
