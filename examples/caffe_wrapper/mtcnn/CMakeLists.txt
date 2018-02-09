
cmake_minimum_required (VERSION 2.8)
project(CAFFE_MTCNN)

set( TENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../ )
set( TENGINE_LIBS tengine wrapper serializer)
set( CAFFE_MTCNN_SRCS caffe_mtcnn.cpp test_caffe_mtcnn.cpp caffe_mtcnn_utils.cpp )

#flag
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")

#opencv
find_package(OpenCV REQUIRED)

#include
include_directories(${TENGINE_DIR}/wrapper/include
                    ${TENGINE_DIR}/core/include
                    ${TENGINE_DIR}/serializer/include)

#lib
link_directories(   ${TENGINE_DIR}/build/wrapper
                    ${TENGINE_DIR}/build/core
                    ${TENGINE_DIR}/build/serializer
                    ${CMAKE_CURRENT_SOURCE_DIR})
#exe
add_executable(CAFFE_MTCNN ${CAFFE_MTCNN_SRCS})
target_link_libraries(CAFFE_MTCNN ${TENGINE_LIBS} ${OpenCV_LIBS})
