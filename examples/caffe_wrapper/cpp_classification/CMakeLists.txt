
cmake_minimum_required (VERSION 2.8)
project(classification)

set( TENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../ )
set( TENGINE_LIBS tengine wrapper serializer protobuf boost_thread glog)
set( CLASSIFICATION_SRC classification.cpp )
set( CLASSIFICATION_M_SRC classification_mobilenet.cpp )

#flag
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -DUSE_OPENCV -DCPU_ONLY")

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
add_executable(classification ${CLASSIFICATION_SRC})
target_link_libraries(classification ${TENGINE_LIBS} ${OpenCV_LIBS})

add_executable(classification_mobilenet ${CLASSIFICATION_M_SRC})
target_link_libraries(classification_mobilenet ${TENGINE_LIBS} ${OpenCV_LIBS})

