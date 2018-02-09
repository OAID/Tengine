
cmake_minimum_required (VERSION 2.8)
project(RESNET)

set( TENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ )
set( TENGINE_LIBS tengine executor serializer operator)
set( RESNET_SRCS resnet50.cpp utils.cpp)

#flag
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")

#opencv
#find_package(OpenCV REQUIRED core highgui imgproc imgcodecs )
find_package(OpenCV REQUIRED)

#include
include_directories(${TENGINE_DIR}/core/include 
                    ${TENGINE_DIR}/operator/include 
                    ${TENGINE_DIR}/serializer/include 
                    ${TENGINE_DIR}/include)

#lib
link_directories(   ${TENGINE_DIR}/build/core 
                    ${TENGINE_DIR}/build/executor 
                    ${TENGINE_DIR}/build/serializer 
                    ${TENGINE_DIR}/build/operator
                    ${CMAKE_CURRENT_SOURCE_DIR})
#exe
add_executable(RESNET ${RESNET_SRCS})
target_link_libraries(RESNET ${TENGINE_LIBS} ${OpenCV_LIBS})


