
cmake_minimum_required (VERSION 2.8)
project(Benchmark)

set( TENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../ )
set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS tengine protobuf)

#flag
set(CMAKE_CFLAGS "-std=gnu89 -O3 -Wall")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")

#include
include_directories(${INSTALL_DIR}/include)

#lib
link_directories(${INSTALL_DIR}/lib)

find_package(OpenCV REQUIRED)

#exe

add_executable(test_convolution test_convolution.c)
target_link_libraries(test_convolution ${TENGINE_LIBS})

add_executable(bench_sqz bench_sqz.cpp)
target_link_libraries(bench_sqz ${TENGINE_LIBS} ${OpenCV_LIBS})

add_executable(bench_mobilenet bench_mobilenet.cpp)
target_link_libraries(bench_mobilenet ${TENGINE_LIBS} ${OpenCV_LIBS})

add_executable(bench_classification bench_classification.cpp)
target_link_libraries(bench_classification ${TENGINE_LIBS} ${OpenCV_LIBS})
