
cmake_minimum_required (VERSION 2.8)
project(classification)

set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS wrapper tengine protobuf boost_thread glog )
set( CODE_SRC classification.cpp )

#flag
#set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -DUSE_OPENCV -DCPU_ONLY")
add_definitions(-DUSE_OPENCV)
add_definitions(-DCPU_ONLY)
#opencv
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")

#include
include_directories(${INSTALL_DIR}/wrapper_include
                    ${INSTALL_DIR}/include)

#lib
link_directories(${INSTALL_DIR}/lib)

#exe
add_executable(classification ${CODE_SRC})
target_link_libraries(classification ${TENGINE_LIBS} ${OpenCV_LIBS})

add_executable(classification_mobilenet ${CODE_SRC})
target_compile_definitions(classification_mobilenet PUBLIC -DMOBILE_NET)
target_link_libraries(classification_mobilenet ${TENGINE_LIBS} ${OpenCV_LIBS})
