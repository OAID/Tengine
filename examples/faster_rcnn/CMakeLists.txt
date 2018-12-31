
cmake_minimum_required (VERSION 2.8)
project(FASTER_RCNN)
link_directories( ${PROTOBUF_DIR}/lib )
add_definitions(-std=c++11)
set( INSTALL_DIR ${TENGINE_DIR}/install )
set( TENGINE_LIBS tengine)
if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm32_lib/libprotobuf.so)
endif()
if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm64_lib/libprotobuf.so)
endif()

set( RESNET_SRCS faster_rcnn.cpp ../common/common.cpp)

if( BLAS_DIR)
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
         set( BLAS_LIB ${BLAS_DIR}/arm32/lib/libopenblas.so)
    endif()
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
         set( BLAS_LIB ${BLAS_DIR}/arm64/lib/libopenblas.so)
    endif()
endif()

#opencv
find_package(OpenCV REQUIRED)

#include
include_directories(${INSTALL_DIR}/include 
                    ${TENGINE_DIR}/examples/common) 
#lib
link_directories( ${INSTALL_DIR}/lib )

#exe
add_executable(FASTER_RCNN ${RESNET_SRCS})
if( ANDROID)
   target_link_libraries(FASTER_RCNN ${TENGINE_LIBS} ${PROTOBUF_LIB} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(FASTER_RCNN ${TENGINE_LIBS} ${OpenCV_LIBS})
endif()



