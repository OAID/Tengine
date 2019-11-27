
cmake_minimum_required (VERSION 2.8)
project(FASTER_RCNN)
link_directories( ${PROTOBUF_DIR}/lib )
add_definitions(-std=c++11)
set( INSTALL_DIR ${TENGINE_DIR}/install )
set( TENGINE_LIBS tengine hclcpu)

#include
include_directories(${INSTALL_DIR}/include 
                    ${TENGINE_DIR}/examples/common) 
#lib
link_directories( ${INSTALL_DIR}/lib )

#exe
set( RESNET_SRCS faster_rcnn.cpp ../common/common.cpp)
add_executable(FASTER_RCNN ${RESNET_SRCS})
#if( ANDROID)
target_link_libraries(FASTER_RCNN ${TENGINE_LIBS} ${PROTOBUF_LIB} ${BLAS_LIB} ${ACL_LIB})
   #else()
   #target_link_libraries(FASTER_RCNN ${TENGINE_LIBS} ${BLAS_LIB})
   #endif()



