
cmake_minimum_required (VERSION 2.8)
project(SSD)

set( INSTALL_DIR ${TENGINE_DIR}/install/)
set( TENGINE_LIBS tengine hclcpu)

#include
include_directories(${INSTALL_DIR}/include 
                    ${TENGINE_DIR}/examples/common )

#lib
link_directories(   ${INSTALL_DIR}/lib)

#exe
set( CODE_SRCS ssd.cpp ../common/common.cpp)
add_executable(SSD ${CODE_SRCS})
#if( ANDROID)
target_link_libraries(SSD ${TENGINE_LIBS} ${PROTOBUF_LIB}  ${BLAS_LIB} ${ACL_LIB})
  #else()
  # target_link_libraries(SSD ${TENGINE_LIBS} ${BLAS_LIB})
  #endif()


