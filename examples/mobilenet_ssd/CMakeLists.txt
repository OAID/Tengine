
cmake_minimum_required (VERSION 2.8)
project(MSSD)
add_definitions(-std=c++11)
set( INSTALL_DIR ${TENGINE_DIR}/install/)
set( TENGINE_LIBS tengine hclcpu)

set( CODE_SRCS mssd.cpp ../common/common.cpp)

#include
include_directories(${INSTALL_DIR}/include 
                    ${TENGINE_DIR}/examples/common )

#lib
link_directories( ${INSTALL_DIR}/lib)

#exe
add_executable(MSSD ${CODE_SRCS})
target_link_libraries(MSSD ${TENGINE_LIBS} ${PROTOBUF_LIB}  ${BLAS_LIB} ${ACL_LIB})

