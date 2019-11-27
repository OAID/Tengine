
cmake_minimum_required (VERSION 2.8)
project(tm_convert)

set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS tengine hclcpu)

set( CODE_SRCS convert_caffe_to_tm.cpp ../../common/common.cpp)
set( BIN_EXE convert_caffe_to_tm )

#flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall")

#include(../../utils.cmake)

#include
include_directories(${INSTALL_DIR}/include
                    ${TENGINE_DIR}/examples/common)

#lib
link_directories(${INSTALL_DIR}/lib)

#exe
add_executable(${BIN_EXE} ${CODE_SRCS})

if( ANDROID)
	target_link_libraries(${BIN_EXE} ${TENGINE_LIBS} ${PROTOBUF_LIB} ${BLAS_LIB} "z" ${ACL_LIB})
else()
	target_link_libraries(${BIN_EXE} ${TENGINE_LIBS} ${BLAS_LIB} ${PROTOBUF_LIB} ${ACL_LIB})
endif()
          
