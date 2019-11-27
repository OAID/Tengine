
cmake_minimum_required (VERSION 2.8)
project(YuFaceDetect)

set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS tengine)

set( CODE_SRCS yu_facedetect.cpp)
set( BIN_EXE YuFaceDetect)


#flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall")

#include
include_directories(${INSTALL_DIR}/include
		   ${TENGINE_DIR}/include
		   ${TENGINE_DIR}/core/include
		   ${TENGINE_DIR}/operator/include
                   ${TENGINE_DIR}/examples/common)

#lib
link_directories( ${INSTALL_DIR}/lib )

#exe
add_executable(${BIN_EXE} ${CODE_SRCS})
#if( ANDROID)
target_link_libraries(${BIN_EXE} ${TENGINE_LIBS} ${PROTOBUF_LIB} ${BLAS_LIB} ${ACL_LIB})
   #else()
   #target_link_libraries(${BIN_EXE} ${TENGINE_LIBS})
   #endif()
