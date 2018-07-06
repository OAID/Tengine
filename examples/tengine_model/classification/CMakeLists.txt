
cmake_minimum_required (VERSION 2.8)
project(tm_classify)

set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS tengine)
if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm32_lib/libprotobuf.so)
endif()
if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm64_lib/libprotobuf.so)
endif()

if( BLAS_DIR)
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
         set( BLAS_LIB ${BLAS_DIR}/arm32/lib/libopenblas.so)
    endif()
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
         set( BLAS_LIB ${BLAS_DIR}/arm64/lib/libopenblas.so)
    endif()
endif()


set( CODE_SRCS classification.cpp ../../common/common.cpp )
set( BIN_EXE tm_classify )

#flag
#set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")

find_package(OpenCV REQUIRED)

#include
include_directories(${INSTALL_DIR}/include
                    ${TENGINE_DIR}/examples/common)

#lib
link_directories(${INSTALL_DIR}/lib)

#exe
add_executable(${BIN_EXE} ${CODE_SRCS})
if(ANDROID)
   target_link_libraries(${BIN_EXE} ${PROTOBUF_LIB} ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(${BIN_EXE} ${TENGINE_LIBS} ${OpenCV_LIBS})
endif()
