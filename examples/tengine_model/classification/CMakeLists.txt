
cmake_minimum_required (VERSION 2.8)
project(tm_classify)

set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS tengine hclcpu)

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

if ( NOT (ARM OR ANDROID))
set (BLAS_LIB openblas)
endif()

set( CODE_SRCS classification.cpp ../../common/common.cpp )
set( BIN_EXE tm_classify )

set( CODE_SRCS_TF classification_tf.cpp ../../common/common.cpp )
set( BIN_EXE_TF tm_classify_tf )

set( CODE_SRCS_INT8 classification_int8.cpp ../../common/common.cpp )
set( BIN_EXE_INT8 tm_classify_int8 )

if(NOT ANDROID AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    add_definitions(-mfp16-format=ieee -mfpu=neon-fp16)
endif()
set( CODE_SRCS_FP16 classification_fp16.cpp ../../common/common.cpp )
set( BIN_EXE_FP16 tm_classify_fp16 )

#flag
#set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")

find_package(OpenCV REQUIRED)

#include
include_directories(${INSTALL_DIR}/include
                    ${TENGINE_DIR}/examples/common
                    ${TENGINE_DIR}/core/include)

#lib
link_directories(${INSTALL_DIR}/lib)

#exe
add_executable(${BIN_EXE} ${CODE_SRCS})
if(ANDROID)
   target_link_libraries(${BIN_EXE} ${PROTOBUF_LIB} ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(${BIN_EXE} ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
endif()

add_executable(${BIN_EXE_TF} ${CODE_SRCS_TF})
if(ANDROID)
   target_link_libraries(${BIN_EXE_TF} ${PROTOBUF_LIB} ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(${BIN_EXE_TF} ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
endif()

add_executable(${BIN_EXE_INT8} ${CODE_SRCS_INT8})
if(ANDROID)
   target_link_libraries(${BIN_EXE_INT8} ${PROTOBUF_LIB} ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(${BIN_EXE_INT8} ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
endif()

add_executable(${BIN_EXE_FP16} ${CODE_SRCS_FP16})
if(ANDROID)
   target_link_libraries(${BIN_EXE_FP16} ${PROTOBUF_LIB} ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(${BIN_EXE_FP16} ${TENGINE_LIBS} ${OpenCV_LIBS} ${BLAS_LIB})
endif()

