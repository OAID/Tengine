
cmake_minimum_required (VERSION 2.8)
project(CAFFE_MTCNN)

set( TENGINE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../ )
set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS wrapper tengine hclcpu)
set( CAFFE_MTCNN_SRCS test_caffe_mtcnn.cpp caffe_mtcnn.cpp caffe_mtcnn_utils.cpp )

#flag
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall")

#include
include_directories(${INSTALL_DIR}/wrapper_include
                    ${INSTALL_DIR}/include)

#lib
link_directories(${INSTALL_DIR}/lib)

add_executable(CAFFE_MTCNN ${CAFFE_MTCNN_SRCS})
target_link_libraries(CAFFE_MTCNN ${TENGINE_LIBS} boost_thread ${BLAS_LIB} ${ACL_LIB})
