
cmake_minimum_required (VERSION 2.8)
project(MTCNN)

set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS tengine hclcpu)

set( MTCNN_SRCS mtcnn_utils.cpp mtcnn.cpp test_mtcnn.cpp ../common/common.cpp )

#include
include_directories(${INSTALL_DIR}/include
                    ${TENGINE_DIR}/examples/common)

#lib
link_directories(${INSTALL_DIR}/lib)

#exe
add_executable(MTCNN ${MTCNN_SRCS})
target_link_libraries(MTCNN ${TENGINE_LIBS} ${PROTOBUF_LIB}  ${BLAS_LIB} ${ACL_LIB})

