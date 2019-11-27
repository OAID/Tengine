
cmake_minimum_required (VERSION 2.8)
project(Classify)

set( INSTALL_DIR ${TENGINE_DIR}/install/ )
set( TENGINE_LIBS  tengine hclcpu)

set( CODE_SRCS classification.cpp model_config.cpp ../common/common.cpp )
set( BIN_EXE Classify )

#include
include_directories(${INSTALL_DIR}/include
                    ${TENGINE_DIR}/examples/common)

#lib
link_directories(${INSTALL_DIR}/lib)

#exe
add_executable(${BIN_EXE} ${CODE_SRCS})
target_link_libraries(${BIN_EXE} ${PROTOBUF_LIB} ${TENGINE_LIBS}  ${BLAS_LIB} ${ACL_LIB})
