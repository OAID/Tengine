#!/bin/bash 

SYS_ROOT=/home/haitao/workshop/tengine/sysroot/ubuntu_rootfs

export PKG_CONFIG_PATH=${SYS_ROOT}/usr/lib/aarch64-linux-gnu/pkgconfig
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${SYS_ROOT}/usr/lib/pkgconfig


PROTOBUF_LIB=`pkg-config --libs protobuf`
PROTOBUF_CFLAGS=`pkg-config --cflags protobuf`

OPENCV_LIB_PATH=`pkg-config --libs-only-L opencv` 
OPENCV_LIB_SO=`pkg-config --libs-only-l opencv`

OPENCV_LIB="${OPENCV_LIB_PATH} ${OPENCV_LIB_SO}"
OPENCV_CFLAGS=`pkg-config --cflags opencv`


ACL_ROOT=/home/haitao/github/ComputeLibrary

mkdir -p build/
cd build/

cmake -DCONFIG_TENGINE_ROOT=/home/haitao/workshop/tengine \
      -DPROTOBUF_LIB="${PROTOBUF_LIB}"  -DPROTOBUF_CFLAGS="${PROTOBUF_CFLAGS}" \
      -DOPENCV_LIB="${OPENCV_LIB}"  -DOPENCV_CFLAGS="${OPENCV_CFLAGS}" \
      -DACL_ROOT="${ACL_ROOT}" -DCONFIG_ACL_OPENCL=ON \
      -DCMAKE_TOOLCHAIN_FILE=../cmake.toolchains     \
      -DSYS_ROOT=${SYS_ROOT} \
      .. 

echo ""
echo "please go directory build/ and make "

echo $PKG_CONFIG_PATH
echo PROTOBUF_LIB=${PROTOBUF_LIB}
echo OPENCV_LIB=${OPENCV_LIB}
