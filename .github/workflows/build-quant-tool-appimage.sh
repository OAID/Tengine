#!/usr/bin/env bash

set -ex

# yum install -y centos-release-scl
# yum install -y devtoolset-7 python3 python3-pip ImageMagick file fuse-libs
# pip3 install -U --user pip
# python3 -m pip install cmake

source /opt/rh/devtoolset-7/enable

OPENCV_REPO_DIR=${OPENCV_REPO_DIR:-opencv}
OPENCV_BUILD_DIR=build-for-ci

# Build opencv by self to avoid depending on libEGL.so
pushd $OPENCV_REPO_DIR
mkdir -p $OPENCV_BUILD_DIR
pushd $OPENCV_BUILD_DIR
/home/dongdong/actions-runners-x86/test_coverage/help/cmake-3.21.2-linux-x86_64/bin/cmake -DWITH_IPP=OFF -DWITH_GTK=OFF -DWITH_OPENCL=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF ..
/home/dongdong/actions-runners-x86/test_coverage/help/cmake-3.21.2-linux-x86_64/bin/cmake --build . -j`nproc`
/home/dongdong/actions-runners-x86/test_coverage/help/cmake-3.21.2-linux-x86_64/bin/cmake -DCMAKE_INSTALL_PREFIX=install -P cmake_install.cmake
popd
popd

export CMAKE_PREFIX_PATH=$OPENCV_REPO_DIR/$OPENCV_BUILD_DIR/install
TENGINE_BUILD_DIR=build-for-ci
mkdir -p $TENGINE_BUILD_DIR
pushd $TENGINE_BUILD_DIR
/home/dongdong/actions-runners-x86/test_coverage/help/cmake-3.21.2-linux-x86_64/bin/cmake -DTENGINE_BUILD_BENCHMARK=OFF -DTENGINE_BUILD_EXAMPLES=OFF -DTENGINE_BUILD_QUANT_TOOL=ON ..
/home/dongdong/actions-runners-x86/test_coverage/help/cmake-3.21.2-linux-x86_64/bin/cmake --build . -j`nproc`
popd

SCRIPT_DIR=`dirname "$0"`

source $SCRIPT_DIR/generate-appimage.sh $TENGINE_BUILD_DIR/tools/quantize/quant_tool_uint8
