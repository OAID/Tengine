#!/usr/bin/env bash

set -ex

yum install -y centos-release-scl
yum install -y devtoolset-7 python3 python3-pip ImageMagick file fuse-libs
# pip3 install -U --user pip
# python3 -m pip install cmake

source /opt/rh/devtoolset-7/enable

PROTOBUF_REPO_DIR=${PROTOBUF_REPO_DIR:-protobuf}
PROTOBUF_BUILD_DIR=build-for-ci
TENGINE_BUILD_DIR=build-for-ci

pushd $PROTOBUF_REPO_DIR
mkdir -p $PROTOBUF_BUILD_DIR
pushd $PROTOBUF_BUILD_DIR
cmake -Dprotobuf_BUILD_TESTS=OFF ../cmake
cmake --build . -j`nproc`
cmake -DCMAKE_INSTALL_PREFIX=install -P cmake_install.cmake
popd
popd

export CMAKE_PREFIX_PATH=$PROTOBUF_REPO_DIR/$PROTOBUF_BUILD_DIR/install
mkdir -p $TENGINE_BUILD_DIR
pushd $TENGINE_BUILD_DIR
cmake -DTENGINE_BUILD_CONVERT_TOOL=ON ..
cmake --build . -j`nproc`
popd

SCRIPT_DIR=`dirname "$0"`

source $SCRIPT_DIR/generate-appimage.sh $TENGINE_BUILD_DIR/tools/convert_tool/convert_tool
