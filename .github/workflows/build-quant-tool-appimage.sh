#!/usr/bin/env bash

set -ex

yum install -y centos-release-scl
yum install -y devtoolset-7 python3 python3-pip ImageMagick file fuse-libs
pip3 install -U --user pip
python3 -m pip install cmake

source /opt/rh/devtoolset-7/enable

yum install -y opencv opencv-devel

TENGINE_BUILD_DIR=build-for-ci
mkdir -p $TENGINE_BUILD_DIR
pushd $TENGINE_BUILD_DIR
cmake -DTENGINE_BUILD_BENCHMARK=OFF -DTENGINE_BUILD_EXAMPLES=OFF -DTENGINE_BUILD_QUANT_TOOL=ON ..
cmake --build . -j`nproc`
popd

SCRIPT_DIR=`dirname "$0"`

source $SCRIPT_DIR/generate-appimage.sh $TENGINE_BUILD_DIR/tools/quantize/quant_tool_uint8
