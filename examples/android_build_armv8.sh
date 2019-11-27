#!/bin/bash

export ANDROID_NDK=/home/usr/android-ndk-r16b

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
     -DANDROID_ABI="arm64-v8a" \
     -DANDROID_PLATFORM=android-22 \
     -DANDROID_STL=c++_shared \
     -DTENGINE_DIR=/root/work/git/tengine_auto/tengine \
     -DPROTOBUF_DIR=/home/usr/protobuf_lib \
     -DBLAS_DIR=/home/usr/openblas_lib \
     -DACL_DIR=/home/usr/acl_lib \
     ..

