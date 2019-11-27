#!/bin/bash

export ANDROID_NDK=/home/usr/android-ndk-r16b

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
     -DANDROID_ABI="armeabi-v7a" \
     -DANDROID_PLATFORM=android-21 \
     -DANDROID_STL=c++_shared \
     -DTENGINE_DIR=/home/usr/tengine \
     -DPROTOBUF_DIR=/home/usr/protobuf_lib \
     -DBLAS_DIR=/home/usr/openblas_lib \
     -DACL_DIR=/home/usr/acl_lib \
     ..

