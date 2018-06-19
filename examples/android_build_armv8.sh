#!/bin/bash

export ANDROID_NDK=/home/usr/android-ndk-r16

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
     -DANDROID_ABI="arm64-v8a" \
     -DANDROID_PLATFORM=android-21 \
     -DANDROID_STL=gnustl_shared \
     -DTENGINE_DIR=/home/usr/tengine \
     -DOpenCV_DIR=/home/usr/OpenCV-android-sdk/sdk/native/jni \
     -DPROTOBUF_DIR=/home/usr/protobuf_lib \
     ..

