#!/bin/bash

export ANDROID_NDK=/home/usr/android-ndk-r16b

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
     -DANDROID_ABI="armeabi-v7a" \
     -DANDROID_PLATFORM=android-21 \
     -DANDROID_STL=c++_shared \
     -DTENGINE_DIR=/home/usr/tengine \
     -DOpenCV_DIR=/home/usr/opencv/sdk/native/jni \
     -DPROTOBUF_DIR=/home/usr/protobuf_lib \
     ..

