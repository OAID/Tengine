#!/bin/bash

if [ $# -gt 1 ]
then
	TENGINE_INCLUDE_PATH=$1
	TENGINE_LIB_PATH=$2
	ANDROID_NDK=$3
	ABI=$4
	API_LEVEL=$5
else
ANDROID_NDK="/root/sf/android-ndk-r16"
ABI="armeabi-v7a"
API_LEVEL=22

TENGINE_INCLUDE_PATH=
TENGINE_LIB_PATH=
fi

mkdir build
cd build

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
     -DANDROID_ABI="${ABI}" \
     -DANDROID_PLATFORM=android-${API_LEVEL} \
     -DANDROID_STL=c++_shared \
     -DTENGINE_INCLUDE=${TENGINE_INCLUDE_PATH} \
	 -DTENGINE_LIB=${TENGINE_LIB_PATH} \
     -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE \
     ..

make -j4 && make install
