#!/bin/bash

if [ -z "$USE_EXTERN_PATH" ]
then
	source ../default.config
fi

if [ $ARCH_TYPE == "Arm64" ]; then
ABI=arm64-v8a
ARM_NEON=OFF
elif [ $ARCH_TYPE == "Arm32" ]; then
ABI=armeabi-v7a
ARM_NEON=ON
fi

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ABI \
    -DANDROID_ARM_NEON=$ARM_NEON \
    -DANDROID_PLATFORM=android-$API_LEVEL \
    -DANDROID_STL=c++_shared \
    -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE \
    -DPROTOBUF_LIB_PATH=$PROTOBUF_LIB_PATH \
    -DPROTOBUF_INCLUDE_PATH=$PROTOBUF_INCLUDE_PATH \
    -DACL_ROOT="${ACL_ROOT}" \
    -DCONFIG_TENGINE_ROOT=$TENGINE_ROOT \
    ..

