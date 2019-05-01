#!/bin/bash

PROTOBUF_PATH=
BLAS_PATH=
ARCH_TYPE=
ANDROID_NDK=
ACL_ROOT=

while read line 
do
   IFS=:
   arr=($line)
   if [ "${arr[0]}" == "PROTOBUF_DIR" ]; then
      PROTOBUF_PATH=${arr[1]}
   elif [ "${arr[0]}" == "BLAS_DIR" ]; then
      BLAS_PATH=${arr[1]}
   elif [ "${arr[0]}" == "CONFIG_ARCH_TYPE" ]; then
      ARCH_TYPE=${arr[1]}
   elif [ "${arr[0]}" == "ANDROID_NDK" ]; then
      ANDROID_NDK=${arr[1]}
   elif [ "${arr[0]}" == "ACL_ROOT" ]; then
      ACL_ROOT=${arr[1]}
   fi
done<../android_config.txt

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" \
    -DCONFIG_ARCH_ARM64=ON \
    -DANDROID_PLATFORM=android-21 \
    -DANDROID_STL=c++_shared \
    -DPROTOBUF_DIR=$PROTOBUF_PATH \
    -DBLAS_DIR=$BLAS_PATH \
    -DACL_ROOT=$ACL_ROOT \
    -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE\
    ..
