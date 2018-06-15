#!/bin/bash

PROTOBUF_PATH=
BLAS_PATH=
ARCH_TYPE=
NDK_PATH=

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
      NDK_PATH=${arr[1]}
   fi
done<android_config.txt

mkdir -p android_pack

if [ "${ARCH_TYPE}" = "ARMv7" ]; then
    cp ${NDK_PATH}/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/libgnustl_shared.so  ./android_pack
    cp ${PROTOBUF_PATH}/arm32_lib/libprotobuf.so  ./android_pack
    cp ${BLAS_PATH}/arm32/lib/libopenblas.so  ./android_pack 
elif [ "${ARCH_TYPE}" = "ARMv8" ]; then
    cp ${NDK_PATH}/sources/cxx-stl/gnu-libstdc++/4.9/libs/arm64-v8a/libgnustl_shared.so  ./android_pack
    cp ${PROTOBUF_PATH}/arm64_lib/libprotobuf.so  ./android_pack 
    cp ${BLAS_PATH}/arm64/lib/libopenblas.so  ./android_pack
else 
  echo "none"
fi

cp build/libtengine.so  ./android_pack  


