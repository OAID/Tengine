#!/bin/bash

if [ $# -lt 1 ] || [[ "$1" == "-h" ]]
then
	echo "Usage : e.g $0 example_config/arm_android_cross.config"
	exit 1
fi

source "$1"

mkdir -p android_pack

if [ "${ARCH_TYPE}" = "Arm32" ]; then
    cp ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/libc++_shared.so  ./android_pack
    if [ ${PROTOBUF_LIB_PATH} ]; then
    cp ${PROTOBUF_LIB_PATH}/libprotobuf.so  ./android_pack
    fi
    if [ ${OPENBLAS_LIB_PATH} ]; then
    cp ${OPENBLAS_LIB_PATH}/libopenblas.so  ./android_pack
    fi
    if [ ${ACL_ROOT} ]; then
    cp ${ACL_ROOT}/build_32/libarm_compute_core.so ./android_pack
    cp ${ACL_ROOT}/build_32/libarm_compute.so ./android_pack
    fi 
elif [ "${ARCH_TYPE}" = "Arm64" ]; then
    cp ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so  ./android_pack
    if [ ${PROTOBUF_LIB_PATH} ]; then
    cp ${PROTOBUF_LIB_PATH}/libprotobuf.so  ./android_pack 
    fi
    if [ ${OPENBLAS_LIB_PATH} ]; then
    cp ${OPENBLAS_LIB_PATH}/libopenblas.so  ./android_pack
    fi
    if [ ${ACL_ROOT} ]; then 
    cp ${ACL_ROOT}/build_64/libarm_compute_core.so ./android_pack
    cp ${ACL_ROOT}/build_64/libarm_compute.so ./android_pack
    fi
else 
  echo "none"
fi

cp build/libtengine.so  ./android_pack  
cp build/libhclcpu.so  ./android_pack


