#!/bin/bash

source ./default.config

if [ "${ARCH_TYPE}" = "Arm32" ]; then
	cp ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/armeabi-v7a/libc++_shared.so  $TENGINE_ROOT/install/lib
	if [ -d ${PROTOBUF_LIB_PATH} ]; then
		cp ${PROTOBUF_LIB_PATH}/libprotobuf.so $TENGINE_ROOT/install/lib
	fi
	if [ -d ${ACL_ROOT} ]; then
		cp ${ACL_ROOT}/build_32/libarm_compute_core.so $TENGINE_ROOT/install/lib
		cp ${ACL_ROOT}/build_32/libarm_compute.so $TENGINE_ROOT/install/lib
	fi
elif [ "${ARCH_TYPE}" = "Arm64" ]; then
	cp ${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so $TENGINE_ROOT/install/lib
	if [ -d ${PROTOBUF_LIB_PATH} ]; then
		cp ${PROTOBUF_LIB_PATH}/libprotobuf.so $TENGINE_ROOT/install/lib
	fi
	if [ -d ${ACL_ROOT} ]; then
		cp ${ACL_ROOT}/build_64/libarm_compute_core.so $TENGINE_ROOT/install/lib
		cp ${ACL_ROOT}/build_64/libarm_compute.so $TENGINE_ROOT/install/lib
	fi
fi
