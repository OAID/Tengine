#!/bin/bash
if [ $# -lt 1 ] || [[ "$1" == "-h" ]]
then
	echo "Usage : e.g $0 example_config/arm_android_cross.config"
	exit 1
fi

CPU_NUMS=4

if [ -n "$2" ]
then
	CPU_NUMS=$2
fi

source "$1"

if [[ "$BUILD_SERIALIZER" == "y" ]]
then
    if [ -z "$PROTOBUF_INCLUDE_PATH" ] || [ -z "$PROTOBUF_LIB_PATH" ]
    then
        echo "Build Serializer is Opened,must configure PROTOBUF_INCLUDE_PATH= PROTOBUF_LIB_PATH="
        exit 1
    fi
fi

if [[ "$BUILD_ACL" == "y" ]]
then
    if [ -z "$ACL_ROOT" ]
    then
        echo "Build ACL is Opened,must configure ACL_ROOT= "
        exit 1
    fi
fi

if [[ "$OPEN_BLAS" == "y" ]]
then
    if [ -z "$OPENBLAS_LIB_PATH" ] || [ -z "$OPENBLAS_INCLUDE_PATH" ]
    then
        echo "Open Blas is Opened,must configure OPENBLAS_LIB_PATH= OPENBLAS_INCLUDE_PATH= "
        exit 1

    fi
fi

API_LEVEL="android-$API_LEVEL"

ARM8_2=$CONFIG_KERNEL_FP16
if [ "$ARCH_TYPE" == "Arm64" ]; then
ARM64=ON
ARM32=OFF
ABI=arm64-v8a
ARM_NEON=OFF
elif [ "$ARCH_TYPE" == "Arm32" ]; then
ARM64=OFF
ARM32=ON
ABI=armeabi-v7a
ARM_NEON=ON
ARM8_2=OFF
fi

TENGINE_ROOT=`pwd`

mkdir build
cd build

if [ "$BUILD_ACL" == "y" ] || [ "$BUILD_SERIALIZER" == "y" ]
then
	BUILD_TENGINE_MODULE=y
fi


if [ "$BUILD_ACL" == "y" ]
then
	ACL_OPEN=ON
else
	ACL_OPEN=OFF
fi

CFG_BLAS=OFF
if [ "$OPEN_BLAS" == "y" ]
then
	CFG_BLAS=ON
fi

CMAKE=cmake

$CMAKE -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ABI \
    -DCONFIG_ARCH_ARM64=$ARM64 \
    -DCONFIG_ARCH_ARM32=$ARM32 \
    -DCONFIG_ARCH_ARM8_2=$ARM8_2 \
    -DANDROID_ARM_NEON=$ARM_NEON \
    -DCONFIG_ARCH_BLAS=$CFG_BLAS \
    -DCONFIG_KERNEL_FP16=$CONFIG_KERNEL_FP16 \
    -DCONFIG_AUTHENICATION=$CFG_AUTH_DEVICE \
    -DCONFIG_ONLINE_REPORT=$CFG_ONLINE_REPORT \
    -DCONFIG_VERSION_POSTFIX=$CFG_VERSION_POSTFIX \
    -DANDROID_PLATFORM=$API_LEVEL \
    -DANDROID_STL=c++_shared \
    -DOPENBLAS_LIB_PATH=$OPENBLAS_LIB_PATH \
    -DOPENBLAS_INCLUDE_PATH=$OPENBLAS_INCLUDE_PATH \
    -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE\
    -DAUTH_LIB=$AUTH_LIB \
    -DAUTH_HEADER=$AUTH_HEADER \
    -DCONFIG_HCL_RELEASE=$HCL_RELEASE \
    -DCONFIG_AUTHED_FAILED_NOT_WORK=$AUTHED_FAILED_NOT_WORK \
    -DCONFIG_BUILD_TENGINE_MODULE=$BUILD_TENGINE_MODULE \
    -DCONFIG_TENGINE_ROOT=$TENGINE_ROOT \
    -DUSE_EXTERN_PATH=y \
    -DBUILD_SERIALIZER=$BUILD_SERIALIZER \
    -DCONFIG_ACL_OPENCL=$ACL_OPEN \
    -DPROTOBUF_LIB_PATH=$PROTOBUF_LIB_PATH \
    -DPROTOBUF_INCLUDE_PATH=$PROTOBUF_INCLUDE_PATH \
    -DACL_ROOT=$ACL_ROOT \
    -DCONFIG_OPT_CFLAGS="$CONFIG_OPT_CFLAGS" \
    ..

make -j$CPU_NUMS && make install

if [ "$?" != "0" ]
then
	echo "please read How_To_Build_Android.md User mumual, also maybe can try delete build"
	exit 1
fi


echo "ANDROID_NDK=$EMBEDDED_CROSS_PATH" > tengine_build.log
echo "API_LEVEL=$API_LEVEL" >> tengine_build.log
echo "ARCH_TYPE=$ARCH_TYPE" >> tengine_build.log
echo "CONFIG_OPT_CFLAGS=$CONFIG_OPT_CFLAGS" >> tengine_build.log
echo "BUILD_SERIALIZER=$BUILD_SERIALIZER" >> tengine_build.log
echo "PROTOBUF_LIB_PATH=$PROTOBUF_LIB_PATH" >> tengine_build.log
echo "PROTOBUF_INCLUDE_PATH=$PROTOBUF_INCLUDE_PATH" >> tengine_build.log
echo "BUILD_ACL=$BUILD_ACL" >> tengine_build.log
echo "ACL_ROOT=$ACL_ROOT" >> tengine_build.log
echo "BUILD_TOOLS=$BUILD_TOOLS" >> tengine_build.log
echo "OPEN_BLAS=$OPEN_BLAS" >> tengine_build.log
echo "OPENBLAS_LIB_PATH=$OPENBLAS_LIB_PATH" >> tengine_build.log
echo "OPENBLAS_INCLUDE_PATH=$OPENBLAS_INCLUDE_PATH" >> tengine_build.log
echo "Build Sucessed"
