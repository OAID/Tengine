#!/bin/bash 

if [ $# -lt 1 ] || [[ "$1" == "-h" ]]
then
	echo "Usage : e.g $0 default_config/x86_linux.config"
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

if [ -n "${EMBEDDED_CROSS_PATH}" ] ; then
	export PATH=${EMBEDDED_CROSS_PATH}:${PATH}
fi

if [ -n "${EMBEDDED_PKG_CONFIG_PATH}" ]; then
	export PKG_CONFIG_PATH=${EMBEDDED_PKG_CONFIG_PATH}:${PKG_CONFIG_PATH}
	echo $PKG_CONFIG_PATH
fi

CFG_BLAS=OFF
if [[ "$OPEN_BLAS" == "y" ]]
then
	CFG_BLAS=ON
fi

if [ "$ARCH_TYPE" == "Arm64" ]; then
	CONFIG_ARCH_ARM64=ON
elif [ "$ARCH_TYPE" == "Arm32" ]; then
	CONFIG_ARCH_ARM32=ON
fi

BUILD_TENGINE_MODULE=OFF
if [ "$BUILD_ACL" == "y" ] || [ "$BUILD_SERIALIZER" == "y" ]
then
	BUILD_TENGINE_MODULE=ON
fi

if [ "$BUILD_ACL" == "y" ]
then
	ACL_OPEN=ON
else
	ACL_OPEN=OFF
fi

CFG_BUILD_TOOLS=OFF
if [[ "$BUILD_TOOLS" == "y" ]]
then
	CFG_BUILD_TOOLS=ON
fi

TENGINE_ROOT=`pwd`

mkdir build
cd build

cmake -DCMAKE_C_COMPILER=${CROSS_COMPILE}gcc \
      -DCMAKE_CXX_COMPILER=${CROSS_COMPILE}g++ \
      -DCONFIG_ARCH_ARM64=$CONFIG_ARCH_ARM64 \
      -DCONFIG_ARCH_ARM32=$CONFIG_ARCH_ARM32 \
      -DCONFIG_ARCH_ARM8_2=$ARM8_2 \
      -DCONFIG_ARCH_BLAS=$CFG_BLAS \
      -DCONFIG_KERNEL_FP16=$CONFIG_KERNEL_FP16 \
      -DCONFIG_AUTHENICATION=$CFG_AUTH_DEVICE \
      -DCONFIG_ONLINE_REPORT=$CFG_ONLINE_REPORT \
      -DCONFIG_VERSION_POSTFIX=$CFG_VERSION_POSTFIX \
      -DOPENBLAS_LIB_PATH=$OPENBLAS_LIB_PATH \
      -DOPENBLAS_INCLUDE_PATH=$OPENBLAS_INCLUDE_PATH \
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
      -DCONFIG_BUILD_CONVERT_TOOLS=$CFG_BUILD_TOOLS \
      -DCONFIG_OPT_CFLAGS="$CONFIG_OPT_CFLAGS" \
      ..

make -j$CPU_NUMS && make install

if [ "$?" != "0" ]
then
	echo "please read How_To_Build_Linux.md User mumual , also maybe can try delete build"
	exit 1
fi

echo "EMBEDDED_CROSS_PATH=$EMBEDDED_CROSS_PATH" > tengine_build.log
echo "CROSS_COMPILE=$CROSS_COMPILE" >> tengine_build.log
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
