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

if [ -n "${EMBEDDED_CROSS_ROOT}" ] ; then
	export PATH=${EMBEDDED_CROSS_ROOT}:${PATH}
fi

if [ -n "${EMBEDDED_PKG_CONFIG_PATH}" ]; then
	export PKG_CONFIG_PATH=${EMBEDDED_PKG_CONFIG_PATH}:${PKG_CONFIG_PATH}
	echo $PKG_CONFIG_PATH
fi

if [ -n "$OPENBLAS_LIB_PATH" ]
then
OPENBLAS_LIB=$OPENBLAS_LIB_PATH
OPENBLAS_CFLAGS=$OPENBLAS_INCLUDE_PATH
else
OPENBLAS_LIB=`pkg-config --libs openblas`
OPENBLAS_CFLAGS=`pkg-config --cflags openblas`
fi

if [[ "$OPEN_BLAS" == "y" ]]
then
	if [ -z "$OPENBLAS_LIB" ] || [ -z "$OPENBLAS_CFLAGS" ]
	then
		echo "Open Blas is Opened,must configure OPENBLAS_LIB_PATH= OPENBLAS_INCLUDE_PATH= "
		exit 1
	fi
fi

if [ "$ARCH_TYPE" == "Arm64" ]; then
	CONFIG_ARCH_ARM64=y
elif [ "$ARCH_TYPE" == "Arm32" ]; then
	CONFIG_ARCH_ARM32=y
fi

make OPENBLAS_LIB_="$OPENBLAS_LIB" OPENBLAS_CFLAGS="$OPENBLAS_CFLAGS" CONFIG_KERNEL_FP16="$CONFIG_KERNEL_FP16" CONFIG_OPT_CFLAGS="$CONFIG_OPT_CFLAGS" CROSS_COMPILE="$CROSS_COMPILE" CONFIG_ONLINE_REPORT="$CONFIG_ONLINE_REPORT" CONFIG_VERSION_POSTFIX="$CONFIG_VERSION_POSTFIX" CONFIG_ARCH_ARM64="$CONFIG_ARCH_ARM64" CONFIG_ARCH_ARM32="$CONFIG_ARCH_ARM32" CONFIG_ARCH_BLAS=$OPEN_BLAS -j$CPU_NUMS && make install

if [ "$?" != "0" ]
then
	echo "please read How_To_Build_Linux.md User mumual , also maybe can try delete build"
	exit 1
fi

if [ "$BUILD_ACL" == "y" ] || [ "$BUILD_SERIALIZER" == "y" ]
then
	MAKE_TENGINE_MODULE=y
fi

if [ "${MAKE_TENGINE_MODULE}" = "y" ]
then
	if [ "$BUILD_ACL" == "y" ]
	then
		ACL_OPEN=ON
	else
		ACL_OPEN=OFF
	fi

TENGINE_ROOT=`pwd`
export TENGINE_ROOT ACL_OPEN BUILD_SERIALIZER PROTOBUF_LIB_PATH PROTOBUF_INCLUDE_PATH ARCH_TYPE CROSS_COMPILE EMBEDDED_CROSS_ROOT
export ACL_ROOT
export USE_EXTERN_PATH=1 
if [[ "$BUILD_TOOLS" == "y" ]]
then
	export BUILD_TOOLS=ON
fi
cd tengine-module
mkdir build
cd build
../linux_build.sh
make -j$CPU_NUMS && make install
if [ "$?" != "0" ]
then
	echo "please read How_To_Build_Linux.md and User mumual ,also maybe can try delete build"
	exit 1
fi

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
