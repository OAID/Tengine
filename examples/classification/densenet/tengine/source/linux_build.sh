#/bin/bash

if [ $# -gt 1 ]
then
TENGINE_INCLUDE_PATH=$1
TENGINE_LIB_PATH=$2
EMBEDDED_CROSS_ROOT=$3
TOOL_CHAIN_PREFIX=$4
else
EMBEDDED_CROSS_ROOT=""
TOOL_CHAIN_PREFIX=""

TENGINE_INCLUDE_PATH=/home/openailab/tengine/install/include
TENGINE_LIB_PATH=/home/openailab/tengine/install/lib
fi

export PATH=${EMBEDDED_CROSS_ROOT}:${PATH}

mkdir build
cd build

cmake   -DTENGINE_INCLUDE=${TENGINE_INCLUDE_PATH} \
	-DTENGINE_LIB=${TENGINE_LIB_PATH} \
	-DCMAKE_C_COMPILER=${TOOL_CHAIN_PREFIX}gcc \
	-DCMAKE_CXX_COMPILER=${TOOL_CHAIN_PREFIX}g++ \
	..

make -j4 && make install
