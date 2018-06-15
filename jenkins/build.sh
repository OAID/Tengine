#!/bin/bash

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ROOT_DIR=$( cd "$LOCAL_DIR"/.. && pwd)

if [ -d "build" ];then
	echo "build is exist!!"
	make clean
	rm -fr build 
fi

make  -j4 
if [ $? != "0" ];then
	make
else
	echo "build success"
fi
make install
