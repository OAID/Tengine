#!/bin/bash

# please do configure
ANDROID_NDK="/root/sf/android-ndk-r16"
ABI="armeabi-v7a"
API_LEVEL=22

TENGINE_PATH=/path/to/tengine

# end configure

TENGINE_INCLUDE_PATH=$TENGINE_PATH/install/include
TENGINE_LIB_PATH=$TENGINE_PATH/install/lib

do_build()
{
	cur_pwd=`pwd`
	cd $1
	if [ -f "android_build.sh" ]
	then
		if [ -d "build" ]
		then
			rm -rf build
		fi
		sh android_build.sh $2 $3 $4 $5 $6
	fi
	cd -
}

all_dir=`ls -R |grep "/source:$"|sed 's/:/ /g'`


for dr in $all_dir
do
	do_build $dr $TENGINE_INCLUDE_PATH $TENGINE_LIB_PATH $ANDROID_NDK $ABI $API_LEVEL
	if [ "$?" != "0" ]
	then
		 echo "build failed"
		 exit 1
	fi
done
