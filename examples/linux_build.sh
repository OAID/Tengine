#/bin/bash

# please do configure

EMBEDDED_CROSS_ROOT=""
TOOL_CHAIN_PREFIX=""

TENGINE_PATH=/path/to/tengine

# end configure

TENGINE_INCLUDE_PATH=$TENGINE_PATH/install/include
TENGINE_LIB_PATH=$TENGINE_PATH/install/lib

do_build()
{
	cur_pwd=`pwd`
	cd $1
	if [ -f "linux_build.sh" ]
	then
		if [ -d "build" ]
		then
			rm -fr build
		fi
		sh linux_build.sh $2 $3 $4 $5
	fi
	cd -
}

all_dir=`ls -R |grep "/source:$"|sed 's/:/ /g'`

for dr in $all_dir
do
	do_build $dr $TENGINE_INCLUDE_PATH $TENGINE_LIB_PATH $EMBEDDED_CROSS_ROOT $TOOL_CHAIN_PREFIX
	if [ "$?" != "0" ]
	then
		echo "build failed"
		exit 1
	fi
done

