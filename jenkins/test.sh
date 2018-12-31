#!/bin/bash

LOCAL_DIR=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/.. && pwd)


if [ -z $TEST_CONFIG_FILE ]
then
    TEST_CONFIG_FILE=${LOCAL_DIR}/core_test.list
fi

TEST_FRAMEWORK=$0


#######################################
if [ ! -d "${ROOT_DIR}/build" ];then
	echo -e "\033[31mDirectory ./build is not exists; please build tengine...\033[0m"
	echo
	exit 1
fi
if [ ! -d "${ROOT_DIR}/build/examples" ];then
	mkdir -p build/examples
	cd build/examples
	cmake -DTENGINE_DIR=${ROOT_DIR} ../../examples/ 
	make -j4
	cd ../../
fi
	# if [ ! -e "${ROOT_DIR}/etc/config" ];then
	# 	cp -r ${ROOT_DIR}/etc/config.example ${ROOT_DIR}/etc/config
	# fi

	

export CONFIG_FILE=${ROOT_DIR}/etc/config
export ROOT_DIR
export LOCAL_DIR

if [ -z $IN_TEST_ALL ]
then
   echo
   echo
   echo -e " \033[33m======================================================="
   echo -e " .................... Testing Start ....................\033[0m"
   echo CONFIG_FILE: $TEST_CONFIG_FILE
fi

#if has no argument, execute all tests in test_config.txt
#if has one argument, it is the test case name, and search TEST_CONFIG file to find the proper script file
#if has two arguments, the first is test case name, the second is test_case_script file name

source ${LOCAL_DIR}/test_func.sh


if [ $# = 0 ]
then 
	mkdir -p build/examples
	cd build/examples
	cmake -DTENGINE_DIR=${ROOT_DIR} ../../examples/ 
	make -j4
	cd ../../

    run_all_tests
elif [ $# = 1 ]
then
    run_single_test $1
elif [ $# = 2 ]
then
   run_single_test $1 $2
fi

test_ret=$?

if [ -z $IN_TEST_ALL ]
then
   if [ $test_ret = 0 ]
   then
      echo
      echo -e "\033[42;37m .................... Test  PASSED ....................\033[0m"
      echo -e "\033[42;37m =======================================================\033[0m "
      echo
   else
      echo
      echo -e "\033[41;37m .................... Test  FAILED ....................\033[0m"
      echo -e "\033[41;37m =======================================================\033[0m "
      echo
   fi
fi

exit $test_ret
