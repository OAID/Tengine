#!/bin/bash

#  external variable
# 
#  TEST_CASE: name of the test
#  TEST_CONFIG_FILE: config file of test



PASSED="\033[32m [ PASSED ] \033[0m"
FAILED="\033[31m [ FAILED ] \033[0m"

function exist_func()
{
  if [ "$(type -t $1)" = "function" ]
  then 
     return 1
  else
     return 0
  fi
}

function perror()
{
    echo -e "${FAILED} ..... ${TEST_CASE}"
    echo -e "\033[41;37m $1 \033[0m"
    echo  "======================================================="
}


function do_test()
{
     exist_func "init_test"

	 if [ $? == 1 ];then
	 	init_test
                if [ $? == 1 ]; then
	   	perror "$TEST_CASE failed on init phase"
		return 1
                fi
	 fi

     run_test > ${TMPFILE} 2>&1  
     if [ $? != 0 ]; then
         perror "$TEST_CASE failed on execution phase"
         echo "Execution logs ..."
         cat ${TMPFILE}; 
         return 1
     fi

     # check if result is good:
     # one is to use the user defined check_result 
     # the other is to search a target string in output log 
 

     exist_func check_result

     if [ $? == 1  ]; then
         check_result < ${TMPFILE}
     else
         if [ -z "$SUCCESS_STRING" ]
         then
             perror "$TEST_CASE does not set RESULT search method"
             return 1; 
         fi

         check_simple "$SUCCESS_STRING" < ${TMPFILE}
     fi

     if [ $? != 1 ]
     then 
          perror "$TEST_CASE failed on result search"
          echo "Execution logs ..."
          cat ${TMPFILE} 
          return 1
     fi

     echo -e "${PASSED} ..... ${TEST_CASE}"


     exist_func "cleanup_test" || cleanup_test

     return 0
}

function check_simple()
{
	while read line
	do
	   if [[ $line =~ $1 ]]
	   then
	       return 1
           fi
	done 
}


function run_all_tests()
{
    if [ ! -e $TEST_CONFIG_FILE ]
    then
        return 1
    fi

    MASTER_TMP=/tmp/tengine.master.$$.log
    ret=1

    while read line 
    do
       list=( $line )

       if [ -z $list ]; then
            continue
       fi

       tmp_case=${list[0]}
       tmp_script=${list[1]}

       ch=${tmp_case:0:1}

       if [ $ch == '#' ]; then
          continue
       fi

       (export IN_TEST_ALL=1 && $TEST_FRAMEWORK $tmp_case $tmp_script > ${MASTER_TMP}) 

       exec_result=$?

       cat ${MASTER_TMP}
   
       grep FAILED ${MASTER_TMP} > /dev/null

       ret=$?
 
       if [ $ret == 0 ] || [ $exec_result != 0 ] 
       then
           ret=1
           break 
       else
           ret=0
       fi       

    done < $TEST_CONFIG_FILE

    rm ${MASTER_TMP}

    return $ret
}

function  find_test_script()
{
    while read line 
    do
       list=( $line )

       if [ -z $list ]; then continue; fi

       tmp_case=${list[0]}
       tmp_script=${list[1]}
   
       if [ $tmp_case == $TEST_CASE ]
       then
           test_script=${tmp_script}
           break
       fi 


    done < $TEST_CONFIG_FILE
}


function run_single_test()
{
    TEST_CASE=$1

    if [ ! -z "$2" ]
    then
        test_script="$2"
    else
        find_test_script $1
    fi

    if [ -z "$test_script" ]
    then
        perror  "$TEST_CASE is not defined in $TEST_CONFIG_FILE"
        return 1
    fi

	test_script=${LOCAL_DIR}/${test_script}
    if [ ! -e $test_script ]
    then
       perror "file $test_script does not exist"
       return 1
    fi

    source $test_script

    TMPFILE=/tmp/tengine.test.$$.log
	
    do_test

    RET=$?

	if [ -e $TMPFILE ];then   
		rm $TMPFILE
	fi


    return $RET
}
