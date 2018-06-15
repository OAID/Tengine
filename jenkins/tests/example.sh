#this is an example to explain how to write a test script

#  a test can be devided into four functions
#  
#  init_test() : optinal
#  run_test()
#  check_result() : optional: return 1 on test pass
#  cleanup_test() : optional
#
#  ONLY run_test() is mandatory to be defined
#  as for check_result(), if it is just to search a target string,
#  just define the SUCCESS_STRING will work  
#
#  A few predefined variables are:
#   TEST_CASE: test case name
#   ROOT_DIR:  the tengine project root directory
#   LOCAL_DIR: the parent directory of the test scripts
#   
#

function init_test()
{
   echo "100" > /tmp/test.dummy
   return 0
}

function run_test()
{
   ORIGIN=`cat /tmp/test.dummy`
   #NEW=$(( $ORIGIN + 2 )) #fail test
   NEW=$(( $ORIGIN + 1 ))  #pass test

   echo $ORIGIN
   echo $NEW > /tmp/test.dummy

   return 0
}

function check_result()
{
   read ORIGIN
   NEW=`cat /tmp/test.dummy`

   OFF=$(( $NEW - $ORIGIN ))
   
   if [ $OFF == 1 ]
   then
       return 1
   else
       return 0
   fi
}

function cleanup_test()
{
  rm /tmp/test.dummy
}
