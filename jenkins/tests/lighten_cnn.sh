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
	if [ ! -e "${ROOT_DIR}/build/examples/lighten_cnn/LIGHTEN_CNN" ];then
		echo "example not exist, please cmake to build "
		return 1
	fi
	cd ${ROOT_DIR}/build/examples/lighten_cnn
}

function run_test()
{
   ./LIGHTEN_CNN
}

function check_result()
{
    while read line
	do
#echo $line
		if [[ ${line} =~ "maxError is" ]] 
		then
			arr=($line)
        	if [ $(echo "${arr[2]} < 0.0001"|bc ) -eq 1 ];then return 1 ;fi
		fi
	done
					
	return 0
}

