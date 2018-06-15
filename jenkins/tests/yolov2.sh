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
	if [ ! -e "${ROOT_DIR}/build/examples/yolov2/YOLOV2" ];then
		echo "example not exist, please cmake to build "
		return 1
	fi
   cd ${ROOT_DIR}/build/examples/yolov2
}

function run_test()
{
   ./YOLOV2
}

function check_result()
{
    while read line
	do
		if [[ $line =~ "%" ]]
		then
			arr=($line)
			NAME=${arr[0]}
			read line 
			arr=($line)
			X0=${arr[1]}
			Y0=${arr[3]}
			X1=${arr[5]}
			Y1=${arr[7]}
#			echo $NAME
#			echo $X0 $Y0 $X1 $Y1
			case $NAME in
				car)
                    if [ $X0 -le 440 ];then return 0 ;fi
                    if [ $X0 -ge 450 ];then return 0 ;fi
                    if [ $Y0 -le 68 ];then return 0 ;fi
                    if [ $Y0 -ge 78 ];then return 0 ;fi
                    if [ $X1 -le 670 ];then return 0 ;fi
                    if [ $X1 -ge 680 ];then return 0 ;fi
                    if [ $Y1 -le 171 ];then return 0 ;fi
                    if [ $Y1 -ge 181 ];then return 0 ;fi
					;;
				dog)
                    if [ $X0 -le 114 ];then return 0 ;fi
                    if [ $X0 -ge 124 ];then return 0 ;fi
                    if [ $Y0 -le 174 ];then return 0 ;fi
                    if [ $Y0 -ge 184 ];then return 0 ;fi
                    if [ $X1 -le 318 ];then return 0 ;fi
                    if [ $X1 -ge 328 ];then return 0 ;fi
                    if [ $Y1 -le 540 ];then return 0 ;fi
                    if [ $Y1 -ge 550 ];then return 0 ;fi
					;;
				bicycle)
                    if [ $X0 -le 116 ];then return 0 ;fi
                    if [ $X0 -ge 126 ];then return 0 ;fi
                    if [ $Y0 -le 138 ];then return 0 ;fi
                    if [ $Y0 -ge 148 ];then return 0 ;fi
                    if [ $X1 -le 552 ];then return 0 ;fi
                    if [ $X1 -ge 562 ];then return 0 ;fi
                    if [ $Y1 -le 435 ];then return 0 ;fi
                    if [ $Y1 -ge 445 ];then return 0 ;fi
					;;
				*)
					return 0;;
			esac
		fi
	done
	if [ ! -n ${arr[0]} ];then return 0;fi
	return 1
}

