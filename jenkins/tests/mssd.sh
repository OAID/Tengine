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
	if [ ! -e "${ROOT_DIR}/build/examples/mobilenet_ssd/MSSD" ];then
		echo "example not exist, please cmake to build "
		return 1
	fi
	cd ${ROOT_DIR}/build/examples/mobilenet_ssd
}

function run_test()
{
   ./MSSD
}

function check_result()
{
    while read line
	do
		if [[ ${line} =~ "detect result " ]] 
		then
			arr=($line)
			if [ ${arr[3]} -eq 0 ] 
			then
				return 0;
			fi
		elif [[ $line =~ "%" ]]
		then
			arr=($line)
			NAME=${arr[0]}
			read line 
			arr=($line)
			X0=${arr[1]}
			Y0=${arr[3]}
			X1=${arr[5]}
			Y1=${arr[7]}
#		echo $NAME
			case $NAME in
				car)
                    if [ $(echo "$X0 < 465"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X0 > 475"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y0 < 67"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y0 > 77"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X1 < 683"|bc ) -eq 1 ];then return 0;fi
                    if [ $(echo "$X1 > 693"|bc ) -eq 1 ];then return 0;fi
                    if [ $(echo "$Y1 < 166"|bc ) -eq 1 ];then return 0;fi
                    if [ $(echo "$Y1 > 176"|bc ) -eq 1 ];then return 0;fi
					;;
				bicycle)
                    if [ $(echo "$X0 < 101"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X0 > 111"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y0 < 133"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y0 > 143"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X1 < 570"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X1 > 580"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y1 < 411"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y1 > 420"|bc ) -eq 1 ];then return 0 ;fi
					;;
				dog)
                    if [ $(echo "$X0 < 133"|bc ) -eq 1 ];then return 0 ;fi 
                    if [ $(echo "$X0 > 143"|bc ) -eq 1 ];then return 0 ;fi 
                    if [ $(echo "$Y0 < 204"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y0 > 214"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X1 < 320"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X1 > 329"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y1 < 538"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y1 > 548"|bc ) -eq 1 ];then return 0 ;fi
					;;
				*)
					return 0;;
			esac
		fi
	done
	return 1
}

