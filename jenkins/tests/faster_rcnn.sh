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
	if [ ! -e "${ROOT_DIR}/build/examples/faster_rcnn/FASTER_RCNN" ];then
		echo "example not exist, please cmake to build "
		return 1
	fi
	cd ${ROOT_DIR}/build/examples/faster_rcnn
}

function run_test()
{
   ./FASTER_RCNN
}

function check_result()
{
    while read line
	do
#echo $line
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
			#echo $NAME
			#echo $X0 $Y0 $X1 $Y1
			case $NAME in
				car)
                    if [ $(echo "$X0 < 426"|bc ) -eq 1 ];then return 0 ;fi 
                    if [ $(echo "$X0 > 436"|bc ) -eq 1 ];then return 0 ;fi 
                    if [ $(echo "$Y0 < 74"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y0 > 84"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X1 < 686"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X1 > 696"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y1 < 156"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y1 > 166"|bc ) -eq 1 ];then return 0 ;fi
					;;
				dog)
                    if [ $(echo "$X0 < 92"|bc ) -eq 1 ];then return 0 ;fi 
                    if [ $(echo "$X0 > 102"|bc ) -eq 1 ];then return 0 ;fi 
                    if [ $(echo "$Y0 < 218"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y0 > 228"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X1 < 353"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$X1 > 363"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y1 < 541"|bc ) -eq 1 ];then return 0 ;fi
                    if [ $(echo "$Y1 > 551"|bc ) -eq 1 ];then return 0 ;fi
					;;
				*)
					return 0;;
			esac
		fi
	done
	return 1
}

