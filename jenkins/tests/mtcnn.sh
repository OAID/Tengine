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
	if [ ! -e "${ROOT_DIR}/build/examples/mtcnn/MTCNN" ];then
		echo "example not exist, please cmake to build "
		return 1
	fi
	cd ${ROOT_DIR}/build/examples/mtcnn
}

function run_test()
{
   ./MTCNN
}

function check_result()
{
    while read line
	do
#echo $line
		if [[ ${line} =~ "detected face num:" ]] 
		then
			arr=($line)
			if [ ${arr[3]} -ne 4 ] 
			then
				return 0;
			fi
			break;
		fi
	done
	read line 
	arr=($line)
	if [[ ${arr[0]} =~ "BOX:(" ]];then
		X0=${arr[1]}
		Y0=${arr[3]}
		X1=${arr[5]}
		Y1=${arr[7]}
		#echo $NAME
		#echo $X0 $Y0 $X1 $Y1
        if [ $(echo "$X0 < 164"|bc ) -eq 1 ];then return 0 ;fi 
        if [ $(echo "$X0 > 174"|bc ) -eq 1 ];then return 0 ;fi 
        if [ $(echo "$Y0 < 76"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y0 > 86"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$X1 < 201"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$X1 > 211"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y1 < 130"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y1 > 140"|bc ) -eq 1 ];then return 0 ;fi
        
		read line 
		arr=($line)
		X0=${arr[1]}
		Y0=${arr[3]}
		X1=${arr[5]}
		Y1=${arr[7]}
		if [ $(echo "$X0 < 38"|bc ) -eq 1 ];then return 0 ;fi 
		if [ $(echo "$X0 > 48"|bc ) -eq 1 ];then return 0 ;fi 
        if [ $(echo "$Y0 < 81"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y0 > 91"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$X1 < 80"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$X1 > 90"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y1 < 144"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y1 > 154"|bc ) -eq 1 ];then return 0 ;fi
		read line 
		arr=($line)
		X0=${arr[1]}
		Y0=${arr[3]}
		X1=${arr[5]}
		Y1=${arr[7]}
		if [ $(echo "$X0 < 287"|bc ) -eq 1 ];then return 0 ;fi 
		if [ $(echo "$X0 > 297"|bc ) -eq 1 ];then return 0 ;fi 
        if [ $(echo "$Y0 < 98"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y0 > 108"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$X1 < 319"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$X1 > 329"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y1 < 145"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y1 > 155"|bc ) -eq 1 ];then return 0 ;fi
		read line 
		arr=($line)
		X0=${arr[1]}
		Y0=${arr[3]}
		X1=${arr[5]}
		Y1=${arr[7]}
		if [ $(echo "$X0 < 374"|bc ) -eq 1 ];then return 0 ;fi 
		if [ $(echo "$X0 > 384"|bc ) -eq 1 ];then return 0 ;fi 
        if [ $(echo "$Y0 < 51"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y0 > 61"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$X1 < 455"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$X1 > 465"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y1 < 140"|bc ) -eq 1 ];then return 0 ;fi
        if [ $(echo "$Y1 > 150"|bc ) -eq 1 ];then return 0 ;fi
					
		return 1
	fi

	return 0
}

