TEST_CHIP="RK3399"
#TEST_CHIP="RK3288"
#TEST_CHIP="R40"

function run_mobile()
{
    if [ ${TEST_CHIP} = "RK3399" ];then
    ./build/tests/bin/bench_mobilenet -p 4   -f$path
    ./build/tests/bin/bench_mobilenet -p 4,5 -f$path
    ./build/tests/bin/bench_mobilenet -p 2   -f$path
    ./build/tests/bin/bench_mobilenet -p 0,1,2,3 -f$path
    elif [ ${TEST_CHIP} = "RK3288" ];then
    echo  "TEST_RK3288"
    ./build/tests/bin/bench_mobilenet -p 2   -f$path
    ./build/tests/bin/bench_mobilenet -p 0,1,2,3 -f$path
    else
    echo "TEST_R40"
    ./build/tests/bin/bench_mobilenet -p 2   -f$path
    ./build/tests/bin/bench_mobilenet -p 0,1,2,3 -f$path
    fi


}
function run_test()
{
    cd ${ROOT_DIR}
    path=`pwd`
	if [ ! -f $path/repo.log]
    then
       `cat $path/repo.log`
    fi
	
   # echo -e "\n"  >> $path/repo.log
    echo "================Mobilenet Test======================" >> $path/repo.log 
#    echo -e "\n"  >> $path/repo.log    
	export CONV_INT_PRIO=20   
 
    echo "Int8:" >> $path/repo.log
    
    run_mobile 
   
    export CONV_INT_PRIO=2000     
 
    echo "Float:" >> $path/repo.log 
    
    run_mobile

    return 0
}

SUCCESS_STRING="8.5976 - \"n02123159"
