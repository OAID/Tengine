function init_test()
{
   export TENGINE_CONFIG_FILE=${ROOT_DIR}/install/etc/tengine/config
   echo "face 0: x0,y0 169.23715 84.18719  x1,y1 205.29367  134.34718" > /tmp/master.dummy
   echo "face 1: x0,y0 42.22129 84.22765  x1,y1 84.91341  148.80046" >> /tmp/master.dummy
   echo "face 2: x0,y0 290.14917 102.54037  x1,y1 324.89871  151.54451" >> /tmp/master.dummy
   echo "face 3: x0,y0 376.13626 51.77087  x1,y1 464.53513  144.84897" >> /tmp/master.dummy
   echo "total detected: 4 faces" >> /tmp/master.dummy

   return 0
}

function run_test()
{
    cd ${ROOT_DIR}/build/examples/caffe_wrapper/mtcnn
    echo ./CAFFE_MTCNN ${ROOT_DIR}/tests/images/mtcnn_face4.jpg ${ROOT_DIR}/models wrapper_result4.jpg 

    ./CAFFE_MTCNN ${ROOT_DIR}/tests/images/mtcnn_face4.jpg ${ROOT_DIR}/models wrapper_result4.jpg \
    | grep face > /tmp/result.dummy
    return 0
}

function check_result()
{
    line1=`cat /tmp/result.dummy | wc -l`
    line2=`cat /tmp/master.dummy | wc -l`
    if [ $line1 != $line2 ]
    then
        return 0
    fi
    X0=`awk 'NR==1{print $4}' /tmp/result.dummy`
    Y0=`awk 'NR==1{print $5}' /tmp/result.dummy`
    X1=`awk 'NR==1{print $7}' /tmp/result.dummy`
    Y1=`awk 'NR==1{print $8}' /tmp/result.dummy`
    if [ $(echo "$X0 < 168"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 170"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 83"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 85"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 204"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 206"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 133"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 > 135"|bc ) -eq 1 ];then return 0 ;fi
    
    X0=`awk 'NR==2{print $4}' /tmp/result.dummy`
    Y0=`awk 'NR==2{print $5}' /tmp/result.dummy`
    X1=`awk 'NR==2{print $7}' /tmp/result.dummy`
    Y1=`awk 'NR==2{print $8}' /tmp/result.dummy`
    if [ $(echo "$X0 < 41"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 43"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 83"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 85"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 83"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 85"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 147"|bc ) -eq 1 ];then return 0 ;fi
     if [ $(echo "$Y1 > 149"|bc ) -eq 1 ];then return 0 ;fi
    X0=`awk 'NR==3{print $4}' /tmp/result.dummy`
    Y0=`awk 'NR==3{print $5}' /tmp/result.dummy`
    X1=`awk 'NR==3{print $7}' /tmp/result.dummy`
    Y1=`awk 'NR==3{print $8}' /tmp/result.dummy`

    if [ $(echo "$X0 < 289"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 291"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 101"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 103"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 323"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 325"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 150"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 > 152"|bc ) -eq 1 ];then return 0 ;fi
    X0=`awk 'NR==4{print $4}' /tmp/result.dummy`
    Y0=`awk 'NR==4{print $5}' /tmp/result.dummy`
    X1=`awk 'NR==4{print $7}' /tmp/result.dummy`
    Y1=`awk 'NR==4{print $8}' /tmp/result.dummy`

    if [ $(echo "$X0 < 375"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 377"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 50"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 52"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 463"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 465"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 143"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 > 145"|bc ) -eq 1 ];then return 0 ;fi

    return 1
}

function cleanup_test()
{
    unset TENGINE_CONFIG_FILE
    rm /tmp/result.dummy
    rm /tmp/master.dummy
}
