function init_test()
{
   export TENGINE_CONFIG_FILE=${ROOT_DIR}/install/etc/tengine/config
   echo "face 0: x0,y0 170.91638 76.79741  x1,y1 208.06985  128.15182" > /tmp/master.dummy
   echo "face 1: x0,y0 104.73532 175.54961  x1,y1 137.31529  236.33575" >> /tmp/master.dummy
   echo "face 2: x0,y0 296.50046 183.12787  x1,y1 334.23788  236.24565" >> /tmp/master.dummy
   echo "face 3: x0,y0 459.72641 123.38977  x1,y1 506.37677  175.55830" >> /tmp/master.dummy
   echo "face 4: x0,y0 66.37959 161.10822  x1,y1 108.99861  243.48486" >> /tmp/master.dummy
   echo "face 5: x0,y0 561.20978 199.41609  x1,y1 587.92566  252.88115" >> /tmp/master.dummy
   echo "total detected: 6 faces" >> /tmp/master.dummy

   return 0
}

function run_test()
{
    cd ${ROOT_DIR}/build/examples/caffe_wrapper/mtcnn
    ./CAFFE_MTCNN ${ROOT_DIR}/tests/images/mtcnn_face6.jpg ${ROOT_DIR}/models wrapper_result6.jpg \
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
    if [ $(echo "$X0 < 169"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 171"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 75"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 77"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 207"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 209"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 127"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 > 129"|bc ) -eq 1 ];then return 0 ;fi

    X0=`awk 'NR==2{print $4}' /tmp/result.dummy`
    Y0=`awk 'NR==2{print $5}' /tmp/result.dummy`
    X1=`awk 'NR==2{print $7}' /tmp/result.dummy`
    Y1=`awk 'NR==2{print $8}' /tmp/result.dummy`
    if [ $(echo "$X0 < 103"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 105"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 174"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 176"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 136"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 138"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 235"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 > 237"|bc ) -eq 1 ];then return 0 ;fi
    X0=`awk 'NR==3{print $4}' /tmp/result.dummy`
    Y0=`awk 'NR==3{print $5}' /tmp/result.dummy`
    X1=`awk 'NR==3{print $7}' /tmp/result.dummy`
    Y1=`awk 'NR==3{print $8}' /tmp/result.dummy`

    if [ $(echo "$X0 < 295"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 297"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 182"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 184"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 333"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 335"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 235"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 > 237"|bc ) -eq 1 ];then return 0 ;fi
    X0=`awk 'NR==4{print $4}' /tmp/result.dummy`
    Y0=`awk 'NR==4{print $5}' /tmp/result.dummy`
    X1=`awk 'NR==4{print $7}' /tmp/result.dummy`
    Y1=`awk 'NR==4{print $8}' /tmp/result.dummy`

    if [ $(echo "$X0 < 458"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 460"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 122"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 124"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 505"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 507"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 174"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 > 176"|bc ) -eq 1 ];then return 0 ;fi
    
    X0=`awk 'NR==5{print $4}' /tmp/result.dummy`
    Y0=`awk 'NR==5{print $5}' /tmp/result.dummy`
    X1=`awk 'NR==5{print $7}' /tmp/result.dummy`
    Y1=`awk 'NR==5{print $8}' /tmp/result.dummy`

    if [ $(echo "$X0 < 65"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 67"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 160"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 162"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 107"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 109"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 242"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 > 244"|bc ) -eq 1 ];then return 0 ;fi

    X0=`awk 'NR==6{print $4}' /tmp/result.dummy`
    Y0=`awk 'NR==6{print $5}' /tmp/result.dummy`
    X1=`awk 'NR==6{print $7}' /tmp/result.dummy`
    Y1=`awk 'NR==6{print $8}' /tmp/result.dummy`

    if [ $(echo "$X0 < 560"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X0 > 562"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 < 198"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y0 > 200"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 < 586"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$X1 > 588"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 < 251"|bc ) -eq 1 ];then return 0 ;fi
    if [ $(echo "$Y1 > 253"|bc ) -eq 1 ];then return 0 ;fi


    return 1
}

function cleanup_test()
{
    unset TENGINE_CONFIG_FILE
    rm /tmp/result.dummy
    rm /tmp/master.dummy
}
