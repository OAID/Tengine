#!/bin/bash -

test_model()
{
    echo -e "\n======================================================"
    onnx_model_path="models/"$1".onnx"
    tmfile_model_path="models/"$1".tmfile"
    # prepare tmfile
    ./tools/convert_tool/convert_tool -f onnx -m ${onnx_model_path} -o ${tmfile_model_path} >log.txt 2>&1
    # run test
    ./tests/test_model_common -m ${tmfile_model_path} -g $2,$3,$4
    out=$?
    if [ $out -gt 0 ]
    then
        exit 1
    fi
    echo -e "======================================================"
}

test_model ssd-sim 3 300 300
test_model mbv1 3 224 224
test_model mbv3 3 224 224
test_model shufflenet_v2 3 224 224
test_model squeezenet1.0-9 3 224 224
test_model retinanet_sim 3 480 640
test_model centerface_0603_sim 3 384 640
test_model mbv3_sim 3 256 256
test_model last 3 384 640
test_model yolov5-face-sim 3 384 640
test_model version-RFB-640_sim 3 480 640
test_model mobilenetv2-7_sim 3 224 224
