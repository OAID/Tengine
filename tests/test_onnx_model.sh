#!/bin/bash -

CONVERT_TOOL=${1:-./tools/convert_tool/convert_tool}

test_model()
{
    echo -e "\n======================================================"
    onnx_model_path="models/"$1".onnx"
    tmfile_model_path="models/"$1".tmfile"
    # prepare tmfile
    $CONVERT_TOOL -f onnx -m ${onnx_model_path} -o ${tmfile_model_path} >log.txt 2>&1
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
test_model yolov5-face-sim 3 384 640
test_model version-RFB-640_sim 3 480 640
test_model mobilenetv2-7_sim 3 224 224
test_model efficientnet_b0-sim 3 224 224
test_model facekeypoints_9_sim 3 96 96
test_model facequality 3 64 64
test_model facerfb 3 320 320
test_model ghostnet-sim 3 224 224
test_model hrnet_w18-sim 3 224 224
test_model landmark_out-sim 1 160 160
test_model RFB_pytorch-sim_face 3 320 320
test_model ultraNet 3 240 320
test_model xception_sim 3 229 229
test_model step-final 3 640 640
test_model eais_face 3 640 960
test_model eais-mnet 3 640 960
test_model eais 3 640 960
test_model eais_retina 3 640 960
test_model facedetect_retinaface_sim 3 384 640
test_model inception-v2-9_sim 3 224 224
test_model efficientnet-lite4-11_sim 224 224 3
test_model facerec 3 112 112
test_model MobileNetSSD_deploy_sim 3 300 300
test_model dense-sim 1 32 277
