#!/bin/bash -

# set -ex

test_models=(
"./tests/test_model_classification -m squeezenet     -i images/cat.jpg   -g 227,227 -w 104.007,116.669,122.679 -s 1,1,1"
"./tests/test_model_classification -m mobilenet      -i images/cat.jpg   -g 224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017"
"./tests/test_model_classification -m mobilenet_v2   -i images/cat.jpg   -g 224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017"
"./tests/test_model_classification -m googlenet      -i images/cat.jpg   -g 224,224 -w 104.007,116.669,122.679 -s 1,1,1"
"./tests/test_model_classification -m inception_v3   -i images/cat.jpg   -g 395,395 -w 104.007,116.669,122.679 -s 0.0078,0.0078,0.0078"
"./tests/test_model_classification -m inception_v4   -i images/cat.jpg   -g 299,299 -w 104.007,116.669,122.679 -s 0.007843,0.007843,0.007843"
"./tests/test_model_classification -m resnet50       -i images/bike.jpg  -g 224,224 -w 104.007,116.669,122.679 -s 1,1,1"
"./tests/test_model_classification -m mnasnet        -i images/cat.jpg   -g 224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017"
"./tests/test_model_classification -m shufflenet_1xg3 -i images/cat.jpg  -g 224,224 -w 103.940,116.780,123.680 -s 0.017,0.017,0.017"
"./tests/test_model_classification -m shufflenet_v2  -i images/cat.jpg   -g 224,224 -w 103.940,116.780,123.680 -s 0.00392156,0.00392156,0.00392156"
"./tests/test_model_alphapose"
"./tests/test_model_crnn"
"./tests/test_model_efficientdet"
"./tests/test_model_hrnet"
"./tests/test_model_landmark"
"./tests/test_model_mobilefacenet"
"./tests/test_model_mobilenet_ssd"
"./tests/test_model_nanodet_m"
"./tests/test_model_openpose"
"./tests/test_model_retinaface"
"./tests/test_model_ultraface"
"./tests/test_model_unet"
"./tests/test_model_yolact"
"./tests/test_model_yolofastest"
"./tests/test_model_yolov3"
"./tests/test_model_yolov3_tiny"
"./tests/test_model_yolov4"
"./tests/test_model_yolov4_tiny"
"./tests/test_model_yolov5s"
)

for (( i = 0 ; i < ${#test_models[@]} ; i++ ))
do
    echo "----> ${test_models[$i]}"
    echo ${test_models[$i]} | xargs -i sh -c "{}"

    if [ "$?" != 0 ]; then
        echo "failed"
        exit 1
    fi
done
