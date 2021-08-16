#!/bin/bash -

set -x

test_models=(
" ./tests/test_model_classification -m squeezenet     -i images/cat.jpg   -g 227,227 -w 104.007,116.669,122.679 -s 1,1,1"
" ./tests/test_model_classification -m mobilenet      -i images/cat.jpg   -g 224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017"
" ./tests/test_model_classification -m mobilenet_v2   -i images/cat.jpg   -g 224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017"
" ./tests/test_model_classification -m googlenet      -i images/cat.jpg   -g 224,224 -w 104.007,116.669,122.679 -s 1,1,1"
" ./tests/test_model_classification -m inception_v3   -i images/cat.jpg   -g 395,395 -w 104.007,116.669,122.679 -s 0.0078,0.0078,0.0078"
" ./tests/test_model_classification -m inception_v4   -i images/cat.jpg   -g 299,299 -w 104.007,116.669,122.679 -s 0.007843,0.007843,0.007843"
" ./tests/test_model_classification -m resnet50       -i images/bike.jpg  -g 224,224 -w 104.007,116.669,122.679 -s 1,1,1"
" ./tests/test_model_classification -m mnasnet        -i images/cat.jpg   -g 224,224 -w 104.007,116.669,122.679 -s 0.017,0.017,0.017"
" ./tests/test_model_classification -m shufflenet_1xg3 -i images/cat.jpg  -g 224,224 -w 103.940,116.780,123.680 -s 0.017,0.017,0.017"
" ./tests/test_model_classification -m shufflenet_v2  -i images/cat.jpg   -g 224,224 -w 103.940,116.780,123.680 -s 0.00392156,0.00392156,0.00392156"
)

for (( i = 0 ; i < ${#test_models[@]} ; i++ ))
do
    echo ${test_models[$i]}
    echo ${test_models[$i]} | xargs -i sh -c "{}"

    if [ "$?" != 0 ]; then
        echo "failed"
        exit 1
    fi
done
