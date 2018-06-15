function init_test()
{
   export TENGINE_CONFIG_FILE=${ROOT_DIR}/install/etc/tengine/config
   return 0
}

function run_test()
{
    cd ${ROOT_DIR}/build/examples/caffe_wrapper/cpp_classification
    ./classification ${ROOT_DIR}/models/sqz.prototxt ${ROOT_DIR}/models/squeezenet_v1.1.caffemodel \
    ${ROOT_DIR}/examples/caffe_wrapper/cpp_classification/imagenet_mean.binaryproto \
    ${ROOT_DIR}/models/synset_words.txt ${ROOT_DIR}/tests/images/cat.jpg
    return 0
}

function cleanup_test()
{
    unset TENGINE_CONFIG_FILE
}

SUCCESS_STRING="0.2763 - \"n02123045"
