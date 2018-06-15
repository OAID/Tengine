function init_test()
{
   export TENGINE_CONFIG_FILE=${ROOT_DIR}/install/etc/tengine/config
   return 0
}

function run_test()
{
    cd ${ROOT_DIR}/build/examples/caffe_wrapper/cpp_classification
    ./classification_mobilenet ${ROOT_DIR}/models/mobilenet_deploy.prototxt ${ROOT_DIR}/models/mobilenet.caffemodel \
    ${ROOT_DIR}/examples/caffe_wrapper/cpp_classification/imagenet_mean.binaryproto \
    ${ROOT_DIR}/models/synset_words.txt ${ROOT_DIR}/tests/images/cat.jpg
    return 0
}

function cleanup_test()
{
    unset TENGINE_CONFIG_FILE
}

SUCCESS_STRING="8.5976 - \"n02123159"
