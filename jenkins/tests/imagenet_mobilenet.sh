
function run_test()
{
    cd ${ROOT_DIR}/build/examples/imagenet_classification
    ./Classify -n mobilenet
    return 0
}

SUCCESS_STRING="8.5976 - \"n02123159"
