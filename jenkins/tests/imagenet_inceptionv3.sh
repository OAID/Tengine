
function run_test()
{
    cd ${ROOT_DIR}/build/examples/imagenet_classification
    ./Classify -n inception_v3
    return 0
}

SUCCESS_STRING="0.0946 - \"n02123159"
