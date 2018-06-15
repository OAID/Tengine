
function run_test()
{
    cd ${ROOT_DIR}/build/examples/imagenet_classification
    ./Classify -n inception_v4
    return 0
}

SUCCESS_STRING="0.7556 - \"n02123159"
