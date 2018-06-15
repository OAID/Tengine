
function run_test()
{
    cd ${ROOT_DIR}/build/examples/imagenet_classification
    ./Classify -n googlenet
    return 0
}

SUCCESS_STRING="0.5009 - \"n02123159"
