
function run_test()
{
    cd ${ROOT_DIR}/build/examples/imagenet_classification
    ./Classify -n googlenet
    return 0
}

SUCCESS_STRING="0.500(8|9) - \"n02123159"
