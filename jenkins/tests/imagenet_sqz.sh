
function run_test()
{
    cd ${ROOT_DIR}/build/examples/imagenet_classification
    ./Classify -n squeezenet
    return 0
}

SUCCESS_STRING="0.2763 - \"n02123045"
