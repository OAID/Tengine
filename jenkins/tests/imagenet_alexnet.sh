
function run_test()
{
    cd ${ROOT_DIR}/build/examples/imagenet_classification
    ./Classify -n alexnet
    return 0
}

SUCCESS_STRING="0.3094 - \"n02124075"
