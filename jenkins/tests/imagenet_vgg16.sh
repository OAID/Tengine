
function run_test()
{
    cd ${ROOT_DIR}/build/examples/imagenet_classification
    ./Classify -n vgg16 -i ${ROOT_DIR}/tests/images/bike.jpg
    return 0
}

SUCCESS_STRING="0.4998 - \"n03792782"
