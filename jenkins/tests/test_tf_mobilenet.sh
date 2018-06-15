
function run_test()
{
    cd ${ROOT_DIR}
    ./build/tests/bin/test_tf_mobilenet
    return 0
}

SUCCESS_STRING="0.5246 - \"n02123394"
