function init_test()
{
   export TENGINE_CONFIG_FILE=${ROOT_DIR}/install/etc/tengine/config
   return 0
}

function run_test()
{
    cd ${ROOT_DIR}/build/examples/tensorflow_wrapper/label_image
    ./label_image_mobilenet
    return 0
}

function cleanup_test()
{
    unset TENGINE_CONFIG_FILE
}

SUCCESS_STRING="0.5246 - \"n02123394"
