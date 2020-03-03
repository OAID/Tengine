# Test serializers with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is the description of serializer test programs with [Tengine](https://github.com/OAID/Tengine).

Including the following serializer test programs:

- Mxnet serializer(test_mxnet_sqz and test_mxnet_mobilenet)
- Onnx serializer(test_onnx_sqz)
- Tensorflow serializer(test_tf_inceptionv3 and test_tf_mobilenet)

## Download required models

You may find the model files required by the test programs in [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb).

Please download the necessary model files and store them into `Tengine_ROOT/models`.

Required models:

- Mxnet serializer
  - squeezenet_v1.1-0000.params
  - squeezenet_v1.1-symbol.json
  - mobilenet-symbol.json
  - mobilenet-0000.params
- Onnx serializer
  - sqz.onnx.model
- Tensorflow serializer
  - inception_v3_2016_08_28_frozen.pb
  - frozen_mobilenet_v1_224.pb

## Run
- go to the tengine root directory

    ```
    cd ${Tengine_ROOT}
    ```
- test an image:

    ```
    [usage]: ./build/tests/bin/test_mxnet_sqz
             ./build/tests/bin/test_mxnet_mobilenet
             ./build/tests/bin/test_onnx_sqz
             ./build/tests/bin/test_tf_inceptionv3
             ./build/tests/bin/test_tf_mobilenet
    ```
