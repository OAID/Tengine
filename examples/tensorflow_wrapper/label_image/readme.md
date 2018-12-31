# Run tensorflow label image program with the tensorflow wrapper of Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

Compile tensorflow label image program with the tensorflow wrapper, and run it with [Tengine](https://github.com/OAID/Tengine).

## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb)

Required models:

- inception_v3_2016_08_28_frozen.pb
- frozen_mobilenet_v1_224.pb
- frozen_resnet50v1.pb

Store these files into `${Tengine_ROOT}/models/`

## Build examples
```
cd  ${Tengine_ROOT}
make install
cd examples
```
build as ${TENGINE_ROOT}/examples/readme.md

## Run
- go to the directory of the executive programs

    ```
    cd ${Tengine_ROOT}/examples/build/tensorflow_wrapper/label_image
    ```
- test an image:

    ```
    [usage]: ./label_image_inceptionv3
             ./label_image_mobilenet
             ./label_image_resnet50
    ```

`label_image_inceptionv3` is used to test inception_v3.

`label_image_mobilenet` is used to test mobilenet.

`label_image_resnet50` is used to test resnet50.


