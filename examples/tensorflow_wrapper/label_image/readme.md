# Run tensorflow label image program with the tensorflow wrapper of Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

Compile tensorflow label image program with the tensorflow wrapper, and run it with [Tengine](https://github.com/OAID/Tengine).

## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb)

Required models:

- inception_v3_2016_08_28_frozen.pb
- frozen_mobilenet_v1_224.pb

Store these files into `${Tengine_ROOT}/models/`

## Build examples
```
cd  ${Tengine_ROOT}
make install
cd  ${Tengine_ROOT}/examples/tensorflow_wrapper/label_image
cmake .
make
```
## Set tengine config file

export TENGINE_CONFIG_FILE=${Tengine_ROOT}/install/etc/tengine/config

## Test
- go to the directory of the executive programs

    ```
    cd ${Tengine_ROOT}/examples/tensorflow_wrapper/label_image
    ```
- test an image:

    ```
    [usage]: ./label_image_inceptionv3
             ./label_image_mobilenet
    ```

`label_image_inceptionv3` is used to test inception_v3.

`label_image_mobilenet` is used to test mobilenet.

