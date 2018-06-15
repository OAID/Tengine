# Test memory usages of Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is the description of memory test programs with [Tengine](https://github.com/OAID/Tengine).

Including the following memory test programs:

- vgg16_mem.sh

## Download required models

You may find the model files required by the test programs in [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb).

Please download the necessary model files and store them into `Tengine_ROOT/models`.

Required models:

- vgg16
  - vgg16.prototxt
  - vgg16.caffemodel

## Test
- go to the tengine root directory

    ```
    cd ${Tengine_ROOT}
    ```
- run the test program:

    ```
    [usage]: ./tests/bin/vgg16_mem.sh
    ```
