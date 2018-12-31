# Lighten_cnn implementation with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is lighten_cnn implementation with [Tengine](https://github.com/OAID/Tengine).

## Download required models
Download the required models and data files from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb).

Store these files into `${Tengine_ROOT}/models/`
- LightenedCNN_B.caffemodel
- LightenedCNN_B.prototxt
- data_16384
- eltwise_fc1_256


## Build examples
```
cd  ${Tengine_ROOT}
make
make install
```
build as ${TENGINE_ROOT}/examples/readme.md

## Run

1. run lighten_cnn by default
    - model dir is `tengine/models/`

    ```
    cd ${Tengine_ROOT}/examples/build/lighten_cnn
    ./LIGHTEN_CNN
    ```

2. run lighten_cnn with model dir specified

    ```
    cd ${Tengine_ROOT}/examples/build/lighten_cnn
    ./LIGHTEN_CNN <model_dir>
    ```
    - `model_dir` is the path of your lighten_cnn models and data files. Under this path, there are 4 files:

      `data_16384` : the input data, 128x128 char

      `eltwise_fc1_256` : the data for comparing, 256 float

      `LightenedCNN_B.caffemodel`

      `LightenedCNN_B.prototxt`

