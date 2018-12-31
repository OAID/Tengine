# SSD implementation with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is SSD implementation with [Tengine](https://github.com/OAID/Tengine).


## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb).

Store these files into `${Tengine_ROOT}/models/`
- VGG_VOC0712_SSD_300.caffemodel
- VGG_VOC0712_SSD_300.prototxt


## Build examples
```
cd  ${Tengine_ROOT}
make
make install
```
build as ${TENGINE_ROOT}/examples/readme.md

## Run

1. run SSD by default
    - model files are `tengine/models/VGG_VOC0712_SSD_300.prototxt` and `tengine/models/VGG_VOC0712_SSD_300.caffemodel`
    - test image is `tengine/tests/images/ssd_dog.jpg`
    ```
    cd ${Tengine_ROOT}/examples/build/ssd
    ./SSD
    ```

2. run ssd with other models and image
    ```
    cd ${Tengine_ROOT}/examples/build/ssd
    ./SSD -p proto_file -m model_file -i image_file
    ```

