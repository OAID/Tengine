# SSD implementation with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is SSD implementation with [Tengine](https://github.com/OAID/Tengine).


## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb)

Store these files into `${Tengine_ROOT}/models/`
- VGG_VOC0712_SSD_300.caffemodel
- VGG_VOC0712_SSD_300.prototxt


## Build examples
```
cd  ${Tengine_ROOT}
make install
cd  ${Tengine_ROOT}/examples/ssd
cmake .
make
```

## Run

1. run SSD by default
    ```
    cd ${Tengine_ROOT}/examples/ssd
    ./SSD
    ```

2. run ssd with other models and image
    ```
    cd ${Tengine_ROOT}/examples/ssd
    ./SSD -p proto_file -m model_file -i image_file
    ```

