# Mobilenet_SSD implementation with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is Mobilenet_SSD implementation with [Tengine](https://github.com/OAID/Tengine).


## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb)

Store these files into `${Tengine_ROOT}/models/`
- MobileNetSSD_deploy.caffemodel
- MobileNetSSD_deploy.prototxt


## Build examples
```
cd  ${Tengine_ROOT}
make install
cd  ${Tengine_ROOT}/examples/mobilenet_ssd
cmake .
make
```

## Run

1. run MSSD by default
    - models are in `tengien/models/MobileNetSSD_deploy.prototxt` and `tengien/models/MobileNetSSD_deploy.caffemodel`
    - test image is `tengine/tests/imasge/ssd_dog.jpg`
    ```
    ./MSSD
    ``````

2. run with your model_path and image_path
    ```
    [Usage]: ./build/examples/mobilenet_ssd/MSSD [-h]
   [-p proto_file] [-m model_file] [-i image_file]

    ./MSSD -p mssd.prototxt -m mssd.caffemodel -i img.jpg
    ```

## Reference
https://github.com/chuanqi305/MobileNet-SSD
