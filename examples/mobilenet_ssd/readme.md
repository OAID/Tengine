# Mobilenet_SSD implementation with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is Mobilenet_SSD implementation with [Tengine](https://github.com/OAID/Tengine).


## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q) (psw: hhgc).

Store these files into `${Tengine_ROOT}/models/`
- MobileNetSSD_deploy.caffemodel
- MobileNetSSD_deploy.prototxt


## Build examples
```
cd  ${Tengine_ROOT}
make
make install
```
build as ${TENGINE_ROOT}/examples/readme.md

## Run

1. run MSSD by default
    - model files are `tengine/models/MobileNetSSD_deploy.prototxt` and `tengine/models/MobileNetSSD_deploy.caffemodel`
    - test image is `tengine/tests/images/ssd_dog.jpg`
    ```
    cd ${TENGINE_ROOT}/examples/build/mobilenet_ssd/
    ./MSSD
    ``````

2. run mssd with other models and image
    ```
    [Usage]: ./MSSD [-h]
                    [-p proto_file] [-m model_file] [-i image_file]

    ./MSSD -p mssd.prototxt -m mssd.caffemodel -i img.jpg
    ```

## Reference
https://github.com/chuanqi305/MobileNet-SSD

