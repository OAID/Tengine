# Convert caffe model files to tengine model file

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is the implementation of the [Tengine](https://github.com/OAID/Tengine) conversion tool which is used to convert caffe model files to tengine model file.

## Download required caffe models
Download the caffe models from [Tengine model zoo](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q) (psw: hhgc)

## Build examples
```
cd  ${Tengine_ROOT}
make install
```
build as ${TENGINE_ROOT}/examples/readme.md

## Run
- go to the directory of the executive program

    ```
    cd ${Tengine_ROOT}/examples/build/tengine_model/convert
    ```
- usage:

    ```
    [Usage]: convert_caffe_to_tm  [-h] [-p proto_file] [-m model_file] [-o output_tmfile]
    ```

- usage examples:

    ```
    ./convert_caffe_to_tm -p ~/tengine/models/sqz.prototxt -m ~/tengine/models/squeezenet_v1.1.caffemodel 
                          -o ~/tengine/models/squeezenet.tmfile

    ./convert_caffe_to_tm -p ~/tengine/models/mobilenet_deploy.prototxt -m ~/tengine/models/mobilenet.caffemodel 
                          -o ~/tengine/models/mobilenet.tmfile
    ```

