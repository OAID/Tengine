# Run caffe classification with the caffe wrapper of Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

Compile caffe classification program with the caffe wrapper, and run it with [Tengine](https://github.com/OAID/Tengine).

## Build examples
```
cd  ${Tengine_ROOT}
make install
```
build as ${TENGINE_ROOT}/examples/readme.md
NOTE: "add_subdirectory(caffe_wrapper)" must be opened in ${TENGINE_ROOT}/examples/CMakeLists.txt

## Run
- go to the directory of the executive programs

    ```
    cd ${Tengine_ROOT}/examples/build/caffe_wrapper/cpp_classification
    ```
- test an image:

    ```
    [usage]: ./classification <deploy.prototxt> <network.caffemodel> <mean.binaryproto> <labels.txt> <img.jpg>
             ./classification_mobilenet <deploy.prototxt> <network.caffemodel> <mean.binaryproto> <labels.txt> <img.jpg>
    ```

`classification` is used to test squeezenet.

`classification_mobilenet` is used to test mobilenet, and uses scale 0.017 as std values for image preprocessing in it.

**`NOTE`**: Old caffe model has to be upgraded using:

```
~/caffe/build/tools/upgrade_net_proto_binary  old.caffemodel new.caffemodel
~/caffe/build/tools/upgrade_net_proto_text  old.prototxt new.prototxt
```

