# Run caffe classification with the caffe wrapper of Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

Compile caffe classification program with the caffe wrapper, and run it with [Tengine](https://github.com/OAID/Tengine).

## Build
1. install [Tengine](https://github.com/OAID/Tengine)
2. install opencv
    ```
    sudo apt-get install libopencv-dev
    ```
3. config
    ```
    cd ~/tengine/examples/caffe_wrapper/cpp_classification
    cp etc/config.example etc/config
    ```

4. cmake & make
    ```
    cd ~/tengine/examples/caffe_wrapper/cpp_classification
    cmake .
    make
    ```

## Test
- test an image:

    ```
    [usage]: ./classification <deploy.prototxt> <network.caffemodel> <mean.binaryproto> <labels.txt> <img.jpg>
             ./classification_mobilenet <deploy.prototxt> <network.caffemodel> <mean.binaryproto> <labels.txt> <img.jpg>
    ```
`classification_mobilenet` is used to test mobilenet, and uses scale 0.017 as std values for image preprocessing in it. 

**`NOTE`**: Old caffe model has to be upgraded using:

```
~/caffe/build/tools/upgrade_net_proto_binary  old.caffemodel new.caffemodel
~/caffe/build/tools/upgrade_net_proto_text  old.prototxt new.prototxt
```

