# ResNet50 implementation with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is ResNet50 implementation with [Tengine](https://github.com/OAID/Tengine).

## ResNet Architecture
This implementation is based on ResNet 50-layer (snipped from [paper](https://arxiv.org/abs/1512.03385))

![resnet50 architecture](images/architecture.jpg?raw=true "resnet50 architecture")
## Build
1. install [Tengine](https://github.com/OAID/Tengine)
2. install opencv
    ```
    sudo apt-get install libopencv-dev
    ```
3. config
    ```
    cp etc/config.example etc/config
    ```
    set your Tengine build path in `etc/config`
    ```
    plugin.operator.so = ~/tengine/build/operator/liboperator.so
    plugin.serializer.so = ~/tengine/build/serializer/libserializer.so
    plugin.executor.so = ~/tengine/build/executor/libexecutor.so
    plugin.driver.so = ~/tengine/build/driver/libdriver.so
    ```

4. cmake & make
    ```
    cd ~/tengine/examples/resnet50
    cmake .
    make
    ```


## Test
- test an image:

    ```
    [usage]: ./RESNET <test.jpg>  <model_dir>
    ```
- `model_dir` is the path of your resnet50 models. Under this path, there is caffemodel and prototxt:
    ```
    ├── resnet50.caffemodel
    ├── resnet50.prototxt
    ```

## Benchmark
- test on RK3399 single core A72@1.8GHz
- run one time for warm-up, repeat 15 runs and take the average time.

    |Tengine|Caffe(Openblas)|
    |-------|---------------|
    |816.5ms| 1188.7ms|

## Reference

He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

https://arxiv.org/abs/1512.03385


