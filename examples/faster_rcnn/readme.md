# Faster_rcnn implementation with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is faster_rcnn implementation with [Tengine](https://github.com/OAID/Tengine).

## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb)

Store these files into `${Tengine_ROOT}/models/`
- VGG16_faster_rcnn.prototxt
- VGG16_faster_rcnn_final.caffemodel


## Build examples
```
cd  ${Tengine_ROOT}
make install

```
build as ${TENGINE_ROOT}/examples/readme.md


## Run

1. run faster_rcnn by default
    ```
    cd ${Tengine_ROOT}/examples/build/faster_rcnn
	./FASTER_RCNN
    ```

2. run faster_rcnn with other models and image
	```
    cd ${Tengine_ROOT}/examples/build/faster_rcnn
	./FASTER_RCNN -p proto_file -m model_file -i image_file
    ```


## NOTE

The default input size is 400x250, and large target could not be detected.

If need to detect large target, please modify the input size. But this may lead to more memory consumption.

