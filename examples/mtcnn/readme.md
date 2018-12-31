# MTCNN implementation with Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is mtcnn implementation with [Tengine](https://github.com/OAID/Tengine).

## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb).

Store these files into `${Tengine_ROOT}/models/`
- det1.caffemodel
- det1.prototxt
- det2.caffemodel
- det2.prototxt
- det3.caffemodel
- det3.prototxt

## Build examples
```
cd  ${Tengine_ROOT}
make
make install
```
build as ${TENGINE_ROOT}/examples/readme.md

## Run

1. run mtcnn by default
    - model dir is `tengine/models/`
    - test image is `tengine/tests/images/mtcnn_face4.jpg`
	```
	cd ${Tengine_ROOT}/examples/build/mtcnn/
    ./MTCNN
    ```

2. run mtcnn with other models and image
    ```
    cd ${Tengine_ROOT}/examples/build/mtcnn/
	./MTCNN  <test.jpg>  <model_dir>  [save_result.jpg]
    ```
    - `model_dir` is the path of your mtcnn models. Under this path, there are 3 models:
        ```
        ├── det1.caffemodel
        ├── det1.prototxt
        ├── det2.caffemodel
        ├── det2.prototxt
        ├── det3.caffemodel
        ├── det3.prototxt
        ```


## Detect Parameters
There are several parameters for mtcnn face detection. You can set these parameters to improve your detection performance (accuracy or speed).

- min_size: `int` (default: 40)
    
    the minimum size of face to be detected

- conf_p_threshold: `float` (default: 0.6)

    the confidence threshold of Pnet of mtcnn, range in [0,1]
- conf_r_threshold: `float` (default: 0.7)

    the confidence threshold of Rnet of mtcnn, range in [0,1]
- conf_o_threshold: `float` (default: 0.8)

    the confidence threshold of Onet of mtcnn, range in [0,1]
- nms_p_threshold: `float` (default: 0.5)

    the nms threshold of Pnet of mtcnn, range in [0,1]
- nms_r_threshold: `float` (default: 0.7)

    the nms threshold of Rnet of mtcnn, range in [0,1]
- nms_o_threshold: `float` (default: 0.7)

    the nms threshold of Onet of mtcnn, range in [0,1]  


## Benchmark
- test on RK3399 single core A72@1.8GHz
- run one time for warm-up, repeat 30 runs and take the average time.
- detect parameters: use default values in [Detect Parameters](#detect-parameters)


|image|img_size|Tengine|Caffe(Openblas)|
|-----|--------|-------|---------------|
|4.jpg|474x316|117.6ms|204.7ms|
|6.jpg|640x480|184.5ms|317.5ms|


## Reference
Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao, "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks", IEEE Signal Processing Letter

https://github.com/kpzhang93/MTCNN_face_detection_alignment



