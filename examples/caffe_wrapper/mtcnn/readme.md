# MTCNN implementation with the caffe wrapper of Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is mtcnn implementation with the caffe wrapper of [Tengine](https://github.com/OAID/Tengine).

## Build
1. install [Tengine](https://github.com/OAID/Tengine)
2. install opencv
    ```
    sudo apt-get install libopencv-dev
    ```
3. config
    ```
    cd ~/tengine/examples/caffe_wrapper/mtcnn
    cp etc/config.example etc/config
    ```

4. cmake & make
    ```
    cd ~/tengine/examples/caffe_wrapper/mtcnn
    cmake .
    make
    ```

## Test
- test an image:

    ```
    [usage]: ./CAFFE_MTCNN  <test.jpg>  <model_dir>  [save_result.jpg]
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

## Reference
Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter

https://github.com/kpzhang93/MTCNN_face_detection_alignment
