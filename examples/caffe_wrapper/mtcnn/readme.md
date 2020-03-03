# MTCNN implementation with the caffe wrapper of Tengine

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

This is mtcnn implementation with the caffe wrapper of [Tengine](https://github.com/OAID/Tengine).

## Download required models
Download the required models from [Tengine model zoo](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q) (psw: hhgc)

- det1.caffemodel
- det1.prototxt
- det2.caffemodel
- det2.prototxt
- det3.caffemodel
- det3.prototxt

And store these files into `${Tengine_ROOT}/models/`

## Build examples
```
cd  ${Tengine_ROOT}
make install
```
build as ${TENGINE_ROOT}/examples/readme.md
NOTE: "add_subdirectory(caffe_wrapper)" must be opened in ${TENGINE_ROOT}/examples/CMakeLists.txt

## Run
- go to the directory of the executive program

    ```
    cd ${Tengine_ROOT}/examples/build/caffe_wrapper/mtcnn
    ```
- test an image:

    ```
    [usage]: ./CAFFE_MTCNN  <test.jpg>  <model_dir>  [save_result.jpg]
    ```
`model_dir` is the path of your mtcnn models.

## Reference
Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter

https://github.com/kpzhang93/MTCNN_face_detection_alignment

