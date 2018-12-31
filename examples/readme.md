# Tengine examples

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE)

- imagenet_classification(squeezenet, mobilenet, mobilenet_v2, resnet50, alexnet, googlenet, inception_v3, inception_v4, vgg16)
- mtcnn
- ssd
- mobilenet_ssd
- yolov2
- faster_rcnn
- lighten_cnn
- caffe_wrapper
  - cpp_classification(squeezenet, mobilenet, etc)
  - mtcnn
- tensorflow_wrapper
  - label_image(inception_v3, mobilenet, resnet50)
- tengine_model
  - classification(squeezenet, mobilenet, mobilenet_v2, resnet50, alexnet, googlenet, inception_v3, inception_v4, vgg16)
  - convert(convert caffe model to tengine model)


## Download required models
Download the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g) (psw: 57vb).


## How to build these examples
### 1. install [Tengine](https://github.com/OAID/Tengine)
### 2. install opencv

```
sudo apt-get install libopencv-dev
```

### 3. make examples
#### 3.1 Linux
```
cd ~/tengine/examples
vim linux_build.sh
```
Set the correct Tengine path.
```
mkdir build
cd build
../linux_build.sh
make -j4 
```
#### 3.2 Android
```
cd ~/tengine/examples
vim android_build_armv7.sh or vim android_build_armv8.sh
```
Set the correct NDK path, Tengine path, Opencv path and protobuf path. If you want to run Tengine with OpenBlas, please add the correct blas path in android_build_armv7.sh or android_build_armv8.sh.
```
-DBLAS_DIR=/home/usr/openbla020_android
```

```
mkdir build
cd build
../android_build_armv7.sh or ../android_build_armv8.sh
make -j4
```

