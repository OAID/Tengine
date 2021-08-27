# 模型转换工具

Tengine Convert Tool 支持将多种训练框架模型转换成 Tengine 推理框架适配的模型格式 tmfile。最新版本已支持以下框架模型：
- Caffe
- MXNet
- PyTorch(ONNX)
- TensorFlow
- TFLite
- Darknet
- MegEngine
- OneFlow
- PaddlePalle 2.0

同时 Tengine Convert Tool 还支持将其他优秀的端侧框架模型转换成 Tengine 推理框架适配的模型格式 tmfile。最新版本已支持以下框架模型：

- ncnn

## 依赖库安装

```shell
sudo apt install libprotobuf-dev protobuf-compiler
```

## 源码编译
```shell
mkdir build && cd build
cmake ..
make -j`nproc` && make install
```
编译完成后，生成的可执行文件 `tm_convert_tool` 存放在 `./build/install/bin/` 目录下。

## 执行模型转换

- 命令解析
```shell
$ ./tm_convert_tool -h
[Convert Tools Info]: optional arguments:
        -h    help            show this help message and exit
        -f    input type      path to input float32 tmfile
        -p    input structure path to the network structure of input model(*.prototxt, *.symbol, *.cfg)
        -m    input params    path to the network params of input model(*.caffemodel, *.params, *.weight, *.pb, *.onnx, *.tflite)
        -o    output model    path to output fp32 tmfile
```
- Caffe

```shell
./tm_convert_tool -f caffe -p mobilenet.prototxt -m mobilenet.caffemodel -o mobilenet.tmfile
```

- MXNet

```shell
./tm_convert_tool -f mxnet -p mobilenet.json -m mobilene.params -o mobileent.tmfile
```

- ONNX

```shell
./tm_convert_tool -f onnx -m mobilenet.onnx -o mobilenet.tmfile
```

- TensorFlow

```shell
./tm_convert_tool -f tensorflow -m mobielenet_v1_1.0_224_frozen.pb -o mobilenet.tmfile
```

- TFLite

```shell
./tm_convert_tool -f tflite -m mobielenet.tflite -o mobilenet.tmfile
```

- Darknet

```shell
./tm_convert_tool -f darknet -p yolov3.cfg -m yolov3.weights -o yolov3.tmfile
```

- MegEngine

```shell
./tm_convert_tool -f megengine -m mobilenet.pkl -o mobilenet.tmfile
```

- OneFlow

```shell
./tm_convert_tool -f oneflow -p mobilenet.prototxt -m mobilenet/ -o mobilenet.tmfile
```

- ncnn
```shell
./tm_convert_tool -f ncnn -p mobilenet.param -m mobilenet.bin -o mobilenet.tmfile
```
