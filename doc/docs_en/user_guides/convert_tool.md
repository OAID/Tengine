# Model Convert Tool

Tengine Convert Tool supports the conversion of various training framework models into tmfile, a model format adapted by teengine reasoning framework. The latest version has supported the following framework models：
- Caffe
- MXNet
- PyTorch(ONNX)
- TensorFlow
- TFLite
- Darknet
- MegEngine
- OneFlow
- PaddlePalle 2.0

At the same time, teengine converttool also supports converting other excellent end-side framework models into tmfile, a model format adapted by teengine reasoning framework. The latest version has supported the following framework models：

- ncnn

## Dependent Library Installation

```shell
sudo apt install libprotobuf-dev protobuf-compiler
```

## Source Code Compilation
```shell
mkdir build && cd build
cmake ..
make -j`nproc` && make install
```
After the compilation is completed, the feasibility file `tm_convert_tool` is generated. Stored in  `./build/install/bin/` directory。

## Execute Model Convert

- Command parsing
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

