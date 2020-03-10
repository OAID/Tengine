# Tengine Overview

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE) [![Build Status](https://img.shields.io/github/workflow/status/OAID/Tengine/Tengine-Actions)](https://github.com/OAID/Tengine/actions?query=workflow%3ATengine-Actions)

**Tengine**, developed by **OPEN** AI LAB, is a lite, high-performance, and modular inference engine for embedded device.

Tengine is composed of five modules: **core/operator/serializer/executor/driver**.

- [**core**](core)  provides the basic components and functionalities of the system.
- [**operator**](operator)  defines the schema of operators, such as convolution, relu, pooling, etc. al. Here is the current support [**operator list**](doc/operator_ir.md).
- [**serializer**](serializer)  is to load the saved model. The serializer framework is extensible to support different format, including the customized one. Caffe/ONNX/Tensorflow/MXNet and Tengine models can be loaded directly by Tengine.
- [**executor**](executor)  implements the code to run graph and operators. Current version provides a highly optimized implementation for multi A72 cores.
- [**driver**](driver)  is the adapter of real H/W and provides service to device executor by HAL API. It is possible for single driver to create multiple devices.

This version can load and run Caffe model of **mobilenet** and **squeezenet** directly.  For more details, please goto [**install**](doc/install.md).

`NOTE`: Old Caffe model has to be upgraded using **upgrade_net_proto_binary/upgrade_net_proto_binary** from Caffe's package.

## Build and Install
please refer to the [**Linux build**](doc/install.md) and [**Android build**](https://github.com/OAID/Tengine/blob/master/doc/build_android.md)

## Tengine examples and model zoo

please visit [examples](examples/readme.md) for demos on classification/detection and download models from [**Tengine model zoo**](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q) (psw: hhgc)

[**tengine applications**](https://github.com/OAID/Tengine-app) is a project for sharing android/linux applications powered by Tengine  

## Develop New Operator

It is easy to add new operator to Tengine. Here is the guide on [**new operator**](doc/operator_dev.md).

## Support New Model Format

Tengine can be extended to support new serialization format, by building new serializer module. 

[How to build new serializer module](doc/serializer_dev.md)

## Communication && Tech Support
* Github issues
* QQ group: 829565581 (Question:Tengine  Answer:openailab)
* Tengine Community: http://www.tengine.org.cn/

## Release History

## version 1.12.0 - 2020/02/29

**New features**

Support Arm Performance Op : 

	Conv,Deconv,Fc,Pooling,Absval,Batchnorm,Cast,Eltwise,Elu,Hardswish,Interp,Prelu,Relu,Scale,Selu,Sigmode,Softmax,Tanh

Support Arm Winograd


**Others**

Remove Makefile replaced with cmake

## version 1.9.0 - 2019/09/25

**Optimization**

Optimize TF-LITE

Optimize NCHW Conv_DW

Optimize NCHW Deconv

**New features**

Support Online Report

Support NCHW Post trainging tool

**More operator support**

TF-LITE operator LayerNormLstm implementation

TF-LITE operator L2Normalization implementation

TF-LITE operator L2Pool implementation

TF-LITE operator ReluN1To1 implementation

**Others**

Remove Tensorflow/Caffe wrapper

Move caffe/tensorflow/tf-lite/onnx/mxnet/darknet serializers to tengime-module project


## version 1.8.0 - 2019/08/27

**New features**

Support Alexnet Googlenet Inceptiov3 Inceptionv4 Resnet50 on ACL 19.02

Support Darknet models serializer (Yolov2, Yolov2-tiny, Yolov3, Yolov3-tiny)

## version 1.7.1 - 2019/08/02

**New features**

GPU backend support with ACL 19.02

Static tool for convert other models to tengine

Tengine graph use origin layout to execute

Add CV implementation

## version 1.3.2 - 2019/04/19

**tengine model 2.0**

**New apis**

get_graph_node_number()
get_graph_node_by_idx()

**New features**

Separate CPU operator as a independent so:  hclcpu.so

Add Reference Operator

Update Testcase & Update permute for mxnet

Update lstm grun mxnet serializer

Support MXNET serializer in CMakelist.txt

Support TFLITE serializer in CMakelist.txt

Support eltwise in TFLITE serializer

**More operator support**

RNN operator definition and blas implementation

LSTM operator definition and blas implementation

GRU operator definition and blas implementation


## version 1.3.2 - 2019/04/19

**tengine model 2.0**

**New apis**

get_graph_node_number()
get_graph_node_by_idx()

**New features**

Separate CPU operator as a independent so:  hclcpu.so

Add Reference Operator

Update Testcase & Update permute for mxnet

Update lstm grun mxnet serializer

Support MXNET serializer in CMakelist.txt

Support TFLITE serializer in CMakelist.txt

Support eltwise in TFLITE serializer

**More operator support**

RNN operator definition and blas implementation

LSTM operator definition and blas implementation

GRU operator definition and blas implementation

## version 1.0.0 - 2018/12/31

**tengine API 2.0**


New API set for NN inference

Simplify graph create process: just create_graph()  instead of load_model() and create_runtime_graph()

Support perf stat and tensor dump 

Support log redirect

Support to build Android NN Driver with new Tengine API

By setting CONFIG_LEGACY_API=y in makefile.config, tengine API 1.0 still works

**more tensorflow models support**

Support inceptionv3/v4, resnet_v2_101, mobilenet v1/v2 models from [tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models.md)





## version 0.8.0 - 2018/11/15

**Support GPU/CPU Heterogeneous Computing**

By calling set_graph_device(graph,"acl_opencl"), operators that GPU supports will be scheduled to GPU, while left operators will be scheduled on CPU automatically.
    
Here is the guide to run [a MSSD example](https://github.com/OAID/Tengine/blob/master/doc/gpu_cpu_mssd.md) with GPU FP16 
    
**Using c++_shared for Android build**


As NDK toolchains will drop gun_stl finally, this version switches to c++_shared 
    
Please download the pre-built libraries with c++_shared from [Tengine Android Build Libraries](https://pan.baidu.com/s/1-zsqxXXcZEXmCip-nQzcIw) (password: *wtcz*).
    
**Support ACL in Android**

Update the cmake system to support ACL in Android build. please refer to [Android build guide ](https://github.com/OAID/Tengine/blob/master/doc/build_android.md)
    
**Bugfix**
    
The issue to load tengine model converted from MXNet


## version 0.7.2 - 2018/10/15


Serializer:

update ONNX module with new onnx proto version


## version 0.7.0 - 2018/9/15

**New features**
   
Serializer: support saving model as c files

ACL GPU:  add FP16 support

NN: mobilenet v2 support in examples

Accuracy tools:  yolov2 accuracy test

Build:

       support cross-building arm32 library 

       support building on raspberry pi 3b

       automatically clean the build directory when makfile.config changed


**Bug fix**

   A few memory leakage issues in library and examples

   A race condition issue between front thread  and the background working thread

   Tensorflow serializer issue: fail to load inception_v3 model


### version 0.6.0 - 2018/7/02

Support Tengine model file. protobuf is optional now.

Please refer to [tengine_model exmaples](examples/tengine_model) 


### version 0.5.0 - 2018/6/15

**New features**

Support GPU: using ACL (Arm computing library) as a backend graph device

Support blas operator implementation: Tengine can run on x86 without caffe now

Support new NN: Inception-v3/vgg16/faster-rcnn/ssd/yolo-v2

Support Android build:  includes 32bit and 64bit

Support cross-compile on x86 (experimental): 
    debian example contributed by **mcharleb** and **Mani-Sadhasivam** @ Linaro

Support Tensorflow serializer: load inception-v3 and mobilenet TF model directly

Support Tensorflow wrapper: label_image.cpp from tensorflow repo

**Others**

 Single so file now and remove the etc/config according to feedback from field.
     
Tengine will automatically probe the CPU arch/part settings, and there is just one CPU driver now.

To assign cpu manually when necessary:
     
     export TENGINE_CPU_LIST=1,2 
     

Besides probing CPU, a few CPUs are defined in cpu_predefined.cpp, including rk3399/a63/kirin960/apq8096.
To use the predefined CPU, refers to below:

    const struct cpu_info * p_info=get_predefined_cpu("rk3399");
    create_cpu_device("rk3399",p_info);


### version 0.3.0 - 2018/2/6

Introduce the driver/device model to support MT(Multi-Thread)

Support new NN: Inception-v4

Caffe Wrapper examples: squeezenet/mobilenet/mtcnn

MXNet model load examples: squeezenet/mobilenet


### version 0.2.0 - 2018/1/24

Support new operator: Eltwise, PReLU, Slice

Support new NN: mtcnn, resnet and lighten_cnn 

Experimental caffe API wrapper: caffe based application just needs to recompile to use Tengine


### version 0.1.2 - 2017/12/30

Update documents, as well a few fixes.


### version 0.1.0 - 2017/12/29

Initial release of single A72 support

