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



## Benchmark

Test on RK3399-1*A72
 Model  |  fp32 | int8 Mixed precision | e2e int8 |
 ---- | ----- | ------  | ------
 Squeezenet v1.1  | 55.3ms | 48.6ms | 44.6ms 
 Mobilenet v1  | 108.7ms |   74.6ms | 64.2ms
