# Tengine Overview

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE) [![Build Status](https://img.shields.io/github/workflow/status/OAID/Tengine/Tengine-Actions)](https://github.com/OAID/Tengine/actions?query=workflow%3ATengine-Actions)

**Tengine**, developed by **OPEN** AI LAB, is an AI application development platform for AIoT scenarios launched by OPEN AI LAB, which is dedicated to solving the fragmentation problem of aiot industrial chain and accelerating the landing of AI industrialization. Tengine is specially designed for AIoT scenarios, and it has several features, such as cross platform, heterogeneous scheduling, chip bottom acceleration, ultra light weight and independent, and complete development and deployment tool chain. Tengine is compatible with a variety of operating systems and deep learning algorithm framework, which simplifies and accelerates the rapid migration of scene oriented AI algorithm on embedded edge devices, as well as the actual application deployment;

Tengine is composed of five modules: **core/operator/serializer/executor/driver**.

- [**core**](core)  provides the basic components and functionalities of the system.
- [**operator**](operator)  defines the schema of operators, such as convolution, relu, pooling, etc. al. Here is the current support [**operator list**](https://github.com/OAID/Tengine/wiki/Tengine-Support-Operators-List).
- [**serializer**](serializer)  is to load the saved model. The serializer framework is extensible to support different format, including the customized one. Caffe/ONNX/Tensorflow/MXNet and Tengine models can be loaded directly by Tengine.
- [**executor**](executor)  implements the code to run graph and operators. Current version provides a highly optimized implementation for multi A72 cores.
- [**driver**](driver)  is the adapter of real H/W and provides service to device executor by HAL API. It is possible for single driver to create multiple devices.


## Build and Install
please refer to Wiki

## Tengine examples and model zoo

please visit [examples](https://github.com/OAID/Tengine/tree/master/examples) for demos on classification/detection and download models from [**Tengine model zoo**](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q) (psw: hhgc)

[**tengine applications**](https://github.com/OAID/Tengine-app) is a project for sharing android/linux applications powered by Tengine  



## Communication && Tech Support
* Github issues
* QQ group: 829565581 (Question:Tengine  Answer:openailab)
* email: Support@openailab.com
* Tengine Community: http://www.tengine.org.cn/



## Benchmark

Test on RK3399-1*A72 

 Model  |  fp32 | int8-hybrid | int8-e2e |
 ---- | ----- | ------  | ------
 Squeezenet v1.1  | 55.3ms | 48.6ms| 44.6ms 
 Mobilenet v1  | 108.7ms | 74.6ms| 64.2ms

More Benchmark data to be added.


## Roadmap

2020.4 updated

##### Feature

- [ ] More examples
- [ ] Netron support Tengine model .tmfile
- [ ] New compile configuration file
- [ ] Easy to use C++ API
- [ ] Easy to use Python API
- [ ] Support more ops of ONNX(PyTorch)

##### Optimization

- [ ] x86 platform ops

