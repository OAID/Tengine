<p align="center"><img width="40%" src="logo-Tengine.png" /></p>

# Tengine Lite

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE) [![Build Status](https://img.shields.io/github/workflow/status/OAID/Tengine/Tengine-Lite-Actions/tengine-lite)](https://github.com/OAID/Tengine/actions?query=workflow%3ATengine-Lite-Actions) [![Build Status](https://img.shields.io/github/workflow/status/OAID/Tengine-Convert-Tools/Tengine-Convert-Tools-Actions?label=tools%20build)](https://github.com/OAID/Tengine-Convert-Tools/actions?query=workflow%3ATengine-Convert-Tools-Actions) [![Test Status](https://img.shields.io/travis/OAID/Tengine/tengine-lite?label=test)](https://travis-ci.org/OAID/Tengine)



[**中文版本**](README.md)



## Introduction

**Tengine Lite** is developed by **OPEN AI LAB**. This project implements the **fast** and **efficient** deployment of deep learning neural network models on embedded devices. In order to achieve cross-platform deployment in many **AIoT** applications, this project is based on the original Tengine project using **C language** for reconstruction, and deep frame tailoring for the characteristics of limited embedded device resources. Also, it adopts a completely separated front-end/back-end design, which makes it possible to be transplanted and deployed onto CPU, GPU, NPU and other heterogeneous computing units rapidly, conveniently. At the same time, it is compatible with the original API and model format `tmfile` of **Tengine**, which reduces the cost of evaluation and migration.



The core code of Tengine Lite consists of 4 modules:

- [**dev**](src/dev): NN Operators back-end module, currently provides CPU code, and gradually open source GPU and NPU reference code;
- [**lib**](src/lib): core components of the framework, including NNIR, Computational Graphs, Hardware Resources, and the scheduling and execution modules of model serializer;
- [**op**](src/op): NN Operators front-end module, which realizes registration and initialization of NN Operators;
- [**serializer**](src/serializer): Model decoder, which decodes binary tmfile format into serialized model parameter.


## Architecture
![Tengine Lite 架构](doc/architecture.png)


## How to use

### Compile

- [Quick Compilation](doc/compile.md) Simple cross-platform compilation based on cmake.

### Example

- [examples](examples/) provides basic classification and detection algorithm use cases, which are continuously updated according to the needs of issues.

### Model Zoo

- [Tengine model zoo](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q) Model zoo samples are compatible with the original Tengine (password: hhgc).

### Model Convert tool

- [Pre-compiled version](https://github.com/OAID/Tengine-Convert-Tools/releases/download/v0.1/tm_convert_tool): Pre-compiled model convert tool is provided on Linux system;
- [Online Convert tool](https://convertmodel.com/): Based on WebAssembly (the models are converted locally by browsers, no private data will be uploaded);
- [Source Compilation](https://github.com/OAID/Tengine-Convert-Tools): Refer to **Tengine-Convert-Tools** project, convert tool could be built by users.

### Speed assessment

-[Benchmark](benchmark/) Basic network speed assessment tool, any pull request is welcomed.

## Roadmap

- [Road map](doc/roadmap.md)

## Acknowledgement

Tengine Lite got ideas and developed based on these projects：

- [Caffe](https://github.com/BVLC/caffe)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [MegEngine](https://github.com/MegEngine/MegEngine)
- [ONNX](https://github.com/onnx/onnx)
- [ncnn](https://github.com/Tencent/ncnn)
- [MNN](https://github.com/alibaba/MNN)
- [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)
- [ACL](https://github.com/ARM-software/ComputeLibrary)
- [stb](https://github.com/nothings/stb)
- [convertmodel](https://convertmodel.com/)

## License

- [Apache 2.0](LICENSE)

## FAQ

- [FAQ common questions](doc/faq.md)

## Tech Forum
- Github issues
- QQ groupchat: 829565581 (Answer: openailab)
- Email: Support@openailab.com
- Tengine Community: http://www.tengine.org.cn/
