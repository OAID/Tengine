<p align="center"><img width="40%" src="logo-Tengine.png" /></p>

# Tengine Lite

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](./LICENSE) [![Build Status](https://img.shields.io/github/workflow/status/OAID/Tengine/Tengine-Lite-Actions/tengine-lite)](https://github.com/OAID/Tengine/actions?query=workflow%3ATengine-Lite-Actions) [![Test Status](https://img.shields.io/travis/OAID/Tengine/tengine-lite?label=test)](https://travis-ci.org/OAID/Tengine)



## 简介

**Tengine Lite** 由 **OPEN AI LAB** 主导开发，该项目实现了深度学习神经网络模型在嵌入式设备上**快速**、**高效**部署。为实现众多 **AIoT** 应用中跨平台部署，本项目基于原有 Tengine 项目使用 **C 语言**进行重构，针对嵌入式设备资源有限的特点进行深度框架裁剪。同时采用完全分离的前后端设计，利于 CPU、GPU、NPU 等异构计算单元快速移植和部署。同时**兼容 Tengine** 框架原有 API 和 模型格式 tmfile，降低评估、迁移成本。

Tengine Lite 核心代码由 4 个模块组成：

- [**dev**](src/dev)：  NN Operators 后端模块，当前提供 CPU 代码，后续逐步开源 GPU、NPU 参考代码；
- [**lib**](src/lib)：框架核心部件，包括 NNIR、计算图、硬件资源、模型解析器的调度和执行模块；
- [**op**](src/op)：NN Operators 前端模块，实现 NN Operators 注册、初始化；
- [**serializer**](src/serializer)：模型解析器，实现 tmfile 格式的网络模型参数解析。


## 架构简析
![Tengine Lite 架构](doc/architecture.png)


## 如何使用

### 编译

- [快速编译](doc/compile.md) 基于 cmake 实现简单的跨平台编译。

### 示例

- [examples](examples/) 提供基础的分类、检测算法用例，根据 issue 需求持续更新。

### 模型仓库

- [Tengine model zoo](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q) 兼容原有 Tengine 的模型示例仓库（密码：hhgc）。

### 转换工具

- [预编译版本](https://github.com/OAID/Tengine/releases/download/lite-v0.1/convert_model_to_tm)：提供 Linux 系统上预编译好的模型转换工具；
- [在线转换版本](https://convertmodel.com/)：基于 WebAssembly 实现（浏览器本地转换，模型不会上传）；
- [源码编译](doc/convert_tm.md)：参考原有 Tengine 项目编译生成。

### 速度评估

- [Benchmark](benchmark/) 基础网络速度评估工具，欢迎大家更新。

## Roadmap

- [Road map](doc/roadmap.md)

## 致谢
Tengine Lite 参考和借鉴了下列项目：

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

- [FAQ 常见问题](doc/faq.md)

## 技术讨论
- Github issues
- QQ 群: 829565581 (答案：openailab)
- Email: Support@openailab.com
- Tengine 社区: http://www.tengine.org.cn/
