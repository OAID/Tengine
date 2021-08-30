# 源码编译（Microsoft Visual Studio）

## 简介

Visual Studio开发工具和服务使任何平台和语言的应用程序开发变得容易。Tengine支持在Windows上编译。


## 准备
CMake >= 3.13, Visual Studio >= 2015

Before the very begging, please check CMake and Visual Studio already has been installed. CMake >= 3.13, Visual Studio Version 2017 or 2019 is recommended.
For CUDA or TensorRT backend user, CMake >= 3.18 is needed. CUDA or TensorRT needs to be installed or unpackaged.


## 构建

### 下载
首先从 GitHub下载 https://github.com/OAID/Tengine.git 

### CMD shell user
打开 "x86 Native Tools Command Prompt for VS 201x" or "x64 Native Tools Command Prompt for VS 201x", "201x" 是你的安装版本. 假设安装的是VS2017 ，操作如下:

```bash
set PATH=X:/your/cmake/bin;%PATH%

cd /d X:/your/downloaded/Tengine
md build
cd build
cmake.exe -G "Visual Studio 15 2017 Win64" -DTENGINE_OPENMP=OFF -DTENGINE_BUILD_EXAMPLES=OFF ..
::cmake.exe -G "Visual Studio 16 2019" -A x64 -DTENGINE_OPENMP=OFF ..
cmake.exe --build . --parallel %NUMBER_OF_PROCESSORS%
cmake.exe --build . --target install
```

## 示例

TODO