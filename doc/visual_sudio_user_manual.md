# Tengine Microsoft Visual Studio User Manual

## Brief

The Visual Studio dev tools & services make app development easy for any platform & language. Tengine support building on windows now.


## Prepare
CMake >= 3.13, Visual Studio >= 2015

Before the very begging, please check CMake and Visual Studio already has been installed. CMake >= 3.13, Visual Studio Version 2017 or 2019 is recommended.
For CUDA or TensorRT backend user, CMake >= 3.18 is needed. CUDA or TensorRT needs to be installed or unpackaged.


## Build

### Download
Download https://github.com/OAID/Tengine.git from GitHub first of all. 

#### CMD shell user
Open "x86 Native Tools Command Prompt for VS 201x" or "x64 Native Tools Command Prompt for VS 201x", "201x" is your installed version. Suppose VS2017 was installed, then:

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

## Demo

TODO