# Tengine Lite 编译

## 0 依赖工具安装

编译 Tengine Lite 依赖 `git, g++, cmake, make` 等以下基本工具，如果没有安装，

- Ubuntu18.04 系统命令如下：

  ```bash
  sudo apt-get install cmake make g++ git
  ```

- Fedora28 系统命令如下：

  ```bash
  sudo dnf install cmake make g++ git
  ```


## 1 本地编译

### 1.1 下载 Tengine Lite 源码

```bash
git clone https://github.com/OAID/Tengine.git  tengine-lite
```

### 1.2 编译 Tengine Lite

```bash
cd tengine-lite
mkdir build 
cd build
cmake ..
make
make install
```

编译完成后 build/install/lib 目录会生成 `libtengine-lite.so` 文件，如下所示：

```bash
install
├── bin
│   ├── tm_benchmark
│   ├── tm_classification
│   └── tm_mobilenet_ssd
├── include
│   └── tengine_c_api.h
└── lib
    └── libtengine-lite.so
```

## 2 交叉编译 Arm32/64 Linux 版本

### 2.1 下载源码

```bash
git clone https://github.com/OAID/Tengine.git  tengine-lite
```

### 2.2 安装交叉编译工具链

Arm64 Linux 交叉编译工具链为：

```bash
sudo apt install g++-aarch64-linux-gnu
```

Arm32 Linux 交叉编译工具链为：

```bash
sudo apt install g++-arm-linux-gnueabihf
```

### 2.3 编译 Tengine Lite

Arm64 Linux 交叉编译

```bash
cd Tengine-Lite
mkdir build 
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make
make install
```

Arm32 Linux 交叉编译

```bash
cd Tengine-Lite
mkdir build 
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake ..
make
make install
```

编译完成后会生成 `libtengine-lite.so` 文件，并且会把相关的头文件、`libtengine-lite.so` 文件和相关的测试程序复制到 `build/install` 目录中。

## 3. 交叉编译 Arm32/64 Android 版本

### 3.1 安装 Android NDK

下载 Android NDK http://developer.android.com/ndk/downloads/index.html 并安装，例如：

```
unzip android-ndk-r21d-linux-x86_64.zip
export ANDROID_NDK=<your-ndk-root-path>
```

### 3.2 准备 android toolchain 文件

android.toolchain.cmake 这个文件可以从 $ANDROID_NDK/build/cmake 找到

(可选) 删除debug编译参数，缩小二进制体积 [android-ndk issue](https://github.com/android-ndk/ndk/issues/243)

```
# 用编辑器打开 $ANDROID_NDK/build/cmake/android.toolchain.cmake
# 删除 "-g" 这行
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
```

### 3.3 下载 Tengine Lite 源码

```bash
git clone https://github.com/OAID/Tengine.git tengine-lite
```

### 3.4 编译 Tengine Lite

Arm64 Android 编译命令如下：

```bash
mkdir build-android-aarch64
cd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 ..
make
make install
```

Arm32 Android 编译命令如下：

```bash
mkdir build-android-armv7
cd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-19 ..
make
make install
```

## 4. 异构计算 版本

#### 4.1 ACL

请参考 [Tengine Lite ACL GPU 使用说明](gpu_acl_user_manual.md)。

#### 4.2 Vulkan

请参考 [Tengine Lite Vulkan GPU 使用说明](gpu_vulkan_user_manual.md)。

#### 4.3 CUDA

请参考 [Tengine Lite CUDA GPU 使用说明](gpu_cuda_user_manual.md)。

#### 4.4 TensoRT

请参考 [Tengine Lite TensorRT GPU 使用说明](gpu_trt_user_manual.md)。

#### 4.5 TIM-VX

请参考 [Tengine Lite VeriSilicon NPU 使用说明](npu_tim-vx_user_manual.md)。

## 5. 交叉编译 Arm64 OHOS（鸿蒙系统） 版本

### 5.1 安装 DevEco Studio 和 OHOS NDK

下载安装 DevEco Studio，[传送门](https://developer.harmonyos.com/cn/develop/deveco-studio#download)。若没有华为开发者账号，需到[HarmonysOS应用开发门户](https://developer.harmonyos.com/cn/home)注册。

打开 DevEco Studio，Configure（或File）-> Settings -> Appearance & Behavior -> System Settings -> HarmonyOS SDK，勾选并下载 Native，完成 OHOS NDK 下载。

### 5.2 准备 OHOS NDK cmake toolchain 文件

ohos.toolchain.cmake 这个文件可以从 $OHOS_NDK/build/cmake 找到，例如 E:\soft\Huawei\sdk\native\3.0.0.80\build\cmake\ohos.toolchain.cmake

(可选) 删除debug编译参数，缩小二进制体积，方法和 android ndk相同 [android-ndk issue](https://github.com/android-ndk/ndk/issues/243)

```
# 用编辑器打开 $ANDROID_NDK/build/cmake/android.toolchain.cmake
# 删除 "-g" 这行
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
```

### 5.3 下载 Tengine Lite 源码

```bash
git clone https://github.com/OAID/Tengine.git tengine-lite
```

### 5.4 编译 Tengine Lite

Arm64 OHOS 编译脚本如下（Windows）

`build/ohos-arm64-v8a.bat`:
```bash
@ECHO OFF
@SETLOCAL

:: Set OHOS native toolchain root
@SET OHOS_NDK=<your-ndk-root_path, such as D:/Program/DevEcoStudio/SDK/native/2.0.1.93>


:: Set ninja.exe and cmake.exe
@SET NINJA_EXE=%OHOS_NDK%/build-tools/cmake/bin/ninja.exe
@SET CMAKE_EXE=%OHOS_NDK%/build-tools/cmake/bin/cmake.exe
@SET PATH=%OHOS_NDK%/llvm/bin;%OHOS_NDK%/build-tools/cmake/bin;%PATH%

mkdir build-ohos-armeabi-v7a
pushd build-ohos-armeabi-v7a
%CMAKE_EXE% -G Ninja -DCMAKE_TOOLCHAIN_FILE="%OHOS_NDK%/build/cmake/ohos.toolchain.cmake"  -DCMAKE_MAKE_PROGRAM=%NINJA_EXE%  -DOHOS_ARCH="armeabi-v7a" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON .. 
%CMAKE_EXE% --build . --parallel %NUMBER_OF_PROCESSORS%
%CMAKE_EXE% --build . --target install
popd

mkdir build-ohos-arm64-v8a
pushd build-ohos-arm64-v8a
%CMAKE_EXE% -G Ninja -DCMAKE_TOOLCHAIN_FILE="%OHOS_NDK%/build/cmake/ohos.toolchain.cmake"  -DCMAKE_MAKE_PROGRAM=%NINJA_EXE%  -DOHOS_ARCH="arm64-v8a" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON .. 
%CMAKE_EXE% --build . --parallel %NUMBER_OF_PROCESSORS%
%CMAKE_EXE% --build . --target install
popd


@ENDLOCAL
```

## 6.  使用 Microsoft Visual Studio 编译

请参考 [Visual Studio 使用说明](visual_sudio_user_manual.md)。

