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

下载 Tengine Lite 源码，位于 Tengine 的分支 tengine-lite 上：

```bash
git clone -b tengine-lite https://github.com/OAID/Tengine.git  Tengine-Lite
```

### 1.2 编译 Tengine Lite

```bash
cd Tengine-Lite
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
git clone -b tengine-lite https://github.com/OAID/Tengine.git  Tengine-Lite
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

传送门 http://developer.android.com/ndk/downloads/index.html

比如我把 android-ndk 解压到 /home/openailab/android-ndk-r18b

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
git clone -b tengine-lite https://github.com/OAID/Tengine.git Tengine-Lite
```

### 3.4 编译 Tengine Lite

Arm64 Android 编译命令如下：

```bash
mkdir build-android-aarch64
cd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build-android-aarch64/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 ..
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

## 4. 总结

本文档只是简单指导如何编译对应的 Tengine Lite 版本，有需要可以参考 ` Tengine-Lite/build.sh` 文件。
