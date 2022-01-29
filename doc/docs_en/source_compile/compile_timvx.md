# Source Code Compilation (TIM-VX)

## 1. Brief

[TIM-VX](https://github.com/VeriSilicon/TIM-VX.git) is a software integration module provided by VeriSilicon to facilitate deployment of Neural-Networks on OpenVX enabled ML accelerators.

Tengine Lite has supported to integrate with TIM-VX Library of Verisilicon to inference CNN by [Khadas VIM3](https://www.khadas.cn/product-page/vim3) (Amlogic A311D).

## 2. How to Build

For some special reasons, only supported on Khadas VIM3 or x86_64 simulator to work the following steps, currently.

### Download Source code of TIM-VX

TIM-VX is updated very frequently, and the latest compatible version is `68b5acb`.

```bash
$ git clone https://github.com/VeriSilicon/TIM-VX.git
$ cd TIM-VX && git checkout 68b5acb && cd ..
```

### Download Tengine Lite

```bash
$ git clone https://github.com/OAID/Tengine.git tengine-lite
```

### 2.1 Prepare for x86_64 simulator platform

**non-cross-compilation**

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/lib/x86_64
$ mkdir -p ./3rdparty/tim-vx/include
$ cp -rf ../TIM-VX/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/src    ./source/device/tim-vx/
$ cp -rf ../TIM-VX/prebuilt-sdk/x86_64_linux/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/prebuilt-sdk/x86_64_linux/lib/*    ./3rdparty/tim-vx/lib/x86_64/
$ rm ./source/device/tim-vx/src/tim/vx/*_test.cc
```

Build Tengine

```bash
$ export LD_LIBRARY_PATH=<tengine-lite-root-dir>/3rdparty/tim-vx/lib/x86_64

$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j4
```

### 2.2 Prepare for Khadas VIM3 platform

Prepare for VIM3 prebuild sdk:

```bash
$ wget -c https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz
$ tar zxvf aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz
$ mv aarch64_A311D_D312513_A294074_R311680_T312233_O312045 prebuild-sdk-a311d

$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/lib/aarch64
$ mkdir -p ./3rdparty/tim-vx/include
$ cp -rf ../TIM-VX/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/src    ./source/device/tim-vx/
$ cp -rf ../prebuild-sdk-a311d/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../prebuild-sdk-a311d/lib/*    ./3rdparty/tim-vx/lib/aarch64/
$ rm ./source/device/tim-vx/src/tim/vx/*_test.cc
```

#### 2.2.1 cross-compilation

TOOLCHAIN_FILE in the <tengine-lite-root-dir>/toolchains
```bash
$ export LD_LIBRARY_PATH=<tengine-lite-root-dir>/3rdparty/tim-vx/lib/aarch64

$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j4
```

#### 2.2.2 non-cross-compilation

Check for galcore:

```bash
$ sudo dmesg | grep Galcore
```

if  ( Galcore version < 6.4.3.p0.286725 )

```bash
$ rmmod galcore
$ insmod galcore.ko
```

Check for libOpenVX.so*:

```bash
$ sudo find / -name "libOpenVX.so*"
```

if  ( libOpenVX.so version <   libOpenVX.so.1.3.0  in  /usr/lib )

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p Backup
$ mv /usr/lib/libOpenVX.so* ./Backup
$ cp -rf ../prebuild-sdk-a311d/lib/libOpenVX.so* /usr/lib
```

Build Tengine

```bash
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j4
```


### 2.3 Prepare for NXP i.MX 8M Plus Linux latform

It is highly recommended to compile Tengine natively rather than cross-compiling.

#### 2.4.1 Prepare 3rd-party library

```bash
wget -c https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_S905D3_D312513_A294074_R311680_T312233_O312045.tgz && tar xvf aarch64_S905D3_D312513_A294074_R311680_T312233_O312045.tgz && mv aarch64_S905D3_D312513_A294074_R311680_T312233_O312045 prebuild-sdk-s905d3
```

#### 2.4.2 Prepare source files

Suppose you have the following dependencies in a same directory:

```
.
├── tengine-lite
└── TIM-VX
└── prebuild-sdk-s905d3
```

Then you can prepare the source files in the following way:

```bash
cd tengine-lite && mkdir -p ./source/device/tim-vx/ && /bin/cp -rf ../TIM-VX/include ./source/device/tim-vx/ && /bin/cp -rf ../TIM-VX/src ./source/device/tim-vx/ && mkdir -p ./3rdparty/tim-vx/include && mkdir -p ./3rdparty/tim-vx/lib/aarch64 && /bin/cp -rf ../prebuild-sdk-s905d3/include/* ./3rdparty/tim-vx/include/
```

#### 2.4.3 Build Tengine


```bash
mkdir -p build && cd ./build && cmake -DTENGINE_ENABLE_TIM_VX=ON .. && make -j4 VERBOSE=1 && make install
```

You will find `libengine-lite.so` from the install directory. And example applications are possible to run.

