# Tengine Lite with VeriSilicon TIM-VX User Manual

## 1. Brief

[TIM-VX](https://github.com/VeriSilicon/TIM-VX.git) is a software integration module provided by VeriSilicon to facilitate deployment of Neural-Networks on OpenVX enabled ML accelerators.

Tengine Lite has supported to integrate with TIM-VX Library of Verisilicon to inference CNN by [Khadas VIM3](https://www.khadas.cn/product-page/vim3)(Amlogic A311D).

## 2. How to Build

For some special reasons, only supported on Khadas VIM3 or x86_64 simulator to work the following steps, currently.

##### Download Source code of TIM-VX 

```bash
$ git clone https://github.com/VeriSilicon/TIM-VX.git
```

##### Download Tengine Lite

```bash
$ git clone https://github.com/OAID/Tengine.git tengine-lite
$ cd tengine-lite
```

#### 2.1 Prepare for x86_64 simulator platform

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

#### 2.2 Prepare for Khadas VIM3 platform

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

**2.2.1 cross-compilation**

TOOLCHAIN_FILE in the <tengine-lite-root-dir>/toolchains
```bash
$ export LD_LIBRARY_PATH=<tengine-lite-root-dir>/3rdparty/tim-vx/lib/aarch64

$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j4
```

**2.2.2 non-cross-compilation**

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

#### 2.3 Prepare for NXP platform

**non-cross-compilation**

```bash
$ cd <tengine-lite-root-dir>
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON ..
$ make -j4
```

## 3. Demo

#### 3.1 Depned librarys

```
build-tim-vx-arm64/install/lib/
└── libtengine-lite.so
```

On the Khadas VIM3, it need to replace those libraries in the /lib/ 

#### 3.2 Set uint8 Inference mode

TIM-VX Library needs the uint8 network model

```bash
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_UINT8;
opt.affinity = 0;
```

#### 3.3 Result

```
[khadas@Khadas tengine-lite]# ./example/tm_classification_timvx -m squeezenet_uint8.tmfile -i cat.jpg -r 1 -s 0.017,0.017,0.017 -r 10
Tengine plugin allocator TIMVX is registered.
Image height not specified, use default 227
Image width not specified, use default  227
Mean value not specified, use default   104.0, 116.7, 122.7
tengine-lite library version: 1.2-dev
TIM-VX prerun.

model file : squeezenet_uint8.tmfile
image file : cat.jpg
img_h, img_w, scale[3], mean[3] : 227 227 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 10 times, thread 1, avg time 2.95 ms, max_time 3.42 ms, min_time 2.76 ms
--------------------------------------
34.786182, 278
33.942883, 287
33.732056, 280
32.045452, 277
30.780502, 282
```

### 4. Support list
| Vendor  | Devices      |
| ------- | ------------ |
| Amlogic | A311D, S905D3|
| NXP     | i.MX 8M Plus |
| JLQ     | JA310        |
| X86-64  | Simulator    |

### 5. The uint8 quantization model
The TIM-VX NPU backend needs the uint8 tmfile as it's input model file, you can **quantize** the tmfile from **float32** to **uint8** from here. 
- [Tengine Post Training Quantization Tools](../tools/quantize/README.md)
- [Download the uint8 quant tool](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_uint8)
