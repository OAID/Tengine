# Tengine Lite with VeriSilicon TIM-VX User Manual

## 1. Brief

[TIM-VX](https://github.com/VeriSilicon/TIM-VX.git) is a software integration module provided by VeriSilicon to facilitate deployment of Neural-Networks on OpenVX enabled ML accelerators.

Tengine Lite has supported to integrate with TIM-VX Library of Verisilicon to inference CNN by [Khadas VIM3](https://www.khadas.cn/product-page/vim3)(Amlogic A311D).

## 2. How to Build

For some special reasons, only supported on Khadas VIM3 or x86_64 simulator to work the following steps, currently.

##### Download Source code of TIM-VX 

```bash
$ git clone https://github.com/VeriSilicon/TIM-VX.git
$ sudo apt-get install libgtest-dev
```

##### Download Tengine Lite

```bash
$ git clone https://github.com/OAID/Tengine.git tengine-lite
$ cd tengine-lite
```

#### 2.1 Prepare for x86_64 simulator platform

##### Create depend files

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/lib/x86_64
$ mkdir -p ./3rdparty/tim-vx/include
$ cp -rf ../TIM-VX/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/src    ./src/dev/tim-vx/
$ cp -rf ../TIM-VX/prebuilt-sdk/x86_64_linux/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/prebuilt-sdk/x86_64_linux/lib/*    ./3rdparty/tim-vx/lib/x86_64/
```

#### 2.2 Prepare for on Khadas VIM3 platform

##### Download prebuild-sdk of A311D

```bash
$ wget -c https://github.com/VeriSilicon/TIM-VX/releases/download/v1.1.28/aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz
$ tar zxvf aarch64_A311D_D312513_A294074_R311680_T312233_O312045.tgz
$ mv aarch64_A311D_D312513_A294074_R311680_T312233_O312045 prebuild-sdk-a311d
```

##### Create depend files

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/tim-vx/lib/aarch64
$ mkdir -p ./3rdparty/tim-vx/include
$ cp -rf ../TIM-VX/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../TIM-VX/src    ./src/dev/tim-vx/
$ cp -rf ../prebuild-sdk-a311d/include/*    ./3rdparty/tim-vx/include/
$ cp -rf ../prebuild-sdk-a311d/lib/*    ./3rdparty/tim-vx/lib/aarch64/
```

#### 2.3 Build Tengine Lite with TIM-VX

```bash
$ mkdir build && cd build
$ cmake -DTENGINE_ENABLE_TIM_VX=ON -DTENGINE_ENABLE_TIM_VX_INTEGRATION=ON ..
$ make -j4
$ make install
```

## 3. Demo

#### 3.1 Depned librarys

```
3rdparty/tim-vx/lib/
├── libArchModelSw.so
├── libCLC.so
├── libGAL.so
├── libNNArchPerf.so
├── libOpenVX.so
├── libOpenVXU.so
└── libVSC.so

build-tim-vx-arm64/install/lib/
└── libtengine-lite.so
```

On the Khadas VIM3, it need to replace those libraries in the /lib/ 

#### 3.2 Replace the kernel module on the board if necessary
- Q: Why?
- A: Because the firmware of Khadas VIM3 maybe pre-install old version kernel module of NPU  
- Q: How to?
- A: Remove the old kernel module and replace it with the new version(in the /prebuild-sdk-a311d/lib/galcore.ko) 
```
$ rmmod galcore
$ insmod galcore.ko
```

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
[khadas@Khadas tengine-lite]# ./tm_classification_timvx -m squeezenet_uint8.tmfile -i cat.jpg -r 1 -s 0.017,0.017,0.017 -r 10
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
