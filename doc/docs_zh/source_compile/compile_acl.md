# 源码编译（ACL）

## 简介

ARM计算库(ACL)是一套计算机视觉和机器学习功能，使用SIMD技术为ARM cpu和gpu优化。
Tengine支持与ACL的OpenCL库集成，通过ARM-Mail GPU对CNN进行推理。


support check:

```bash
sudo apt install clinfo
clinfo

结果:
Number of platforms                               1
.....
```

## Build

### ACL GPU Library

下载 ACL

```bash
$ git clone https://github.com/ARM-software/ComputeLibrary.git
$ git checkout v20.05
```

构建 ACL 

```bash
$ scons Werror=1 -j4 debug=0 asserts=1 neon=0 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a
```

### 下载 Tengine

```bash
$ git clone https://github.com/OAID/Tengine.git Tengine
$ cd Tengine
```

### 创建依赖文件

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/acl/lib
$ mkdir -p ./3rdparty/acl/include
$ cp -rf ComputeLibrary/include/*    Tengine/3rdparty/acl/include
$ cp -rf ComputeLibrary/arm_compute  Tengine/3rdparty/acl/include
$ cp -rf ComputeLibrary/support      Tengine/3rdparty/acl/include
$ cp -rf ComputeLibrary/build/libarm_compute*.so Tengine/3rdparty/acl/lib/
```

### 构建选项

```bash
$ mkdir build-acl-arm64 && cd build-acl-arm64
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
	-DTENGINE_ENABLE_ACL=ON ..
$ make -j4
$ make install
```

## 示例

### 依赖库

```bash
3rdparty/acl/lib/
├── libarm_compute.so
├── libarm_compute_core.so
└── libarm_compute_graph.so

build-acl-arm64/install/lib/
└── libtengine-lite.so
```

### Set FP16 Inference mode

Enable GPU FP16 mode

```c
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_FP16;
opt.affinity = 0;
```

### 结果

```bash
[root@localhost tengine-lite]# ./tm_mssd_acl -m mssd.tmfile -i ssd_dog.jpg -t 1 -r 10
start to run register cpu allocator
start to run register acl allocator
tengine-lite library version: 1.0-dev
run into gpu by acl
Repeat 10 times, thread 2, avg time 82.32 ms, max_time 135.70 ms, min_time 74.10 ms
--------------------------------------
detect result num: 3 
dog     :99.8%
BOX:( 138 , 209 ),( 324 , 541 )
car     :99.7%
BOX:( 467 , 72 ),( 687 , 171 )
bicycle :99.6%
BOX:( 106 , 141 ),( 574 , 415 )
======================================
[DETECTED IMAGE SAVED]:
======================================
```

## 支持硬件列表

| 芯片厂家  | 型号      |
| -------- | --------- |
| arm-mali | T-860、G31、G52|

## 支持算子列表
