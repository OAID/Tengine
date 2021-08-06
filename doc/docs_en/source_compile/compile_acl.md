# Source Code Compilation (ACL)

## Brief

The ARM Compute Library(ACL) is a set of computer vision and machine learning functions optimised for both Arm CPUs and GPUs using SIMD technologies.

Tengine Lite has supported to integrate with OpenCL Library of ACL to inference CNN by ARM-Mail GPU.

support check:

```bash
sudo apt install clinfo
clinfo

result:
Number of platforms                               1
.....
```

## Build

### ACL GPU Library

Download ACL

```bash
$ git clone https://github.com/ARM-software/ComputeLibrary.git
$ git checkout v20.05
```

Build ACL 

```bash
$ scons Werror=1 -j4 debug=0 asserts=1 neon=0 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a
```

### Download Tengine Lite

```bash
$ git clone https://github.com/OAID/Tengine.git Tengine-Lite
$ cd Tengine-Lite
```

### Create depend files

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p ./3rdparty/acl/lib
$ mkdir -p ./3rdparty/acl/include
$ cp -rf ComputeLibrary/include/*    Tengine-Lite/3rdparty/acl/include
$ cp -rf ComputeLibrary/arm_compute  Tengine-Lite/3rdparty/acl/include
$ cp -rf ComputeLibrary/support      Tengine-Lite/3rdparty/acl/include
$ cp -rf ComputeLibrary/build/libarm_compute*.so Tengine-Lite/3rdparty/acl/lib/
```

### Build option

```bash
$ mkdir build-acl-arm64 && cd build-acl-arm64
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
	-DTENGINE_ENABLE_ACL=ON ..
$ make -j4
$ make install
```

## Demo

### Depned librarys

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

### Result

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

## List of Supported Hardware

| Chip Manufacturer | Product Model |
| -------- | --------- |
| arm-mali | T-860、G31、G52|

## List of Supported Operators
