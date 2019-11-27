# Installation Guide

This guide gives instructions on how to build and test Tengine on Linux system. If want to use Tengine for android, refer to **[build_android](build_android.md)**.


## 1. Preparation

### **1.1 Download source code**

To get started, git clone the latest Tengine repository.
	
	git clone --recurse-submodules https://github.com/OAID/tengine/
	
### **1.2 Install dependent libraries**

* For loading caffe model or tensorflow model
``` 
    sudo apt install libprotobuf-dev protobuf-compiler
```
* For image preprocessing in test samples
```
    sudo apt install libopencv-dev
```
* Tengine-module dependent library 
``` 
    opencv, protobuf
```

### **1.3 Edit script**
install pkg-config
```
sudo apt install pkg-config
```
if pkg-config can't find the pc of lib or the libs, you can reference example to edit your linux_build.sh

```
OPENBLAS_LIB=`-L<lib-dir> -l<libname>`
OPENBLAS_CFLAGS=`-I<include-dir>`
```

## 2. Build

```
cd ~/tengine
bash linux_build.sh default_config/x86_linux_native.config
```

## 3. Run demo

Tengine also provides some example programs for tests, and you can easily validate whether your Tengine is successfully built by running these test programs and inspecting the results.

### 3.1 Run SqueezeNet
    cd ~/tengine
    ./build/benchmark/bin/bench_sqz

Output message:

	0.2831 - "n02123045 tabby, tabby cat"
	0.2714 - "n02123159 tiger cat"
	0.1687 - "n02119789 kit fox, Vulpes macrotis"
	0.0843 - "n02124075 Egyptian cat"
	0.0750 - "n02085620 Chihuahua"

### 3.2 Run MobileNet
    
    cd ~/tengine
    ./build/benchmark/bin/bench_mobilenet

Output message:

	8.6167 - "n02123159 tiger cat"
	7.9423 - "n02119022 red fox, Vulpes vulpes"
	7.8727 - "n02119789 kit fox, Vulpes macrotis"
	7.3940 - "n02113023 Pembroke, Pembroke Welsh corgi"
	6.3881 - "n02123045 tabby, tabby cat"

For more information about the performance test of Tengine, please refer to the documentation of [benchmark](benchmark.md).
