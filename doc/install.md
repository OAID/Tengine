# Installation Guide

This guide gives instructions on how to build and test Tengine on Linux system. If want to use Tengine for android, refer to **[build_android](build_android.md)**.


## 1. Preparation

### **1.1 Download source code**

To get started, git clone the latest Tengine repository.
	
	git clone https://github.com/OAID/tengine/
	
### **1.2 Install dependent libraries**

* For loading caffe model or tensorflow model
``` 
    sudo apt install libprotobuf-dev protobuf-compiler
```
* For image preprocessing in test samples
```
    sudo apt install libopencv-dev
```
* For caffe/tensorflow wrapper
```
    sudo apt install libboost-all-dev libgoogle-glog-dev
```

### **1.3 Prepare config files**
* Copy config example file `makefile.config.example`
	```
	cd ~/tengine
	
	cp makefile.config.example makefile.config
	
	```
* Edit `makefile.config`
	- **ARCH**

	    By default, `CONFIG_ARCH_ARM64` option is valid.

		* if you want to run Tengine on ARM32 or X86 system, you can run the openblas branch by comment `CONFIG_ARCH_ARM64`, and enable `CONFIG_ARCH_BLAS=y`
		```
		#CONFIG_ARCH_ARM64=y

		CONFIG_ARCH_BLAS=y
		```
	- **ACL GPU**

		By default, ACL GPU option is invalid. Support ACL GPU need uncomment `CONFIG_ACL_GPU=y` and set `ACL_ROOT`
		```
		CONFIG_ACL_GPU=y
		ACL_ROOT = /home/firefly/ComputeLibrary(your ACL root)
		```
	- **Serializer**

		By default, caffe / tengine serializer, option is valid . 
		
		If you want to support mxnet, onnx, tensorflow, uncomment options
		```
		CONFIG_CAFFE_SERIALIZER=y
		# CONFIG_MXNET_SERIALIZER=y
		# CONFIG_ONNX_SERIALIZER=y
		# CONFIG_TF_SERIALIZER=y
		CONFIG_TENGINE_SERIALIZER=y
		```
		
	- **Wrapper**
 
		By default, wrapper option is invalid. if want to use caffe/tensorflow wrapper, uncomment `CONFIG_FRAMEWORK_WRAPPER=y`
        ```
        CONFIG_FRAMEWORK_WRAPPER=y
        ```

## 2. Build
```
cd ~/tengine
make -j4
```

## 3. Run demo

Tengine also provides some example programs for tests, and you can easily validate whether your Tengine is successfully built by running these test programs and inspecting the results.

### 3.1 Run SqueezeNet
    cd ~/tengine
    ./build/tests/bin/bench_sqz

Output message:

	0.2763 - "n02123045 tabby, tabby cat"
	0.2673 - "n02123159 tiger cat"
	0.1766 - "n02119789 kit fox, Vulpes macrotis"
	0.0827 - "n02124075 Egyptian cat"
	0.0777 - "n02085620 Chihuahua"

### 3.2 Run MobileNet
    
    cd ~/tengine
    ./build/tests/bin/bench_mobilenet

Output message:

	8.5976 - "n02123159 tiger cat"
	7.9550 - "n02119022 red fox, Vulpes vulpes"
	7.8679 - "n02119789 kit fox, Vulpes macrotis"
	7.4274 - "n02113023 Pembroke, Pembroke Welsh corgi"
	6.3647 - "n02123045 tabby, tabby cat"

For more information about the performance test of Tengine, please refer to the documentation of [benchmark](benchmark.md).
