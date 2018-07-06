# Installation Guide

This guide gives instructions on how to build and test Tengine on your system.

## 1. Preparation

### **1.1 Download source code**

To get started, git clone the latest Tengine repository.
	
	git clone https://github.com/OAID/tengine/
	
### **1.2 Install Depency Libraries**

* libprotobuf: for load caffemodel
	``` 
	sudo apt install libprotobuf-dev protobuf-compiler libboost-all-dev libgoogle-glog-dev
	```
* libopencv: for image preprocessing in test samples
	```
	sudo apt install libopencv-dev
	```
* [Caffe](https://github.com/BVLC/caffe) (Optional): use Caffe's operators for verifing your operator implementations in Tengine

	Please see http://caffe.berkeleyvision.org/installation.html



### **1.3 Prepare config files**
* copy config example file
	```
	cd ~/tengine
	
	cp makefile.config.example makefile.config
	
	```
* edit `makefile.config`
	- **arm64** 
		
		By default, `CONFIG_ARCH_ARM64` option is valid.

	- if your want to use **Caffe**, set
		```
		CONFIG_CAFFE_REF=y
		CAFFE_ROOT = /home/firefly/caffe (your caffe path)
		```
	- if you want to run using **Openblas**, you install `sudo apt-get install libopenblas-dev` and set
		```
		CONFIG_ARCH_BLAS=y
		```

	- if you want to **run GPU using ACL**, see [acl_driver.md](acl_driver.md) for how to build **ACL** and set
		```
		CONFIG_ACL_GPU=y
		ACL_ROOT = /home/firefly/ComputeLibrary(your ACL root)
		```
	- **Serializer support**: 
		by default, caffe model serializer option is valid `CONFIG_CAFFE_SUPPORT=y`. 
		
		If you want to support mxnet, tensorflow, onnx serializer, you uncomment the options
		```
		#CONFIG_MXNET_SERIALIZER=y
		#CONFIG_ONNX_SERAILIZER=y
		# CONFIG_TF_SERIALIZER=y
		```
## 2. Build
```
cd ~/tengine
make
```

## 3. Run Demo

Tengine also provides some example programs for tests, and you can easily validate whether your Tengine is successfully built by running these test programs and inspecting the results.

### 3.1 Run SqueezeNet
   
    	./build/tests/bin/bench_sqz -r1

	    `-r1` means repeat one time.
Output message:

	    0.2763 - "n02123045 tabby, tabby cat"
	    0.2673 - "n02123159 tiger cat"
	    0.1766 - "n02119789 kit fox, Vulpes macrotis"
    	0.0827 - "n02124075 Egyptian cat"
	    0.0777 - "n02085620 Chihuahua"

### 3.2 Run MobileNet
    
   
	    ./build/tests/bin/bench_mobilenet -r1

Output message:

    	8.5976 - "n02123159 tiger cat"
	    7.9550 - "n02119022 red fox, Vulpes vulpes"
    	7.8679 - "n02119789 kit fox, Vulpes macrotis"
	    7.4274 - "n02113023 Pembroke, Pembroke Welsh corgi"
	    6.3647 - "n02123045 tabby, tabby cat"

For more information about the performance test of Tengine, please refer to the documentation of **[benchmark](benchmark.md)**.

Please visit **[exmaples](../examples/readme.md)** for applications on classification/detection etc.
