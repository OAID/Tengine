# Installation Guide

This guide gives instructions on how to build and test Tengine on your system.

## 1. Preparation

### 1.1 Download source code

To get started, git clone the latest Tengine repository.
	
	git clone https://github.com/OAID/tengine/
	
### 1.2 Install Depency Libraries

* libprotobuf: for load caffemodel
	``` 
	sudo apt install libprotobuf-dev
	```
* libopencv: for image preprocessing in test samples
	```
	sudo apt install libopencv-dev
	```
* [Caffe](https://github.com/BVLC/caffe) (Optional): use Caffe's operators for verifing your operator implementations in Tengine

	Please see http://caffe.berkeleyvision.org/installation.html



### 1.3 Prepare config files
* copy config example file
	```
	cd ~/tengine
	
	cp makefile.config.example makefile.config
	
	cp etc/config.example etc/config
	```
* edit `makefile.config`
	- By default, `CONFIG_ARCH_ARM64` option is valid,
	- if your want to use Caffe, set
		```
		CONFIG_CAFFE_REF=y
		CAFFE_ROOT = /home/firefly/caffe (your caffe path)
		```
* edit `etc/config`
	- the default driver is `RK3399`
		```
		driver.probe.0=RK3399
		```
	- choose your cpu to run Tengine:
		*	single A72: `device.default= cpu.rk3399.a72.0`
		*   two A72's: `device.default= cpu.rk3399.a72.all`
		*   single A53: `device.default= cpu.rk3399.a53.2`
		*   four A53's: `device.default= cpu.rk3399.a53.all`
		*   2 A72's + 4 A53's: `device.default= cpu.rk3399.cpu.all`

		default setting uses two A72's.

## 2. Build
```
cd ~/tengine
make
make test (Optional)
```
`make test` is executed when you need to build and run some additional test programs in the project.

## 3. Run Demo

Tengine also provides some example programs for tests, and you can easily validate whether your Tengine is successfully built by running these test programs and inspecting the results.

### 3.1 Run SqueezeNet

	cd ~/tengine
	
	./build/tests/bin/test_sqz

Output message:

	0.2763 - "n02123045 tabby, tabby cat"
	0.2673 - "n02123159 tiger cat"
	0.1766 - "n02119789 kit fox, Vulpes macrotis"
	0.0827 - "n02124075 Egyptian cat"
	0.0777 - "n02085620 Chihuahua"

### 3.2 Run MobileNet

	cd ~/tengine
	
	./build/tests/bin/test_mobilenet

Output message:

	8.5976 - "n02123159 tiger cat"
	7.9550 - "n02119022 red fox, Vulpes vulpes"
	7.8679 - "n02119789 kit fox, Vulpes macrotis"
	7.4274 - "n02113023 Pembroke, Pembroke Welsh corgi"
	6.3647 - "n02123045 tabby, tabby cat"

For more information about the performance test of Tengine, please refer to the documentation of [benchmark](benchmark.md).
