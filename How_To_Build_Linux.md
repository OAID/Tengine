# Installation Guide

This guide gives instructions on how to build and test Tengine on Linux system. If want to use Tengine for android, refer to **[build_android](build_android.md)**.


## 1. Preparation

### **1.1 Download source code**

To get started, git clone the latest Tengine repository.
	
	git clone https://github.com/OAID/tengine/
	
## 2. Build
install pkg-config 
```
sudo apt install pkg-config
```

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

	0.2763 - "n02123045 tabby, tabby cat"
	0.2673 - "n02123159 tiger cat"
	0.1766 - "n02119789 kit fox, Vulpes macrotis"
	0.0827 - "n02124075 Egyptian cat"
	0.0777 - "n02085620 Chihuahua"

### 3.2 Run MobileNet
    
    cd ~/tengine
    ./build/benchmark/bin/bench_mobilenet

Output message:

	8.5976 - "n02123159 tiger cat"
	7.9550 - "n02119022 red fox, Vulpes vulpes"
	7.8679 - "n02119789 kit fox, Vulpes macrotis"
	7.4274 - "n02113023 Pembroke, Pembroke Welsh corgi"
	6.3647 - "n02123045 tabby, tabby cat"

For more information about the performance test of Tengine, please refer to the documentation of [benchmark](benchmark.md).
