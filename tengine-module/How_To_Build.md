# Build 

## Build Linux

### **1.1 Install dependent libraries**

* For loading caffe model or tensorflow model
``` 
    sudo apt install libprotobuf-dev protobuf-compiler
```

if use the fedora/Centos ,use follow command instead.
```
    sudo dnf install protobuf-devel
    sudo dnf install boost-devel glog-devel
```

### **1.2 Build**

Set *CONFIG_TENGINE_ROOT*, *ARCH_TYPE*,*PROTOBUF_INCLUDE_PATH* and *PROTOBUF_LIB_PATH*  in file default.config

```
if want to run Tengine with ACL, please set the correct ACL_ROOT path in default.config and turn on CONFIG_ACL_OPENCL in CMakeLists.txt

mkdir build

cd build

../linux_build.sh

make -j4
```

## Build Android

### **1.1 Download Android ndk,Protobuf and ComputeLibrary

Download the below files from [Tengine Android build](https://pan.baidu.com/s/1-zsqxXXcZEXmCip-nQzcIw) (password: *wtcz*):
```
  - android-ndk-r16-linux-x86_64.zip
  - protobuf_lib.tgz
  - ComputeLibrary.tgz
```

### **1.2 Unpack Android ndk, OpenBLAS, Protobuf and ComputeLibrary
```
unzip android-ndk-r16-linux-x86_64.zip
tar -zxvf protobuf_lib.tgz
tar -zxvf ComputeLibrary.tgz
```

### **1.3 Set *CONFIG_TENGINE_ROOT*, *PROTOBUF_INCLUDE_PATH* and *PROTOBUF_LIB_PATH*in file default.confg

```
if want to run Tengine with ACL, please set the correct ACL_ROOT path in file default.confg and turn on CONFIG_ACL_OPENCL in file CMakeLists.txt

```

### **1.4 Build

```
mkdir build

cd build

../android_build.sh

make -j4
```

## Build Linux by other toolchains

### **1.1 Setting**

Set *CONFIG_TENGINE_ROOT*, *ARCH_TYPE*,*PROTOBUF_INCLUDE_PATH* and *PROTOBUF_LIB_PATH*  in file default.confg

Uncomment '#EMBEDDED_CROSS_ROOT=/opt/hisi-linux/x86-arm/arm-himix200-linux/bin/' and set corrent cross compiler path

Uncomment 'CROSS_COMPILE=arm-himix200-linux-' and set corrent cross compiler prefix

### **1.2 Build**

mkdir build

cd build

../linux_build.sh

make -j4

