# **Build Tengine for android**  
The document describes how to build Tengine for android.

## **Catalog**

#### [**Use prebuilt Tengine android package**](build_android.md#use-prebuilt-tengine-android-package-1)
#### [**Building Tengine for android**](build_android.md#building-tengine-for-android-1)
#### [**Run Tengine on android device**](build_android.md#run-tengine-on-android-device-1)

##  Building Tengine for android

### 1. Download Tengine project
```
git clone --recurse-submodules https://github.com/OAID/tengine/
```
### 2. Download Android ndk, OpenBLAS, OpenCV, Protobuf and ComputeLibrary

Download the below files from [Tengine Android build](https://pan.baidu.com/s/1-zsqxXXcZEXmCip-nQzcIw) (password: *wtcz*):
```
  - android-ndk-r16-linux-x86_64.zip
  - Openblas_0220_android.tgz
```
### 3. Unpack Android ndk and OpenBLAS
```
unzip android-ndk-r16-linux-x86_64.zip
tar -zxvf Openblas_0220_android.tgz
```
### 4. Set system environment
Edit example_config/arm_android_cross.config
```
ANDROID_NDK=/home/test/android-ndk-r16
PROTOBUF_LIB_PATH=/home/test/protobuf/arm64_lib
PROTOBUF_INCLUDE_PATH=/home/test/protobuf/include
ACL_ROOT=/home/test/acl
```

### 5. Build tengine
```
cd ~/tengine
bash android_build.sh example_config/arm_android_cross.config
```

### 6. Build example
### 6.1. Set the *TENGINE_DIR*
If want to run Tengine with OpenBlas, please add the correct blas path in example/android_build_armv7.sh or example/android_build_armv8.sh. `-DBLAS_DIR=/home/usr/Openblas_0220_android`
```
cd ~/tengine/example
vim android_build_armv7.sh or  vim android_build_armv8.sh
```
Make sure the install directory is in your *TENGINE_DIR*.

### 6.2. Build the example    
```
cd ~/tengine/example
mdkir build
cd build
../android_build_armv7.sh or ../android_build_armv8.sh
make -j4
```
---
### 7. Run Tengine on android device
### 7.1. Prepare files
```
cd ~/tengine
./android_pack.sh example_config/arm_android_cross.config
cp -rf ./models ./android_pack
cp -rf ./install/benchmark ./android_pack
```
Other files:
   - images(cat.jpg, bike.jpg or others)
   - examples/build/imagenet_classification/Classify 

<br />
Copy these files into `~/tengine/android_pack`

### 7.2. Install adb

```
sudo apt-get install android-tools-adb
adb push ./android_pack /data/local/tmp/
```

### 7.3. Run
```
adb root
adb shell
cd /data/local/tmp/android_pack
export LD_LIBRARY_PATH=.
./benchmark/bench_sqz
```
Ouput message:
```
0.2831 - "n02123045 tabby, tabby cat"
0.2714 - "n02123159 tiger cat"
0.1687 - "n02119789 kit fox, Vulpes macrotis"
0.0843 - "n02124075 Egyptian cat"
0.0750 - "n02085620 Chihuahua"
```

