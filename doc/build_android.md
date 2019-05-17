# **Build Tengine for android**  
The document describes how to build Tengine for android.

## **Catalog**

#### [**Use prebuilt Tengine android package**](build_android.md#use-prebuilt-tengine-android-package-1)
#### [**Building Tengine for android**](build_android.md#building-tengine-for-android-1)
#### [**Run Tengine on android device**](build_android.md#run-tengine-on-android-device-1)

---

##  Use prebuilt Tengine android package
Download the below file from [Tengine Android build](https://pan.baidu.com/s/1-zsqxXXcZEXmCip-nQzcIw) (password: *wtcz*), and unpack it.
```
  - Tengine_android_build_package_arm64.tgz
```
This package supports ACL GPU, Caffe serializer and Tensorflow serializer.

adb push the package to the android device.

And then follow the steps: [Run](build_android.md#run-tengine-on-android-device-1).

---

##  Building Tengine for android

### 1. Download Tengine project
```
git clone https://github.com/OAID/tengine/
```
### 2. Download Android ndk, OpenBLAS, OpenCV, Protobuf and ComputeLibrary

Download the below files from [Tengine Android build](https://pan.baidu.com/s/1-zsqxXXcZEXmCip-nQzcIw) (password: *wtcz*):
```
  - android-ndk-r16-linux-x86_64.zip
  - openblas020_android.tgz
  - opencv.tgz
  - protobuf_lib.tgz
  - ComputeLibrary.tgz
```
### 3. Unpack Android ndk, OpenBLAS, Protobuf and ComputeLibrary
```
unzip android-ndk-r16-linux-x86_64.zip
tar -zxvf openblas020_android.tgz
tar -zxvf protobuf_lib.tgz
tar -zxvf ComputeLibrary.tgz
```
### 4. Set *ANDROID_NDK*, *PROTOBUF_DIR*, *BLAS_DIR*, *ACL_ROOT* and *CONFIG_ARCH_TYPE* in file *android_config.txt*
If build the library for armv7, set the CONFIG_ARCH_TYPE: **ARMv7**.

Otherwise, set the CONFIG_ARCH_TYPE: **ARMv8**

```
vim  ~/tengine/android_config.txt
```

If run Tengine with Openblas, remove the DCONFIG_ARCH_ARM64 in android_build_armv8.sh or DCONFIG_ARCH_ARM32 in android_build_armv7.sh, and set the **-DCONFIG_ARCH_BLAS=ON**, and you must set the correct **BLAS_DIR** in android_config.txt.

If run Tengine with ACL GPU, set the **ACL_ROOT** to ComputeLibrary directory in android_config.txt, and set **CONFIG_ACL_GPU** to **ON** in the CMakeLists.txt: `option(CONFIG_ACL_GPU  "build acl gpu version" ON)` 

```
vim ~/tengine/CMakeLists.txt
```

### 5. Build tengine
```
cd ~/tengine
mkdir build
cd build
../android_build_armv7.sh or ../android_build_armv8.sh
make -j4
make install
```

### 6. Build example
### 6.1. Unpack opencv
```
cd ~
tar -zxvf opencv.tgz
```
### 6.2. Set the *TENGINE_DIR*, *OpenCV_DIR*, *PROTOBUF_DIR*
If want to run Tengine with OpenBlas, please add the correct blas path in example/android_build_armv7.sh or example/android_build_armv8.sh. `-DBLAS_DIR=/home/usr/openbla020_android`

```
cd ~/tengine/example
vim android_build_armv7.sh or
vim android_build_armv8.sh
```
Make sure the install directory is in your *TENGINE_DIR*.
### 6.3. Build the example    
```
#For armv8:
cp ~/ComputeLibrary/build_64/libarm_compute* ~/android-ndk-r16b/platforms/android-21/arch-arm64/usr/lib/
#For armv7:
cp ~/ComputeLibrary/build_32/libarm_compute* ~/android-ndk-r16b/platforms/android-21/arch-arm/usr/lib/
cd ~/tengine/example
mdkir build
cd build
../android_build_armv7.sh or ../android_build_armv8.sh
make -j4
```
---
##  Run Tengine on android device
### 1. Prepare files
```
cd ~/tengine
./android_pack.sh
cp -rf ./models ./android_pack
```
Other files:
   - images(cat.jpg, bike.jpg or others)
   - examples/build/imagenet_classification/Classify 

<br />
Copy these files into `~/tengine/android_pack`

### 2. Install adb

```
sudo apt-get install android-tools-adb
adb push ./android_pack /data/local/tmp/
```

### 3. Run
```
adb root
adb shell
cd /data/local/tmp/android_pack
chmod u+x Classify 
export LD_LIBRARY_PATH=.
./Classify -i cat.jpg
#if use ACL GPU:
./Classify -i cat.jpg -d acl_opencl
```
Ouput message:
```
0.2763 - "n02123045 tabby, tabby cat"
0.2673 - "n02123159 tiger cat"
0.1766 - "n02119789 kit fox, Vulpes macrotis"
0.0827 - "n02124075 Egyptian cat"
0.0777 - "n02085620 Chihuahua"
```

