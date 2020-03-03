# **Build Tengine for android**  
The document describes how to build Tengine for android.

##  Building Tengine for android

### 1. Download Tengine project
```
git clone https://github.com/OAID/tengine/
```
### 2. Download Android ndk, OpenBLAS, Protobuf and ComputeLibrary

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
### 4. Set *ANDROID_NDK* in system environment
```
If run Tengine with Openblas turn on OPEN_BLAS and set OPENBLAS_LIB_PATH,OPENBLAS_INCLUDE_PATH in file example_config/arm_android_cross.config.
If want to build with Armv8 please set *ARCH_TYPE=Arm64* in file example_config/arm_android_cross.config.  

```

### 5. Build tengine
```
cd ~/tengine
./android_build.sh example_config/arm_android_cross.config 
```

### 6. Build example
### 6.1. Set the *TENGINE_DIR*
If want to run Tengine with OpenBlas, please add the correct blas path in example/android_build_armv7.sh or example/android_build_armv8.sh. `-DBLAS_DIR=/home/usr/Openblas_0220_android`
```
cd ~/tengine/example
vim android_build_armv7.sh or  vim android_build_armv8.sh
```
Make sure the install directory is in your *TENGINE_DIR*.
### 6.3. Build the example    
```
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
```
Ouput message:
```
0.2763 - "n02123045 tabby, tabby cat"
0.2673 - "n02123159 tiger cat"
0.1766 - "n02119789 kit fox, Vulpes macrotis"
0.0827 - "n02124075 Egyptian cat"
0.0777 - "n02085620 Chihuahua"
```

