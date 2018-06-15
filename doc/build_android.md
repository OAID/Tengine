## 1.Download Tengine poject
```
git clone https://github.com/OAID/tengine/ 
```
## 2.Download OpenCV protobuf OpenBLAS and Android ndk
    
Download the below files from [Tengine Android build](https://pan.baidu.com/s/1RPHK_ji0LlL3ztjUa893Yg),the pass word is *ka6a*:
```
  - android-ndk-r16-linux-x86_64.zip
  - openblas020_android.tgz
  - opencv-3.4.0-android-sdk.zip
  - protobuf_lib.tgz
```
## 3.Unpack the protobuf  android-ndk-r16-linux-x86_64.zip and openblas 
```
unzip android-ndk-r16-linux-x86_64.zip
tar -zxvf protobuf_lib.tgz
tar -zxvf openblas020_android.tgz
```
## 4.Set the *NDK PATH*,*PROTOBUF_DIR*,*BLAS PATH* and *CONFIG_ARCH_TYPE*
if you want to build android for armv7,set the CONFIG_ARCH_TYPE:**ARMv7**
,otherwise ,set the CONFIG_ARCH_TYPE:**ARMv8**
```
vim  ~/tengine/android_config.txt
```

## 5. Build tengine
```
cd ~/tengine
mkdir build
cd build
../android_build_armv7.sh or ../android_build_armv8.sh
make -j4
make install
```
if you want to run tengine with openblas, remove the DCONFIG_ARCH_ARM64 or DCONFIG_ARCH_ARM32, set the** -DCONFIG_ARCH_BLAS=ON \**,and you must set the correct** BLAS_DIR** in file **android_config.txt** 
## 6. Build example
### 6.1 Unpack opencv
```
cd ~
unzip opencv-3.4.0-android-sdk.zip
```
### 6.2 Set the *TENGINE_DIR*, *OpenCV_DIR*, *PROTOBUF_DIR*
```
cd ~/tengine/example
vim android_build_armv7.sh or
vim android_build_armv8.sh
```
make sure the install directory is in your *TENGINE_DIR*
### 6.3 Build the example    
```
cd ~/tengine/example
mkdir build
cd build
../android_build_armv7.sh or ../android_build_armv8.sh
make -j4
```
## 7. Install adb,adb_driver

## 8. Test squeezenet on Tengine android
### 8.1 prepare files
```
cd ~/tengine
./android_pack.sh
cp -rf ./models ./android_pack
```
Other files:
   - images(cat.jpg,bike.jpg or others)
   - examples/build/imagenet_classification/Classify 

<br />
Put these files into **~/tengine/android_pack**
```
adb push ./android_pack /data/local/tmp/
```

### 8.2 Run
```
adb root
adb shell
cd /data/local/tmp
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
`
