# Source Code Compilation (Vulkan)

## Brief

Vulkan is a new generation graphics and compute API that provides high-efficiency, cross-platform access to modern GPUs used in a wide variety of devices from PCs and consoles to mobile phones and embedded platforms.

## How to build

### Build for Linux

On Debian, Ubuntu , you can install vulkan sdk: 

```bash
sudo apt instal libvulkan-dev
```

Download Vulkan SDK

```bash
# download vulkan sdk
$ wget https://sdk.lunarg.com/sdk/download/1.1.114.0/linux/vulkansdk-linux-x86_64-1.1.114.0.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.1.114.0.tar.gz
$ tar -xf vulkansdk-linux-x86_64-1.1.114.0.tar.gz

# setup env
$ export VULKAN_SDK=`pwd`/1.1.114.0/x86_64
```

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p build-linux-vulkan
$ cd build-linux-vulkan
$ cmake -DTENGINE_ENABLE_VULKAN=ON ..

$ make -j4
$ make install
```
### Build Android Library

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p build-android-aarch64-vulkan
$ cd build-android-aarch64-vulkan
$ cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-24 -DTENGINE_ENABLE_VULKAN=ON ..

$ make -j4
$ make install
```

## Demo

```
violet:/data/local/tmp/tengine/vulkan $ ./tm_classification_vulkan -m mobilenet.tmfile -i cat.jpg -g 224,224 -s 0.017,0.017,0.017 -r 10
start to run register cpu allocator
start to run register vk allocator
Mean value not specified, use default   104.0, 116.7, 122.7
tengine-lite library version: 1.0-dev

model file : mobilenet.tmfile
image file : cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 10 times, thread 1, avg time 114.83 ms, max_time 169.14 ms, min_time 107.62 ms
--------------------------------------
8.574147, 282
7.880115, 277
7.812572, 278
7.286460, 263
6.357491, 281
--------------------------------------
```

