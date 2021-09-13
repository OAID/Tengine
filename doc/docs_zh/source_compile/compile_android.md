# 源码编译（Android）

## 安装 Android NDK

[Android NDK 下载地址](http://developer.android.com/ndk/downloads/index.html)

## 准备 android toolchain 文件

(可选) 删除debug编译参数，缩小二进制体积 [android-ndk issue](https://github.com/android-ndk/ndk/issues/243) ，android.toolchain.cmake 这个文件可以从 $ANDROID_NDK/build/cmake 找到：

```cmake
# 用编辑器打开 $ANDROID_NDK/build/cmake/android.toolchain.cmake
# 删除 "-g" 这行
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
  ...)
```

## 下载 Tengine 源码

```bash
git clone -b tengine-lite https://github.com/OAID/Tengine.git Tengine
```

## 编译 Tengine

### Arm64 Android

```bash
cd Tengine
mkdir build-android-aarch64
cd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-21 ..
make -j$(nproc)
make install
```

### Arm32 Android

```bash
cd Tengine
mkdir build-android-armv7
cd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-19 ..
make -j$(nproc)
make install
```
