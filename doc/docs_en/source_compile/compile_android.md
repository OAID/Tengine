# Source Code Compilation (Android)

## Install Android NDK

[Android NDK url](http://developer.android.com/ndk/downloads/index.html)

## Prepare Android Toolchain Files

(optional) delete debug compilation parameters to reduce the binary volume [android-ndk issue](https://github.com/android-ndk/ndk/issues/243)，The file android.toolchain.cmake can be found from $ANDROID_NDK/build/cmake:

```cmake
# vi $ANDROID_NDK/build/cmake/android.toolchain.cmake
# delete line "-g"
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
```

## Downlad Tengine Source Code

```bash
git clone -b tengine-lite https://github.com/OAID/Tengine.git Tengine
```

## Compile Tengine

### Arm64 Android

```bash
mkdir build-android-aarch64
cd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build-android-aarch64/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 ..
make
make install
```

### Arm32 Android

```bash
mkdir build-android-armv7
cd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-19 ..
make
make install
```
