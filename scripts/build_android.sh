#!/bin/bash

export ANDROID_NDK=<your-ndk-root_path, such as /home/user/libraries/android-ndk-r15c>


##### android armv7
mkdir -p build-android-armv7
pushd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-19 ..
cmake --build . --parallel 1 && cmake --build . --target install
popd


##### android aarch64
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 ..
cmake --build . --parallel 1 && cmake --build . --target install
popd


##### android x86
mkdir -p build-android-x86
pushd build-android-x86
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="x86" -DANDROID_PLATFORM=android-19 ..
cmake --build . --parallel 1 && cmake --build . --target install
popd


##### android x86
mkdir -p build-android-x86_64
pushd build-android-x86_64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="x86_64" -DANDROID_PLATFORM=android-21 ..
cmake --build . --parallel 1 && cmake --build . --target install
popd

