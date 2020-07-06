#!/bin/bash 
  
CPU_NUMS=4

if [ $# -gt 0 ]
then
	CPU_NUMS=$1
fi

##### android armv7
mkdir -p build-android-armv7
pushd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_ARM_NEON=ON -DANDROID_PLATFORM=android-19 ..
make -j$CPU_NUMS && make install
popd

##### android aarch64
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 ..
make -j$CPU_NUMS && make install
popd

##### linux of arm-linux-gnueabi toolchain
mkdir -p build-arm-linux-gnueabi
pushd build-arm-linux-gnueabi
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabi.toolchain.cmake ..
make -j$CPU_NUMS && make install
popd

##### linux of arm-linux-gnueabihf toolchain
mkdir -p build-arm-linux-gnueabihf
pushd build-arm-linux-gnueabihf
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake ..
make -j$CPU_NUMS && make install
popd

##### linux for aarch64-linux-gnu toolchain
mkdir -p build-aarch64-linux-gnu
pushd build-aarch64-linux-gnu
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j$CPU_NUMS && make install
popd

##### linux of hisiv200
mkdir -p build-hisiv200-linux
pushd build-hisiv200-linux
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm.himix200.toolchain.cmake ..
make -j$CPU_NUMS && make install
popd

##### linux for native
mkdir -p build-linux
pushd build-linux
cmake ..
make -j$CPU_NUMS && make install
popd
