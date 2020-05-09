#!/bin/bash 
  
CPU_NUMS=4

if [ $# -gt 0 ]
then
	CPU_NUMS=$1
fi

##### android armv7
mkdir -p build-android-armv7
pushd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="armeabi-v7a" -DANDROID_PLATFORM=android-22 -DANDROID_STL=c++_shared -DANDROID_ARM_NEON=ON -DCONFIG_ARCH_ARM32=ON -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE ..
make -j$CPU_NUMS && make install
popd

##### android aarch64
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-22 -DANDROID_STL=c++_shared -DANDROID_ARM_NEON=ON -DCONFIG_ARCH_ARM64=ON -DCONFIG_ARCH_ARM8_2=ON -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE ..
make -j$CPU_NUMS && make install
popd

##### android aarch64 armv8.2 for fp16
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-22 -DANDROID_STL=c++_shared -DANDROID_ARM_NEON=ON -DCONFIG_ARCH_ARM64=ON -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE ..
make -j$CPU_NUMS && make install
popd

##### linux of himix200
mkdir -p build-himiv200-linux
pushd build-himiv200-linux
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix200.toolchain.cmake ..
make -j$CPU_NUMS && make install
popd

##### linux of hisiv500
mkdir -p build-hisiv500-linux
pushd build-hisiv500-linux
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv500_linux_cross.config ..
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

##### linux x86_convert_tools
mkdir -p build-linux-x86
pushd build-linux-x86
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/x86_convert_tool.gcc.toolchain.cmake ..
make -j$CPU_NUMS && make install
popd

##### linux for arch64-linux-gnu toolchain
mkdir -p build-aarch64-linux-gnu_gpu
pushd build-aarch64-linux-gnu_gpu
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake \
-DCONFIG_ACL_OPENCL=ON  -DACL_ROOT=/home/cmeng/ComputeLibrary ..
make -j$CPU_NUMS && make install
popd

##### linux x86 ncnn serializer cv api
mkdir -p build-linux-x86-ncnn-cv
pushd build-linux-x86-ncnn-cv
cmake -DCONFIG_ARCH_X86=ON -DCONFIG_ARCH_X86_AVX=ON -DCONFIG_NCNN_SERIALIZER=ON -DOPENCV_EXAMPLE=ON ..
make -j$CPU_NUMS && make install
popd
