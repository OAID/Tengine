# 交叉编译 Arm64 OHOS（鸿蒙系统）版本

## 1 安装 DevEco Studio 和 OHOS NDK

下载安装 DevEco Studio，[传送门](https://developer.harmonyos.com/cn/develop/deveco-studio#download)。若没有华为开发者账号，需到[HarmonysOS应用开发门户](https://developer.harmonyos.com/cn/home)注册。

打开 DevEco Studio，Configure（或File）-> Settings -> Appearance & Behavior -> System Settings -> HarmonyOS SDK，勾选并下载 Native，完成 OHOS NDK 下载。

## 2 准备 OHOS NDK cmake toolchain 文件

ohos.toolchain.cmake 这个文件可以从 $OHOS_NDK/build/cmake 找到，例如 E:\soft\Huawei\sdk\native\3.0.0.80\build\cmake\ohos.toolchain.cmake

(可选) 删除debug编译参数，缩小二进制体积，方法和 android ndk相同 [android-ndk issue](https://github.com/android-ndk/ndk/issues/243)

```
# 用编辑器打开 $ANDROID_NDK/build/cmake/android.toolchain.cmake
# 删除 "-g" 这行
list(APPEND ANDROID_COMPILER_FLAGS
  -g
  -DANDROID
```

## 3 下载 Tengine Lite 源码

```bash
git clone https://github.com/OAID/Tengine.git tengine-lite
```

## 4 编译 Tengine Lite

Arm64 OHOS 编译脚本如下（Windows）

`build/ohos-arm64-v8a.bat`:
```bash
@ECHO OFF
@SETLOCAL

:: Set OHOS native toolchain root
@SET OHOS_NDK=<your-ndk-root_path, such as D:/Program/DevEcoStudio/SDK/native/2.0.1.93>


:: Set ninja.exe and cmake.exe
@SET NINJA_EXE=%OHOS_NDK%/build-tools/cmake/bin/ninja.exe
@SET CMAKE_EXE=%OHOS_NDK%/build-tools/cmake/bin/cmake.exe
@SET PATH=%OHOS_NDK%/llvm/bin;%OHOS_NDK%/build-tools/cmake/bin;%PATH%

mkdir build-ohos-armeabi-v7a
pushd build-ohos-armeabi-v7a
%CMAKE_EXE% -G Ninja -DCMAKE_TOOLCHAIN_FILE="%OHOS_NDK%/build/cmake/ohos.toolchain.cmake"  -DCMAKE_MAKE_PROGRAM=%NINJA_EXE%  -DOHOS_ARCH="armeabi-v7a" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON .. 
%CMAKE_EXE% --build . --parallel %NUMBER_OF_PROCESSORS%
%CMAKE_EXE% --build . --target install
popd

mkdir build-ohos-arm64-v8a
pushd build-ohos-arm64-v8a
%CMAKE_EXE% -G Ninja -DCMAKE_TOOLCHAIN_FILE="%OHOS_NDK%/build/cmake/ohos.toolchain.cmake"  -DCMAKE_MAKE_PROGRAM=%NINJA_EXE%  -DOHOS_ARCH="arm64-v8a" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON .. 
%CMAKE_EXE% --build . --parallel %NUMBER_OF_PROCESSORS%
%CMAKE_EXE% --build . --target install
popd


@ENDLOCAL
```
