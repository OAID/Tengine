# Cross-compile Arm64 OHOS version
## Install DevEco Studio and OHOS NDK
Download and install DevEco Studio, portal (https://developer.harmonyos.com/cn/develop/deveco-studio#download).If no developer's account, huawei need to [HarmonysOS application development portal] (https://developer.harmonyos.com/cn/home) registered.
Open DevEco Studio, Configure (or File) ->Settings -> Appearance & Behavior -> System Settings -> HarmonyOS SDK, check and download Native to complete the OHOS NDK download.
## 2 Prepare OHOS NDK cmake toolchain file
Ohos.toolchain-cmake the file can be found with $OCOMPANIES NDK/build/cmake,Such as E:\soft\Huawei\SDK\native\3.0.0.80\build\cmake\ohos.toolchain.cmake
(optional) delete the debug compilation parameters, the binary volume, the method and the android the NDK same [android - the NDK issue] (https://github.com/android-ndk/ndk/issues/243)
```
# with editor opens $ANDROID_NDK/build/cmake/android. The toolchain. Cmake
# delete the line "-g"
list(APPEND ANDROID_COMPILER_FLAGS
- g
- DANDROID
```

## 3 Download Tengine Lite source code
```bash
git clone https://github.com/OAID/Tengine.git tengine-lite
```
## 4 Build Tengine Lite
Arm64 OHOS compile script as follows (Windows)
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