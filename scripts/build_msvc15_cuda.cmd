@ECHO OFF
@SETLOCAL

:: Set CMake root path
@SET CMake_ROOT=<your-cmake-root_path, such as D:/Program/CMake/bin>
@SET CUDA_ROOT=<your-cuda-root_path, such as D:/Program/NVIDIA/Toolkit/v10.2>

:: Set ninja.exe and nvcc.exe
@SET CMAKE_EXE=%CMake_ROOT%/cmake.exe
@SET PATH=%CMake_ROOT%;%PATH%

mkdir build-msvc15-cuda
pushd build-msvc15-cuda
%CMAKE_EXE% -G "Visual Studio 15 2017 Win64"        ^
    -DTENGINE_OPENMP=OFF                            ^
    -DTENGINE_ENABLE_CUDA=ON                        ^
    -DCUDAToolkit_ROOT=%CUDA_ROOT%                  ^
    ..
%CMAKE_EXE% --build . --parallel %NUMBER_OF_PROCESSORS%
%CMAKE_EXE% --build . --target install
popd

@ENDLOCAL
