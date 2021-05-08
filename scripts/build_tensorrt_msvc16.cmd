@ECHO OFF
@SETLOCAL

:: Set CMake root path
@SET CMake_ROOT=<your-cmake-root_path, such as D:/Program/CMake/bin>

:: Set CUDA root path
@SET CUDA_ROOT=<your-cuda-root_path, such as D:/Program/NVIDIA/Toolkit/v10.2>

:: Set TensorRT root path
@SET TensorRT_ROOT=<your-tensorrt-root_path, such as D:/Program/NVIDIA/TensorRT/5.1.5.0-10.1-7.5>


:: Set ninja.exe and nvcc.exe
@SET CMAKE_EXE=%CMake_ROOT%/cmake.exe
@SET PATH=%CMake_ROOT%;%CUDA_ROOT%/bin;%TensorRT_ROOT%/bin;%PATH%

mkdir build-msvc16-tensorrt
pushd build-msvc16-tensorrt
%CMAKE_EXE% -G "Visual Studio 16 2019" -A x64       ^
    -DTENGINE_OPENMP=OFF                            ^
    -DTENGINE_ENABLE_TENSORRT=ON                    ^
    -DCUDA_INCLUDE_DIR=%CUDA_ROOT%/include          ^
    -DTENSORRT_INCLUDE_DIR=%TensorRT_ROOT%/include  ^
    -DCUDA_LIBRARY_DIR=%CUDA_ROOT%/lib/x64          ^
    -DTENSORRT_LIBRARY_DIR=%TensorRT_ROOT%/lib      ^
    ..
pause
%CMAKE_EXE% --build . --parallel %NUMBER_OF_PROCESSORS%
%CMAKE_EXE% --build . --target install
popd


@ENDLOCAL
