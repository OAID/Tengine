# 源码编译（libTorch）

## How to build

### Build for Linux

"-DPROJECT_BINARY_DIR" : Specify any path to create "detect_cuda_version.cc" file

### build
```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p build-linux-torch && cd build-linux-torch
$ cmake -DTENGINE_ENABLE_TORCH=ON \
        -DCMAKE_PREFIX_PATH=/path/to/libtorch \
        -DPROJECT_BINARY_DIR=..... \
        ..

$ make -j4
$ make install
```
