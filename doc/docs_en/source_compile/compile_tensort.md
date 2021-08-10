# Source Code Compilation (TensorRT)

## Brief

Todo

## How to build

### Build for Linux

On Ubuntu

### build
```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p build-linux-trt
$ cmake -DTENGINE_ENABLE_TENSORRT=ON \
    -DTENSORRT_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
    -DTENSORRT_LIBRARY_DIR=/usr/lib/aarch64-linux-gnu ..

$ make -j4
$ make install
```
