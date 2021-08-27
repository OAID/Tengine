# 源码编译（CUDA）

## How to build

### Build for Linux

On Ubuntu

### setup nvcc enva
```bash
$ export CUDACXX=/usr/local/cuda/bin/nvcc
```
### build
```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p build-linux-cuda && cd build-linux-cuda
$ cmake -DTENGINE_ENABLE_CUDA=ON ..

$ make -j4
$ make install
```
