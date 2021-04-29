# Tengine Lite OpenCL User Manual

## Brief

Todo

## How to build

### Setup Tengine-Lite project ROOT_PATH
```
$ export ROOT_PATH={Path of tengine-lite}
```
### Build

`-DOPENCL_LIBRARY: libOpenCL.so 文件夹路径。可通过 <sudo find /usr -name "libOpenCL.so"> 命令查询`

`-DOPENCL_INCLUDE_DIRS：指定CL/cl.h 路径。可通过 <sudo find /usr -name "cl.h"> 命令查询`

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p build-linux-opencl
$ cmake \
-DTENGINE_ENABLE_OPENCL=ON \
-DOPENCL_LIBRARY=/usr/lib/aarch64-linux-gnu \
-DOPENCL_INCLUDE_DIRS=/usr/include ..

$ make -j4
$ make install
```

## Demo

```
Todo
```
