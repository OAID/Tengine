# Tengine uses OpenCL for deployment
## Brief

Todo

## How to build

### Setup Tengine-Lite project ROOT_PATH
```
$ export ROOT_PATH={Path of tengine-lite}
```
### Build

`-DOPENCL_LIBRARY: libOpenCL.so path.It can be queried through the <sudo find /usr -name "libOpenCL.so"> command`

`-DOPENCL_INCLUDE_DIRS: Specify the CL/cl.h path.It can be queried through the <sudo find /usr -name "cl.h"> command`

```bash
$ cd <tengine-lite-root-dir>
$ mkdir -p build-linux-opencl
$ cmake \
-DTENGINE_ENABLE_OPENCL=ON \
-DOPENCL_LIBRARY=/usr/lib/aarch64-linux-gnu/libOpenCL.so \
-DOPENCL_INCLUDE_DIRS=/usr/include ..

$ make -j4
$ make install
```

## Demo

```
Todo
```
