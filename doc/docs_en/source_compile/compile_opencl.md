# Source Code Compilation (OpenCL)

## Brief

Todo

## How to build

### Setup Tengine-Lite project ROOT_PATH
```
$ export ROOT_PATH={Path of tengine-lite}
```
### Build

`-DOPENCL_LIBRARY: Specify libOpenCL.so directory path. Query by <sudo find /usr -name "libOpenCL.so"> `

`-DOPENCL_INCLUDE_DIRS：Specify the CL/cl.h path. Query by <sudo find /usr -name "cl.h">`

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

