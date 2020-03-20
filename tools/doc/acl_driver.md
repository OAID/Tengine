# ARM Compute Library Driver

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](../LICENSE)


[Tengine](https://github.com/OAID/Tengine) has already supported the driver of ARM Compute 
Library. This version, ACL Driver only can use OpenCL device.


## 1.How to use ACL Driver in Tengine

### build ACL

```
git clone https://github.com/ARM-software/ComputeLibrary.git
git checkout v19.02
scons Werror=1 -j4 debug=0 asserts=1 neon=0 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a
```

### build Tengine with ACL

To enable ACL driver support in Tengine, please turn on `CONFIG_ACL_OPENCL` and `ACL_ROOT` in android_config.txt or linux_build.sh

```
build Android:
# Enable GPU support by Arm Computing Library
CONFIG_ACL_OPENCL:ON

# Set the path of ACL
ACL_ROOT=/home/firefly/ComputeLibrary

```

```
build Linux:
# Enable GPU support by Arm Computing Library
CONFIG_ACL_OPENCL=ON

# Set the path of ACL
ACL_ROOT=/home/firefly/ComputeLibrary

```


## 2.How to use ACL Driver    

Normally, the ACL Driver is not the default driver for Tengine to run a graph.

If you want to use ACL Driver to run a graph, please change the default device by using `set_graph_device()` in your application.

Below is an example about using ACL Driver.

```
Example:
    ./build/tests/bin/bench_sqz -d acl_opencl
```
`acl_opencl` is the ACL device name.

```
tests/bin/bench_sqz.cpp:
    std::string device;
    ...
    while((res = getopt(argc, argv, "p:d:r:")) != -1)
    {
        switch(res)
        {
            ...
            case 'd':
                device = optarg;
                break;
            ...
        }
    }
    ...
    if(!device.empty())
        set_graph_device(graph, device.c_str());
    ...

```
The whole source code, please refer to [tests/bin/bench_sqz.cpp](../tests/bin/bench_sqz.cpp).

