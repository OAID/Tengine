# ARM Compute Library Driver

[![GitHub license](http://OAID.github.io/pics/apache_2.0.svg)](../LICENSE)


[Tengine](https://github.com/OAID/Tengine) is already supported **ARM Comporte 
Library (ACL)**. This version ,ACL Driver only can use OpenCL device.


## 1.How to enable Tengine with ACL Driver

### 1.1 build ACL

    ```
    git clone https://github.com/ARM-software/ComputeLibrary.git    
    git checkout v18.05
    scons Werror=1 -j4 debug=0 asserts=1 neon=0 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a
    ```


### 1.2 build Tengine with ACL

    Enable ACL_GPU support in Tegnine `makefile.config`.

    ```
    CONFIG_ACL_GPU=y
    ACL_ROOT=/home/firefly/ComputeLibrary
    ``` 
    if you already build tengine, remember to remove `rm -r build/driver` and then rebuild 
    ```
    make
    ```

## 2.How to use ACL Driver    

If you want to use ACL Driver to run a graph, you need to explicitly set default device by ACL Device. You can see how to explicitly set device in[tests/bin/bench_sqz.cpp](../tests/bin/bench_sqz.cpp).

```
    ...
    while((res=getopt(argc,argv,"d:f:r:"))!=-1)
    {  
      switch(res)
      {  
         case 'd': 
            device = optarg;
            break;
    ...
    if("" != device)
        set_default_device( device.c_str());
    ...

```

### Running: 

>> Example:  ./build/tests/bin/bench_sqz -d acl_opencl



