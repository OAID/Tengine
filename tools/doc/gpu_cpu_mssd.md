* Platform: RK3399 (Linux)
  
1. build ACL
    ```
    git clone https://github.com/ARM-software/ComputeLibrary.git
    git checkout v19.02
    scons Werror=1 -j4 debug=0 asserts=0 neon=0 opencl=1 embed_kernels=1 os=linux arch=arm64-v8a
    ```

2. set GPU frequency
    ```
    sudo su
    echo "performance" >/sys/devices/platform/ff9a0000.gpu/devfreq/ff9a0000.gpu/governor
    cat /sys/devices/platform/ff9a0000.gpu/devfreq/ff9a0000.gpu/cur_freq
    ```
    the GPU frequency set to `800000000`

3. build tengine with ACL

    - edit *makefile.config*
    ```
    CONFIG_ACL_GPU=y
    ACL_ROOT=/home/firefly/ComputeLibrary
    ```
    - build
    ```
    make -j4 
    make install
    ```
4. build mssd
    - download model from [Tengine model zoo](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q) (psw: hhgc) to `~/tengine/models/`:
    - build
    ```
    cd example/mobilenet_ssd
    cmake -DTENGINE_DIR=/home/firefly/tengine .
    make 
    ```
5. run mssd
    ```
    export GPU_CONCAT=0           # disable gpu run concat,     avoid frequent data transfer between cpu and gpu
    export ACL_FP16=1             # enable gpu fp16
    export REPEAT_COUNT=50        # repeat count to run mssd,     get avg time
    export ACL_NHWC=1             # run acl graph on NHWC layout
    taskset 0x1 ./MSSD -d acl_opencl          # -d acl_opencl to use gpu, taskset 0x1 to bind CPU0(A53)
    ```
    It costs `118ms` to run mobilenetssd using `GPU + 1A53`
    ```
    repeat 50 times, avg time per run is 118.33 ms
    detect result num: 3
    dog     :100%
    BOX:( 138.332 , 209.167 ),( 324.072 , 541.449 )
    car     :100%
    BOX:( 465.977 , 72.2475 ),( 688.351 , 171.257 )
    bicycle :100%
    BOX:( 107.306 , 141.191 ),( 573.994 , 415.038 )
    ```
