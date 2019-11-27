Winograd in Tengine
====================
Tengine use winograd algorithm to accelerate convolution computations.


## Winograd Option
* `default set to OPEN winograd`
* if want to close winograd, `export NO_WINO=1`
* if want to re-open winograd, `unset NO_WINO`
  
## Example on RK3399
This example tests the winograd acceleration on 1A53 cpu on RK3399. 


* Set CPU to 1A53:
    > export TENGINE_CPU_LIST=0
* run with default settings (with winograd ON)
    > ./build/tests/bin/bench_sqz -r10

    Here is the results. It costs `128ms`.
    ```
    
    ENV SET: [0]
    run-time library version: 1.6.1-github
    REPEAT COUNT= 10
    Repeat [10] time 128022.50 us per RUN. used 1280225 us
    0.2763 - "n02123045 tabby, tabby cat"
    0.2673 - "n02123159 tiger cat"
    0.1766 - "n02119789 kit fox, Vulpes macrotis"
    0.0827 - "n02124075 Egyptian cat"
    0.0777 - "n02085620 Chihuahua"
    ```


* run without winograd 
    ```
    export NO_WINO=1
    ./build/tests/bin/bench_sqz -r10
    ```
    Here is the results. It costs `168ms`.
    ```
    ENV SET: [0]
    run-time library version: 1.6.1-github
    REPEAT COUNT= 10
    Repeat [10] time 168045.09 us per RUN. used 1680451 us
    0.2763 - "n02123045 tabby, tabby cat"
    0.2673 - "n02123159 tiger cat"
    0.1766 - "n02119789 kit fox, Vulpes macrotis"
    0.0827 - "n02124075 Egyptian cat"
    0.0777 - "n02085620 Chihuahua"
    ```