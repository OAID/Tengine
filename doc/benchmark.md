# **Tengine  Performance Report**  

## **Revision Record**
|    Date    | Rev |Change Description|Author
| ---------- | --- |---|---|
| 2017-12-29 |  0.1 |Initial version|FeyaHan
| 2018-01-06 |  0.2 |Add multi CPU performance|HaoLuo
| 2018-06-14 |  0.3 |Add ACL_GPU performance| Chunying


---

## **Catalog**

#### [Test Environment](benchmark.md#test-environment-1)
#### [Test](benchmark.md#test-1)
#### [Performance](benchmark.md#performance-1)

---



## Test Environment
- Tengine : v0.3
- Broad : ROCK960
- CPU : Rockchip RK3399. 

    *   Dual-core Cortex-A72 up to 2.0GHz (real frequency is 1.8GHz); 

    *   Quad-core Cortex-A53 up to 1.5GHz (real frequency is 1.4GHz).  

- GPU : Mali T864 (800MHz).  
- Operating System : Ubuntu 16.04.


---

## Test 

### Step1. install Tengine

    For more information about the build of Tengine, please refer to the documentation of [install](install.md) 

### Step2. lock the cpu frequency at maximum
```bash
    #switch to root user
    > sudo su 

    #check which available policy, policy4 for A72, policy0 for A53
    > cat  /sys/devices/system/cpu/cpufreq/policy4/scaling_available_governors  

    #set performance policy
    > echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor 
    
    #check cpu frequency
    > cat /sys/devices/system/cpu/cpufreq/policy4/cpuinfo_cur_freq
```

### Step3: test bench_sqz, bench_mobilenet

* **get model**

    You can get the models from [Tengine model zoo](https://pan.baidu.com/s/1LXZ8vOdyOo50IXS0CUPp8g),the pass word is `57vb`.
    And then, put the "mobilenet.caffemodel" "mobilenet_deploy.prototxt" "squeezenet_v1.1.caffemodel" "sqz.prototxt" in `~/tengine/models`

* **set device**
    1. use ACL_GPU

        For how to build tengine with ACL_GPU, see [acl_driver.md](acl_driver.md).  
        You can run the test as 

        ```
        ./build/tests/bin/bench_sqz -d acl_opencl
        ./build/tests/bin/bench_mobilenet -d acl_opencl
        ```
    2. use CPU: single-core/multi-cores

        To assign on different cpu core, there are two methods:

        - `export TENGINE_CPU_LIST=0,1,2,3`
        - `tests/bin/bench_sqz –p 0,1,2,3`

        For rk3399, cpu(0-3) are A53, cpu(4-5) are A72. 
        
        - 1A72 `tests/bin/bench_sqz –p 4`
        - 2A72 `tests/bin/bench_sqz –p 4,5`
        - 1A53 `tests/bin/bench_sqz –p 0`
        - 4A53 `tests/bin/bench_sqz –p 0,1,2,3`

## Performance


|       | SqueezeNet(ms) |Mobilenet (ms) |
| ---------- | ---|---|
| rk3399(1*A72) | 91.2 |122.1  |
| rk3399(2*A72) | 51.2  |65.4 |
| rk3399(1*A53) | 232.5 |323.6 |
| rk3399(4*A53) | 79.2  |96.3  |
| ACL(GPU)| 61.4| 95.9|


Notes:<br>
(1) We run N=100 times per test case.<br>
(2) We take the average time of N repeats.
                  
---






