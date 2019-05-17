# **Tengine  Performance Report**  

## **Revision Record**
|    Date    | Rev |Change Description|Author
| ---------- | --- |---|---|
| 2018-12-27 |  0.9 |update newest benchmark|ZhangRui/LuoHao


---

## **Catalog**

#### [**Test Environment**](benchmark.md#test-environment-1)
#### [**Test Steps**](benchmark.md#test-steps-1)
#### [**Performance**](benchmark.md#performance-1)

---


## **Test Environment**
- Tengine : 0.9.0
- Broad : **Firefly-3399 (RK3399), TinkerBoard (RK3288)**
- Operating System : Ubuntu 16.04.


---

## **Test Steps**

### Step1. **install Tengine**

For more information about the build of Tengine, please refer to the documentation of [install](install.md).

### Step2. **lock the cpu frequency at maximum**

Please set the scaling governer into performance. Below is an example to set the big core of RK33399 to performance mode.

```bash
> sudo su #switch to root user
> cat  /sys/devices/system/cpu/cpufreq/policy4/scaling_available_governors   #check which available policy, note that policy4 is for A72 on RK3399 and policy0 is for A53
conservative ondemand userspace powersave interactive performance
> echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor #set performance policy
> cat /sys/devices/system/cpu/cpufreq/policy4/cpuinfo_cur_freq     #check cpu frequency
1800000
```

### Step3: **test benchmark squeezenet_v1.1 and mobilenet_v1**

* **get model**

    You can get the models from [Tengine model zoo](https://pan.baidu.com/s/1Ar9334MPeIV1eq4pM1eI-Q) (psw: hhgc)
    And then, put the "mobilenet.caffemodel", "mobilenet_deploy.prototxt", "squeezenet_v1.1.caffemodel", "sqz.prototxt" in `~/tengine/models`.

* **set CPU**

    By setting the environment variable `TENGINE_CPU_LIST`, different working CPUs can be set.
    
    For RK3399:
    ```
        1 A72: export TENGINE_CPU_LIST=5
        2 A72: export TENGINE_CPU_LIST=4,5
        1 A53: export TENGINE_CPU_LIST=2
        4 A53: export TENGINE_CPU_LIST=0,1,2,3
        
    ```
    For RK3288:
    ```
        1 A17: export TENGINE_CPU_LIST=2
        4 A17: export TENGINE_CPU_LIST=0,1,2,3
        
    ```

* **run int8/float32 inference**

    By default, Tengine run inference as **float32**. To run int8 inference, you need to set the env_variable `KERNEL_MODE` as `2`. And set it back to `0` to run float32 inference.
    ```
    export KERNEL_MODE=2  # run int8 inference
    export KERNEL_MODE=0  # run float32 inference
    ```

---

## Performance

### RK3399 

#### MobileNet

|   | Float32(ms) | INT8（ms） |
| ---------- | ---|---|
| rk3399(1*A72) | 111.8 |80.1  |
| rk3399(2*A72) | 63.7  |46.5  |
| rk3399(1*A53) | 259.6 |198.0 |
| rk3399(4*A53) | 81.6  |63.7  |


#### SqueezeNet
|   | Float32(ms) | INT8（ms） |
| ---------- | ---|---|
| rk3399(1*A72) | 79.4  |60.4 |
| rk3399(2*A72) | 49.3  |37.6 |
| rk3399(1*A53) | 177.0 |151.2 |
| rk3399(4*A53) | 68.4  |59.6 |


### RK3288

#### MobileNet

|   | Float32(ms) | INT8（ms） |
| ---------- | ---|---|
| rk3288(1*A17) | 201 |111  |
| rk3288(4*A17) | 67.4 |40  |


#### SqueezeNet
|   | Float32(ms) | INT8（ms） |
| ---------- | ---|---|
| rk3288(1*A17) | 142 |88 |
| rk3288(4*A17) | 55  |35 |

Notes:<br>
(1) We take the average time of N repeats.<br>
(2) We run N=100 times per test case.<br>


