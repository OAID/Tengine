# **TEngine  Performance Report**  

## **Revision Record**
|    Date    | Rev |Change Description|Author
| ---------- | --- |---|---|
| 2017-12-29 |  0.1 |Initial version|FeyaHan


---

## **Catalog**


#### [**Test Environment**](benchmark.md#test-environment-1)
#### [**Test Steps**](benchmark.md#test-steps-1)
#### [**Performance**](benchmark.md#performance-1)
#### [**How we get the time cost**](benchmark.md#how-we-get-the-time-cost-1)


---



## **Test Environment**
- TEngine : v0.1  
- Hardware SoC : Rockchip RK3399.  
- CPU : 

    *   Dual-core Cortex-A72 up to 2.0GHz (real frequency is 1.8GHz); 

    *   Quad-core Cortex-A53 up to 1.5GHz (real frequency is 1.4GHz).  

- GPU : Mali T864 (800MHz).  
- Operating System : Ubuntu 16.04.


---

## **Test Steps**

### Step1. **install tengine**

For more information about the build of TEngine, please refer to the documentation of [install](install.md) 

### Step2. **lock the cpu frequency at maximum**
```bash
> sudo su #switch to root user
> cat  /sys/devices/system/cpu/cpufreq/policy4/scaling_available_governors #check which available policy, note that policy4 is for A72 on RK3399 and policy0 is for A53
conservative ondemand userspace powersave interactive performance
> echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor #set performance policy
> cat /sys/devices/system/cpu/cpufreq/policy4/cpuinfo_cur_freq #check cpu frenquency
1800000
```

### Step3: **test benchmark squeezenetv1.1 and mobilenet**

```shell
> taskset -a 0x10 ./build/tests/bin/bench_sqz -r10  
> taskset -a 0x10 ./build/tests/bin/bench_mobilenet -r10 
```

* "taskset -a 0x10" :  We use this to bind our test program to single ARM A72 core, since RK3399 has 4 A53 (0-3) and 2 A72 (4-5).

* r10 :  We run the test program for 10 times.  

* 'bench\_sqz', 'bench\_mobilenet' : Specify the neural network we were testing.


---

## **Performance**

The data in the tables below are in micro second **us**. 

For looking the profile of each layer, we need set an enviroment variable:
```bash
> export PROF_TIME=1
``` 
We run 10 times per test case.

#### **Mobilenet**
| Mobilenet  | TPI |Pooling|Fused.BNScaleRelu|Convolution|
| ---------- | --- |---|---|---|
| TimeElapse | 122856 |380 | 8569| 114249|
| Percentage | 100%|0.03%| 6.97%| 92.99%|
||


#### **Squeezenet**
| Squeezenet | TPI |SoftMax |Convolution |Pooling |Concat|
| ---------- | --- |---  |---         |---      | ---  |
| TimeElapse | 90677 |69 | 85371|2643|2973|
| Percentage | 100% |0.08% | 93.76%|2.90%|3.27%|  
||

All items in the tables are:  
* **TPI** : The average total time for per inference within the whole loops.  
* **SoftMax** : The average SoftMax time consumption for per inference within the whole loops.  
* **Convolution** : The average Convolution time consumption for per inference within the whole loops.  
* **Pooling** : The average Pooling time consumption for per inference within the whole loops.  
* **Dropout** : The average Dropout time consumption for per inference within the whole loops.  
* **Concat** : The average Concat time consumption for per inference within the whole loops. 

----


## **How we get the time cost**  
#### **A simple principle**  
We make preparations including: prepare input data,load model,create runtime graph,set input/output node,setup input/output buffer and prerun graph before we test the benchmark demo.  
We mark the start time t0 and run the demo for loopsize(which was specified by user through terminal) times,finally,we mark the end time t1 and then we get the total time cost by totaltimecost = t1 - t0. Also,we get the average time cost per run by avgtimecost = totaltimecost/loopsize.  

---


<br />

We get the total time cost and the corresponding time spent on each operator within the process of executing the specified neural network   and also the percentage of the time cost for certain operator in the whole run.  

Besides, we jump in to the process of executing the graph node sequence and dump node info such as node index,node name,operator name,input/output shape and Mfops for this node one by one.  

* *total_time* :   
We determine this by adding all the total used time for every single time record node across the graph node sequence.  

* *time_cost for SoftMax/ReLu/Convolution/Pooling/Dropout/Concat* :  
We determine this by adding individually all the record nodes which belong to the same operator across the graph sequence through loopsize run processes.  
We dumped more information by ProfTime's Dump method which was called after we Postran the graph,and we got this :  

* *accum_time* :  
We determine this by adding all the time cost for every single node across the node graph sequence through loopsize run processes.  

* *total_usedtime* :   
We can determine how many times every certain node was executed across the node graph sequence through loopsize run processes and mark it as count,then we determine total_usedtime by adding all the time cost for one certain node.   

* *avg_time* :   
Clearly,now the average time cost for every single node within the graph node sequence through the all loop can be calculated by : avg_time = total_usedtime/count.  

* *time_percentage* :   
The percentage of one certain node's time cost in all nodes'.  
time_percentage = 100.0*total_usedtime/accum_time.  

