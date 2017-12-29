# **TEngine  Performance Report**  

## **Revision Record**
|    Date    | Rev |Change Description|Author
| ---------- | --- |---|---|
| 2017-12-29 |  0.1 |Initial version|FeyaHan


---

## **Catalog**


#### [**Test Environment**](benchmark.md#test-environment-2)
#### [**Steps**](benchmark.md#steps)
#### [**Performance**](benchmark.md#performance-1)
#### [**How we get the time cost**](benchmark.md#how-we-get-the-time-cost-1)


---



## **Test Environment**
TEngine : v0.1  
Hardware SoC : Rockchip RK3399.  
CPU : Dual-core Cortex-A72 up to 2.0GHz (real frequency is 1.8GHz); Quad-core
Cortex-A53 up to 1.5GHz (real frequency is 1.4GHz).  
GPU : Mali T864 (800MHz).  
Operating System : Ubuntu 16.04.


---

## **Steps**

(For more information about the build of TEngine, please refer to the documentation of [install](install.md).)  

 
taskset -a 0x10 ./build/tests/bin/bench_sqz -r10  
taskset -a 0x10 ./build/tests/bin/bench_mobilenet -r10 


"taskset -a 0x10" :  We use this to bind our test program to single Arm A72 core, since RK3399 has 4 A53 (0-3) and 2 A72 (4-5).

r10 :  We run the test program for 10 times.  

bench_sqz/bench_mobilenet : Specify the neural network we were testing.


---

## **Performance**

The data in the tables below are in the unit of **us**.  
LoopSize : 10  
All items in the tables below are:  
**TPI** : The average total time for per inference within the whole loops.  
**SoftMax** : The average SoftMax time consumption for per inference within the whole loops.  
**Convolution** : The average Convolution time consumption for per inference within the whole loops.  
**Pooling** : The average Pooling time consumption for per inference within the whole loops.  
**Dropout** : The average Dropout time consumption for per inference within the whole loops.  
**Concat** : The average Concat time consumption for per inference within the whole loops.  

 <br />
 
#### **Mobilenet**
| Mobilenet  | TPI |Pooling|Fused.BNScaleRelu|Convolution|
| ---------- | --- |---|---|---|
| TimeElapse | 122856 |380 | 8569| 114249|
| Percentage | 100%|0.03%| 6.97%| 92.99%|

<br />

#### **Squeezenet**
| Squeezenet | TPI |SoftMax |Convolution |Pooling |Concat|
| ---------- | --- |---  |---         |---      | ---  |
| TimeElapse | 90677 |69 | 85371|2643|2973|
| Percentage | 100% |0.08% | 93.76%|2.90%|3.27%|  




---


## **How we get the time cost**  
#### **A simple principle**  
We make preparations including: prepare input data,load model,create runtime graph,set input/output node,setup input/output buffer and prerun graph before we test the benchmark demo.  
We mark the start time t0 and run the demo for loopsize(which was specified by user through terminal) times,finally,we mark the end time t1 and then we get the total time cost by totaltimecost = t1 - t0. Also,we get the average time cost per run by avgtimecost = totaltimecost/loopsize.  
#### **More detail**  
We can zoom in on the process of executing the specified neural network and get further information about the time cost through the whole loopsize run by this :   
$ export PROF_TIME=1  
$ taskset -a 0x10 ./build/tests/bin/bench_sqz -r40  
Then you can get some dump information like below on your terminal:  


<br />

---


======== time stats by operator: repeat 50  =====  
total time: 4848510 us with 778.85 Mfops  
0: SoftMax used 3825 us (0.08%)  
1: Convolution used 4540421 us (93.65%)  
2: Pooling used 132753 us (2.74%)  
3: Concat used 171506 us (3.54%)  
4: Dropout used 5 us (0.00%)  
    
  

0: total used time [906128:18.69%] us(count: 50 avg 18122 min 17207 max 19338)     
	&nbsp; &nbsp; &nbsp; &nbsp; Node: 90 conv10.fused  op: Convolution input: [1,512,14,14] output: [1,1000,14,14] Mfops: 200.70    
	1: total used time [344658:7.11%] us(count: 50 avg 6893 min 6585 max 8940)    
	&nbsp; &nbsp; &nbsp; &nbsp; Node: 22 fire3/expand3x3.fused  op: Convolution input: [1,16,56,56] output: [1,64,56,56] Mfops: 57.80   

	...  
	...  
	...  

39: total used time [5:0.00%] us(count: 50 avg 0 min 0 max 1)    
 	&nbsp; &nbsp; &nbsp; &nbsp; Node: 89 drop9  op: Dropout input: [1,512,14,14] output: [1,512,14,14] Mfops: 0.00  

total accumulated time: 4848510 us. roughly [96970] us per run  
Release Graph Executor for graph graph0  
ALL TEST DONE  
Release workspace default resource  
 

========end=====  

---


<br />

We get the total time cost and the corresponding time spent on each operator within the process of executing the specified neural network   and also the percentage of the time cost for certain operator in the whole run.  

Besides,we jump into the process of executing the graph node sequence and dump node info such as node index,node name,operator name,input/output shape and Mfops for this node one by one.  

total_time :   
We determine this by adding all the total used time for every single time record node across the graph node sequence.  

time_cost for SoftMax/ReLu/Convolution/Pooling/Dropout/Concat :  
We determine this by adding individually all the record nodes which belong to the same operator across the graph sequence through loopsize run processes.  
We dumped more information by ProfTime's Dump method which was called after we Postran the graph,and we got this :  

accum_time :  
We determine this by adding all the time cost for every single node across the node graph sequence through loopsize run processes.  

total_usedtime :   
We can determine how many times every certain node was executed across the node graph sequence through loopsize run processes and mark it as count,then we determine total_usedtime by adding all the time cost for one certain node.   

avg_time :   
Clearly,now the average time cost for every single node within the graph node sequence through the all loop can be calculated by : avg_time = total_usedtime/count.  

time_percentage :   
The percentage of one certain node's time cost in all nodes'.  
time_percentage = 100.0*total_usedtime/accum_time.  

Also,we can parse every node and dump the input/output tensor shape and list the node's index and name.




