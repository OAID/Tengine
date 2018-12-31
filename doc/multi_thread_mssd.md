## Tengine Multi-thread for CPU/GPU Heterogeneous scheduling Guide

This doc will show you how to run Tengine with multi-thread to run Mobilenet-SSD with Heterogeneous scheduling for both CPU and GPU on RK3399.

## Platform
Firefly-Rk3399 (Linux)
* GPU: Mali-T860
* CPU: dual-core Cortex-A72 + quad-core Cortex-A53

### Set CPU & GPU Frequency
In order to get the best performance, we set both cpu frequency and gpu frequency to be maximum.

```bash
sudo su
# set cpu A53
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
# set cpu A72
echo performance > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor 
# set gpu
echo "performance" >/sys/devices/platform/ff9a0000.gpu/devfreq/ff9a0000.gpu/governor
```
you can check the frequencies by
```bash
cat /sys/devices/system/cpu/cpufreq/policy4/cpuinfo_cur_freq   #1800000
cat /sys/devices/system/cpu/cpufreq/policy0/cpuinfo_cur_freq   #1416000
cat /sys/devices/platform/ff9a0000.gpu/devfreq/ff9a0000.gpu/cur_freq #800000000
```
### Cooling Processors
You can also use a fan to cool processors in order to stabilize processor temperatures.

## How it works
In this demo, we create three threads to run Mobilenet-SSD
* thread 0 for `CPU 2 A72`
* thread 1 for `GPU + 1A53` 
* thread 2 for `CPU 4 A53`
  
### Load One Model 
Despite we want to create three threads to run Mobilenet-SSD, we only load one model, and all thread use the same model to create their corresponding graphs.


### Create Three Graph & Set Devices
For each thread, we create graph independently. And we set corresponding devices. 
```cpp
graph_t graph=create_runtime_graph("cpu_a72",model_name,NULL);
graph_t graph=create_runtime_graph("gpu"    ,model_name,NULL);
graph_t graph=create_runtime_graph("cpu_a53",model_name,NULL);

set_graph_device(graph,"a72");
set_graph_device(graph,"acl_opencl");
set_graph_device(graph,"a53");
```
These are done in three functions:
```
cpu_thread_a72
gpu_thread
cpu_thread_a53
```

### Create Three Threads
```cpp
std::thread * t0=new std::thread(cpu_thread_a72,model_name,&avg_times[0]);
std::thread * t1=new std::thread(gpu_thread    ,model_name,&avg_times[1]);
std::thread * t2=new std::thread(cpu_thread_a53,model_name,&avg_times[2]);
```
You can see that three threads use the same model. We also collect each threads's avg_time to run the graph.

### Run & Results
After running, you can see the performance log:
```
=================================================
 Using 3 thread, MSSD performance 23.1135  FPS
=================================================
```
You can also verify the results:
* cpu_2A72_save.jpg
* gpu_save.jpg
* cpu_4A53_save.jpg


