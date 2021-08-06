# Tengine uses ACL for deployment

## Compile

Refer to [Source Compile（ACL）](../source_compile/compile_acl.md)chapter to generate the following library files required for deployment:

```
3rdparty/acl/lib/
├── libarm_compute.so
├── libarm_compute_core.so
└── libarm_compute_graph.so

build-acl-arm64/install/lib/
└── libtengine-lite.so
```

## Run

### Model format

ACL supports direct loading of Float32 tmfile. If working in Float16 inference precision mode, Tengine framework will automatically convert to Float16 data for inference online after loading Float32 tmfile.

### Inference precision setting

CUDA supports **Float32** and **Float16** two precision model for network model inference. It is necessary to  set the inference precision explicitly through `struct options opt` before executing `prerun_graph_multithread(graph_t graph, struct options opt)`.

Enable GPU FP32 mode

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_FP32;
opt.affinity = 0;
```

Enable GPU FP16 mode

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_FP16;
opt.affinity = 0;
```

### Back-end hardware binding

Before loading the model, you need to explicitly specify the **ACL** hardware backend **context**, and pass it when calling `graph_t create_graph(context_t context, const char* model_format, const char* fname, ...)` Enter the parameter.

```c++
/* create arm acl backend */
acl_context = create_context("acl", 1);
add_context_device(acl_context, "ACL");

/* create graph, load tengine model xxx.tmfile */
create_graph(acl_context, "tengine", model_file);
```

## Demo for reference

Please refer to the source code [tm_classification_acl.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_acl.c)

### The result of execution

```bash
[root@localhost tengine-lite]# ./tm_mssd_acl -m mssd.tmfile -i ssd_dog.jpg -t 1 -r 10
start to run register cpu allocator
start to run register acl allocator
tengine-lite library version: 1.0-dev
run into gpu by acl
Repeat 10 times, thread 2, avg time 82.32 ms, max_time 135.70 ms, min_time 74.10 ms
--------------------------------------
detect result num: 3 
dog     :99.8%
BOX:( 138 , 209 ),( 324 , 541 )
car     :99.7%
BOX:( 467 , 72 ),( 687 , 171 )
bicycle :99.6%
BOX:( 106 , 141 ),( 574 , 415 )
======================================
[DETECTED IMAGE SAVED]:
======================================
```
