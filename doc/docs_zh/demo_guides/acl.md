# Tengine 使用 ACL 进行部署

## 编译

参考 [源码编译（ACL）](../source_compile/compile_acl.md) 章节生成部署所需要的以下库文件：

```
3rdparty/acl/lib/
├── libarm_compute.so
├── libarm_compute_core.so
└── libarm_compute_graph.so

build-acl-arm64/install/lib/
└── libtengine-lite.so
```

## 运行

### 模型格式

ACL 支持直接加载 Float32 tmfile，如果工作在 Float16 推理精度模式下，Tengine 框架将在加载 Float32 tmfile 后自动在线转换为 Float16 数据进行推理。

### 推理精度设置

ACL 支持 **Float32** 和 **Float16** 两种精度模型进行网络模型推理，需要在执行 `prerun_graph_multithread(graph_t graph, struct options opt)` 之前通过 `struct options opt` 显式设置推理精度。

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

### 后端硬件绑定

在加载模型前，需要显式指定 **ACL** 硬件后端 **context**，并在调用 `graph_t create_graph(context_t context, const char* model_format, const char* fname, ...)` 时传入该参数。

```c++
/* create arm acl backend */
acl_context = create_context("acl", 1);
add_context_device(acl_context, "ACL");

/* create graph, load tengine model xxx.tmfile */
create_graph(acl_context, "tengine", model_file);
```

## 参考 Demo

源码请参考 [tm_classification_acl.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_acl.c)

### 执行结果

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
