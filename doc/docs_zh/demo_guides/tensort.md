# Tengine 使用 TensorRT 进行部署

## 编译

参考 [源码编译（TensorRT）](../source_compile/compile_tensort.md) 章节。
## 运行

### 模型格式

TensorRT 支持加载 Float32 tmfile，如果工作在 Float16 推理精度模式下，Tengine 框架将在加载 Float32 tmfile 后自动在线转换为 Float16 数据进行推理。

### 推理精度设置

TensorRT 支持 **Float32** 、 **Float16** 、 **Int8** 三种精度模型进行网络模型推理，需要在执行 `prerun_graph_multithread(graph_t graph, struct options opt)` 之前通过 `struct options opt` 显式设置推理精度。

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

Enable GPU Int8 mode

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_INT8;
opt.affinity = 0;
```

### 后端硬件绑定

在加载模型前，需要显式指定 **TensorRT** 硬件后端 **context**，并在调用 `graph_t create_graph(context_t context, const char* model_format, const char* fname, ...)` 时传入该参数。

```
/* create NVIDIA TensorRT backend */
context_t trt_context = create_context("trt", 1);
add_context_device(trt_context, "TRT");

/* create graph, load tengine model xxx.tmfile */
create_graph(trt_context, "tengine", model_file);
```

## 参考 Demo

源码请参考 [tm_classification_trt.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_trt.cpp)

### 执行结果

```
nvidia@xaiver:~/tengine-lite-tq/build-linux-trt$ ./tm_classification_trt -m mobilenet_v1.tmfile -i cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679 -r 10
Tengine plugin allocator TRT is registered.
tengine-lite library version: 1.2-dev
Tengine: Try using inference precision TF32 failed, rollback.

model file : /home/nvidia/tengine-test/models/mobilenet_v1.tmfile
image file : /home/nvidia/tengine-test/images/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 2.10 ms, max_time 3.10 ms, min_time 2.03 ms
--------------------------------------
8.574147, 282
7.880117, 277
7.812574, 278
7.286457, 263
6.357487, 281
--------------------------------------
```
