# Tengine 使用 CUDA 进行部署

## 编译

参考 [源码编译（CUDA）](../source_compile/compile_cuda.md) 章节生成部署所需要的以下库文件：

待补充

## 运行

### 模型格式

CUDA 当前仅支持加载 Float32 tmfile。

### 推理精度设置

CUDA 支持 **Float32** 一种精度模型进行网络模型推理，需要在执行 `prerun_graph_multithread(graph_t graph, struct options opt)` 之前通过 `struct options opt` 显式设置推理精度。

Enable GPU FP32 mode

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_FP32;
opt.affinity = 0;
```

### 后端硬件绑定

在加载模型前，需要显式指定 **CUDA** 硬件后端 **context**，并在调用 `graph_t create_graph(context_t context, const char* model_format, const char* fname, ...)` 时传入该参数。

```c++
/* create NVIDIA CUDA backend */
context_t cuda_context = create_context("cuda", 1);
add_context_device(cuda_context, "CUDA");

/* create graph, load tengine model xxx.tmfile */
create_graph(cuda_context, "tengine", model_file);
```

## 参考 Demo

源码请参考 [tm_classification_tensorrt.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_tensorrt.c)

### 执行结果

```bash
nvidia@xaiver:~/tengine-lite-tq/build-linux-cuda$ ./tm_classification_cuda -m mobilenet_v1.tmfile -i cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679 -r 10
Tengine plugin allocator CUDA is registered.
tengine-lite library version: 1.2-dev

model file : /home/nvidia/tengine-test/models/mobilenet_v1.tmfile
image file : /home/nvidia/tengine-test/images/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 10 times, thread 1, avg time 4.58 ms, max_time 5.72 ms, min_time 4.24 ms
--------------------------------------
8.574145, 282
7.880118, 277
7.812578, 278
7.286452, 263
6.357486, 281
--------------------------------------
```
