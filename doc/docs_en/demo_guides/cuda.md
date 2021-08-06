# Tengine uses CUDA for deployment

## Compile

Refer to the [Source Compile(CUDA)](../source_compile/compile_cuda.md) chapter to generate the following library files required for deployment:

To do

## Run

### Model format

CUDA currently only supports loading Float32 tmfile.

### Inference precision setting

CUDA supports **Float32**, a precision model for network model inference. It is necessary to  set the inference precision explicitly through `struct options opt` before executing `prerun_graph_multithread(graph_t graph, struct options opt)`.
Enable GPU FP32 mode

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_FP32;
opt.affinity = 0;
```

### Back-end hardware binding

Before loading the model, you need to explicitly specify the **CUDA** hardware backend **context**, and pass it when calling `graph_t create_graph(context_t context, const char* model_format, const char* fname, ...)` Enter the parameter.

```c++
/* create NVIDIA CUDA backend */
context_t cuda_context = create_context("cuda", 1);
add_context_device(cuda_context, "CUDA");

/* create graph, load tengine model xxx.tmfile */
create_graph(cuda_context, "tengine", model_file);
```

## Reference Demo

For source code, please refer to [tm_classification_tensorrt.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_tensorrt.c)

### The results of execution
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
