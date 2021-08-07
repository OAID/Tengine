# Tengine uses TensorRT for deployment

## Compile

Refer to the [Source Compilation (TensorRT)](../source_compile/compile_tensort.md) chapter to generate the following library files required for deployment:

To be added

## Run
### Model format

TensorRT supports loading Float32 tmfile. If working in Float16 inference precision mode, Tengine framework will automatically convert to Float16 data online for inference after loading Float32 tmfile.

### Inference precision setting

TensorRT supports **Float32** 、 **Float16** 、 **Int8**  three precision model for network model inference. It is necessary to  set the inference precision explicitly through `struct options opt` before executing `prerun_graph_multithread(graph_t graph, struct options opt)`.

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

### Back-end hardware binding

Before loading the model, you need to specify the **TensorRT** hardware backend **context** explicitly, and pass it when calling `graph_t create_graph(context_t context, const char* model_format, const char* fname, ...)` Enter the parameter.

```
/* create NVIDIA TensorRT backend */
context_t trt_context = create_context("trt", 1);
add_context_device(trt_context, "TRT");

/* create graph, load tengine model xxx.tmfile */
create_graph(trt_context, "tengine", model_file);
```

## Demo for reference
 
Please refer to the source code [tm_classification_trt.cpp](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_trt.cpp)

### The Result of execution

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
