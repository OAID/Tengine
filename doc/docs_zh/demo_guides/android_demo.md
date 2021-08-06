# Android 工程示例

Android工程示例用于展示 Tengine 基于 Android 系统的各种 CPU 架构的硬件后端运行网络模型推理。

## 编译

参考 [源码编译（Android）](../source_compile/compile_android.md) 章节生成部署所需要的以下库文件：

```
build-android/install/lib/
└── libtengine-lite.so
```

## 运行

### 模型格式

CPU 后端支持加载 Float32/Float16/Uint8/Int8 tmfile，其中 Float16/Uint8/Int8 需要通过相应的模型量化工具获取。

- [Int8 量化工具使用手册](../user_guides/quant_tool_int8.md)
- [Uint8 量化工具使用手册](../user_guides/quant_tool_uint8.md)
- [Int8 量化工具下载地址](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_int8)
- [Uint8 量化工具下载地址](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_uint8)

### 推理精度设置

CPU 支持 **Float32**/**Float16**/**Uint8**/**Int8** 四种精度模型进行网络模型推理，需要在执行 `prerun_graph_multithread(graph_t graph, struct options opt)` 之前通过 `struct options opt` 显式设置推理精度。

Enable CPU FP32 mode

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_FP32;
opt.affinity = 0;
```

Enable CPU FP16 mode

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_FP16;
opt.affinity = 0;
```

Enable CPU Uint8 mode

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_UINT8;
opt.affinity = 0;
```

Enable CPU Int8 mode

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_INT8;
opt.affinity = 0;
```

## 参考 Demo

- 源码请参考 [tm_classification.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification.c)
- 源码请参考 [tm_classification_fp16.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_fp16.c)
- 源码请参考 [tm_classification_uint8.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_uint8.c)
- 源码请参考 [tm_classification_int8.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_int8.c)


### 使用 C API 预测

Android demo 大多数基于 C API 开发，调用 C API 大致分为以下几个步骤。更详细的 API 描述请参考：[Tengine C API](../api_reference/c_api_doc.md)。

```c++
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_FP32;
opt.affinity = affinity;

/* inital tengine */
init_tengine();

/* create graph, load tengine model xxx.tmfile */
graph_t graph = create_graph(NULL, "tengine", model_file);

/* set the shape, data buffer of input_tensor of the graph */
int img_size = img_h * img_w * 3;
int dims[] = {1, 3, img_h, img_w};    // nchw
float* input_data = ( float* )malloc(img_size * sizeof(float));

tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
set_tensor_shape(input_tensor, dims, 4);
set_tensor_buffer(input_tensor, input_data, img_size * 4);

/* prerun graph, set work options(num_thread, cluster, precision) */
prerun_graph_multithread(graph, opt);

/* prepare process input data, set the data mem to input tensor */
get_input_data(image_file, input_data, img_h, img_w, mean, scale);

/* run graph */
run_graph(graph, 1);

/* get the result of classification */
tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
float* output_data = ( float* )get_tensor_buffer(output_tensor);

/* release tengine */
free(input_data);
postrun_graph(graph);
destroy_graph(graph);
release_tengine();
```

### 使用 C++ API 预测

Android demo 同时提供 C++ API 简化开发流程，调用 C++ API 大致分为以下几个步骤。更详细的 API 描述请参考：[Tengine C++ API](../api_reference/cxx_api_doc.md)。

```c++
/* inital tengine */
init_tengine();

tengine::Net somenet;
tengine::Tensor input_tensor;
tengine::Tensor output_tensor;

/* set runtime options of Net */
somenet.opt.num_thread = num_thread;
somenet.opt.cluster = TENGINE_CLUSTER_ALL;
somenet.opt.precision = TENGINE_MODE_FP32;
somenet.opt.affinity = affinity;

/* load model */
somenet.load_model(nullptr, "tengine", model_file.c_str());

/* prepare input data */
input_tensor.create(1, 3, img_h, img_w);
get_input_data(image_file.c_str(), ( float* )input_tensor.data, img_h, img_w, mean, scale);

/* set input data */
somenet.input_tensor("data", input_tensor);

/* forward */
somenet.run();

/* get result */
somenet.extract_tensor("prob", output_tensor);

/* release tengine */
release_tengine();
```

### 执行结果

使用adb 连接上Android 设备，以ubuntu环境为例，命令如下：
```bash
sudo apt install adb  #安装adb，使电脑可以与Android设备通信。并查看Android设备的ip。
adb connect  [安卓设备ip]
adb devices #确保可以看到设备 

adb push tm_classification   /data/local/tmp/ 
adb push cat.jpg             /data/local/tmp/
adb push mobilenet.tmfile    /data/local/tmp/
adb push libtengine-lite.so  /data/local/tmp/

adb shell 

#此时进入了Android设备的终端
cd /data/local/tmp
./tm_classification -m mobilenet.tmfile -i cat.jpg -g 224,224 -s 0.017,0.017,0.017 -w 104.007,116.669,122.679
```
```bash
start to run register cpu allocator
tengine-lite library version: 1.0-dev

model file : ./temp/models/mobilenet.tmfile
image file : ./temp/images/cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 656.76 ms, max_time 656.76 ms, min_time 656.76 ms
--------------------------------------
8.574148, 282
7.880116, 277
7.812579, 278
7.286453, 263
6.357488, 281
--------------------------------------
```