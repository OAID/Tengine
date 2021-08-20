# Android project example

The Android project examples are used to show the hardware back-end running network model inference of Tengine based on the various CPU architectures of the Android system.

## Compile

Refer to the [Source Compile(Android)](../source_compile/compile_android.md) chapter to generate the following library files required for deployment:

```
build-android/install/lib/
└── libtengine-lite.so
```

## Run

### Model format

The CPU backend supports loading Float32/Float16/Uint8/Int8 tmfile, among which Float16/Uint8/Int8 needs to be obtained through the corresponding model quantization tool.

- [Int8 Quantization Tool User Manual](../user_guides/quant_tool_int8.md)
- [Uint8 Quantization Tool User Manual](../user_guides/quant_tool_uint8.md)
- [Int8 quantization tool download address](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_int8)
- [Uint8 quantization tool download address](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_uint8)

### Inference precision setting

CPU supports **Float32**/**Float16**/**Uint8**/**Int8** four precision model for network model inference. It is necessary to  set the inference precision explicitly through `struct options opt` before executing `prerun_graph_multithread(graph_t graph, struct options opt)`.

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

## Demo for reference

- Please refer to the source code [tm_classification.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification.c)
- Please refer to the source code [tm_classification_fp16.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_fp16.c)
- Please refer to the source code [tm_classification_uint8.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_uint8.c)
- Please refer to the source code [tm_classification_int8.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_int8.c)


### Using C API to make predictions

Most Android demos are developed based on C API. Calling C API is roughly divided into the following steps. For a more detailed API description, please refer to: [Tengine C API](../api_reference/c_api_doc.md).

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

### Using C++ API to make predictions

Android demo also provides C++ API to simplify the development process. Calling C++ API is roughly divided into the following steps. For more detailed API description, please refer to: [Tengine C++ API](../api_reference/cxx_api_doc.md).
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

### The result of execution

Use adb to connect to the Android device, taking the ubuntu environment as an example, the command is as follows:
```bash
sudo apt install adb #Install adb so that the computer can communicate with Android devices. And check the ip of the Android device.
adb connect [Android device ip]
adb devices #Ensure that the device can be seen

adb push tm_classification   /data/local/tmp/ 
adb push cat.jpg             /data/local/tmp/
adb push mobilenet.tmfile    /data/local/tmp/
adb push libtengine-lite.so  /data/local/tmp/

adb shell 

#Enter the terminal of the Android device at this time
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