# Tengine uses TIM-VX for deployment

## Compile

Refer to the [Source Compilation (TIM-VX)](../source_compile/compile_timvx.md) chapter to compile and generate or obtain the following library files required for deployment from a third party:
```
3rdparty/tim-vx/lib/
├── libArchModelSw.so
├── libCLC.so
├── libGAL.so
├── libNNArchPerf.so
├── libOpenVX.so
├── libOpenVXU.so
└── libVSC.so

build-tim-vx-arm64/install/lib/
└── libtengine-lite.so
```

- When running on Khadas VIM3, you need to use the appeal dynamic library to replace the existing library files in the `/lib` directory on the board;
- It Need to use the `galcore.ko` (/prebuild-sdk-a311d/lib/galcore.ko) kernel driver file in the A311D pre-compiled package provided by TIM-VX to update.

## Run

### Model format

TIM-VX backend only supports loading Uint8 tmfile, so you need to use **model quantization tool** to quantize Float32 tmfile into Uint8 tmfile.

### Model quantification

Float32 is quantified into Uint8 tmfile. For specific implementation steps and related tools, please refer to the following link:

- [Model Quantization-Asymmetric Quantization](../user_guides/quant_tool_uint8.md)
- [Uint8 quantization tool download address](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_uint8)

### Reasoning accuracy setting

TIM-VX supports **Uint8** a precision model for network model inference. It is necessary to  set the inference precision explicitly through `struct options opt` before executing `prerun_graph_multithread(graph_t graph, struct options opt)`.

Enable Uint8 mode

```bash
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_UINT8;
opt.affinity = 0;
```

### Back-end hardware binding

Before loading the model, you need to specify the **TIM-VX** hardware backend **context** explicitly, and pass it when calling `graph_t create_graph(context_t context, const char* model_format, const char* fname, ...)` Enter the parameter.

```
/* create VeriSilicon TIM-VX backend */
context_t timvx_context = create_context("timvx", 1);
add_context_device(timvx_context, "TIMVX");

/* create graph, load tengine model xxx.tmfile */
create_graph(timvx_context, "tengine", model_file);
```

## Demo for reference

Please refer to the source code [tm_classification_timvx.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_timvx.c)

### The Results of execution

The running hardware is Khadas VIM3, with a built-in 5Tops computing power AI accelerator.
```
[khadas@Khadas tengine-lite]# ./tm_classification_timvx -m squeezenet_uint8.tmfile -i cat.jpg -r 1 -s 0.017,0.017,0.017 -r 10
Tengine plugin allocator TIMVX is registered.
Image height not specified, use default 227
Image width not specified, use default  227
Mean value not specified, use default   104.0, 116.7, 122.7
tengine-lite library version: 1.2-dev
TIM-VX prerun.

model file : squeezenet_uint8.tmfile
image file : cat.jpg
img_h, img_w, scale[3], mean[3] : 227 227 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 10 times, thread 1, avg time 2.95 ms, max_time 3.42 ms, min_time 2.76 ms
--------------------------------------
34.786182, 278
33.942883, 287
33.732056, 280
32.045452, 277
30.780502, 282
```

## The List of Supported hardware

| Vendor  | Device      |
| -------- | --------- |
| Amlogic | A311D, S905D3        |
| NXP     | iMX 8M Plus |
| JLQ     | JA310 |
| X86-64  | Simulator    |

## The List of supported operators
