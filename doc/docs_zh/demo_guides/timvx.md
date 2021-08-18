# Tengine 使用 TIM-VX 进行部署

## 编译

参考 [源码编译（TIM-VX）](../source_compile/compile_timvx.md) 章节，编译生成或从第三方获取部署所需要的以下库文件：

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

- 在 Khadas VIM3 上运行时，需要使用上诉动态库替代板上 `/lib` 目录下的已有库文件；
- 需要使用 TIM-VX 提供的 A311D 预编译包中的 `galcore.ko` ( /prebuild-sdk-a311d/lib/galcore.ko)内核驱动文件进行更新。

## 运行

### 模型格式

TIM-VX 后端只支持加载 Uint8 tmfile，因此需要使用**模型量化工具**将 Float32 tmfile 量化成 Uint8 tmfile。

### 模型量化

Float32 量化成 Uint8 tmfile 具体实现步骤及相关工具获取请参考以下链接：

- [模型量化-非对称量化](../user_guides/quant_tool_uint8.md)
- [Uint8 量化工具下载地址](https://github.com/OAID/Tengine/releases/download/lite-v1.3/quant_tool_uint8)

### 推理精度设置

TIM-VX 只支持 **Uint8** 精度模型进行网络模型推理，需要在执行 `prerun_graph_multithread(graph_t graph, struct options opt)` 之前通过 `struct options opt` 显式设置推理精度。

Enable Uint8 mode

```bash
/* set runtime options */
struct options opt;
opt.num_thread = num_thread;
opt.cluster = TENGINE_CLUSTER_ALL;
opt.precision = TENGINE_MODE_UINT8;
opt.affinity = 0;
```

### 后端硬件绑定

在加载模型前，需要显式指定 **TIM-VX** 硬件后端 **context**，并在调用 `graph_t create_graph(context_t context, const char* model_format, const char* fname, ...)` 时传入该参数。

```
/* create VeriSilicon TIM-VX backend */
context_t timvx_context = create_context("timvx", 1);
add_context_device(timvx_context, "TIMVX");

/* create graph, load tengine model xxx.tmfile */
create_graph(timvx_context, "tengine", model_file);
```

## 参考 Demo

源码请参考 [tm_classification_timvx.c](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_classification_timvx.c)

### 执行结果

运行硬件为 Khadas VIM3，内置 5Tops 算力 AI 加速器。

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

## 支持硬件列表

| 芯片厂家  | 设备      |
| -------- | --------- |
| Amlogic | A311D、S905D3        |
| NXP     | iMX 8M Plus |
| JLQ     | JA310 |
| X86-64  | Simulator    |

## 支持算子列表
