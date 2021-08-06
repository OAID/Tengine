# 调试方法

## 计算图 Profiler

计算图 Profiler，显示完成 infer shape 操作后的已序列化 `ir_graph` 信息，用于确认 `infer shape` 是否正确。

### 使用方法

- 程序执行前，添加环境变量 `export TG_DEBUG_GRAPH=1`，启用计算图 Profiler 功能；
- 删除环境变量 `unset TG_DEBUG_GRAPH`， 关闭计算图 Profiler 功能。

### logo 信息

```
$./tm_classification -m mobilenet.tmfile -i cat.jpg  -r 10
tengine-lite library version: 1.4-dev
graph node_num 86 tensor_num: 86  subgraph_num: 1
graph layout: NCHW model layout: NCHW model_format: tengine

node: 0 op: Const name: conv1-conv1/bn-conv1/scale.bias.bn.fused.fused
        output tensors: 1
            0: [id: 0] conv1-conv1/bn-conv1/scale.bias.bn.fused.fused type: fp32/const shape: [32] from node: 0 (consumer: 1)

node: 1 op: Const name: conv1/weight.fused.fused
        output tensors: 1
            0: [id: 1] conv1/weight.fused.fused type: fp32/const shape: [32,3,3,3] from node: 1 (consumer: 1)

node: 2 op: InputOp name: input
        output tensors: 1
            0: [id: 2] data type: fp32/input shape: [1,3,224,224] from node: 2 (consumer: 1)

node: 3 op: Const name: conv2_1/dw-conv2_1/dw/bn-conv2_1/dw/scale.bias.bn.fused.fused
        output tensors: 1
            0: [id: 3] conv2_1/dw-conv2_1/dw/bn-conv2_1/dw/scale.bias.bn.fused.fused type: fp32/const shape: [32] from node: 3 (consumer: 1)

(#### 太多了，直接跳到末尾 ####)

node: 84 op: Pooling name: pool6
        input tensors: 1
            0: [id: 81] relu6/sep/0 type: fp32/var shape: [1,1024,7,7] from node: 81 (consumer: 1)
        output tensors: 1
            0: [id: 84] pool6 type: fp32/var shape: [1,1024,1,1] from node: 84 (consumer: 1)

node: 85 op: Convolution name: fc7
        input tensors: 3
            0: [id: 84] pool6 type: fp32/var shape: [1,1024,1,1] from node: 84 (consumer: 1)
            1: [id: 83] fc7/weight type: fp32/const shape: [1000,1024,1,1] from node: 83 (consumer: 1)
            2: [id: 82] fc7/bias type: fp32/const shape: [1000] from node: 82 (consumer: 1)
        output tensors: 1
            0: [id: 85] fc7 type: fp32/var shape: [1,1000,1,1] from node: 85

graph inputs: 1
        input
graph outputs: 1
        fc7

model file : mobilenet.tmfile
image file : cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 10 times, thread 1, avg time 44.12 ms, max_time 73.76 ms, min_time 37.14 ms
--------------------------------------
8.574144, 282
7.880117, 277
7.812573, 278
7.286458, 263
6.357486, 281
--------------------------------------
Tengine plugin device CPU is unregistered.
```

## 性能 Profiler

性能 Profiler，用于逐层耗时统计，在网络模型运行时统计 CPU 上 kernel 耗时信息，用于分析潜在的耗时问题。

### 使用方法

- 程序执行前，添加环境变量 `export TG_DEBUG_TIME=1`，启用性能 Profiler 功能；
- 删除环境变量 `unset TG_DEBUG_TIME`， 关闭性能 Profiler 功能。

### logo 信息

```
$./tm_classification -m mobilenet.tmfile -i cat.jpg  -r 10
tengine-lite library version: 1.4-dev
model file : mobilenet.tmfile
image file : cat.jpg
img_h, img_w, scale[3], mean[3] : 224 224 , 0.017 0.017 0.017, 104.0 116.7 122.7
Repeat 10 times, thread 1, avg time 42.36 ms, max_time 65.59 ms, min_time 38.26 ms
--------------------------------------
8.574144, 282
7.880117, 277
7.812573, 278
7.286458, 263
6.357486, 281
--------------------------------------
   0 [ 3.42% :    1.2 ms]   Convolution idx:    5 shape: {1   3 224 224} -> {1  32 112 112}       fp32 ->  fp32 K: 3x3 | S: 2x2 | P: 1 1 1 1         MFLOPS: 21.68 Rate:17722
   1 [ 4.60% :    1.6 ms]   Convolution idx:    8 shape: {1  32 112 112} -> {1  32 112 112}       fp32 ->  fp32 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW( 32) MFLOPS:  7.23 Rate:4392
   2 [ 5.66% :    2.0 ms]   Convolution idx:   11 shape: {1  32 112 112} -> {1  64 112 112}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS: 51.38 Rate:25423
   3 [ 3.28% :    1.2 ms]   Convolution idx:   14 shape: {1  64 112 112} -> {1  64  56  56}       fp32 ->  fp32 K: 3x3 | S: 2x2 | P: 1 1 1 1 DW( 64) MFLOPS:  3.61 Rate:3085
   4 [ 4.25% :    1.5 ms]   Convolution idx:   17 shape: {1  64  56  56} -> {1 128  56  56}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS: 51.38 Rate:33824
   5 [ 3.13% :    1.1 ms]   Convolution idx:   20 shape: {1 128  56  56} -> {1 128  56  56}       fp32 ->  fp32 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(128) MFLOPS:  7.23 Rate:6458
   6 [ 7.85% :    2.8 ms]   Convolution idx:   23 shape: {1 128  56  56} -> {1 128  56  56}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:102.76 Rate:36658
   7 [ 1.72% :    0.6 ms]   Convolution idx:   26 shape: {1 128  56  56} -> {1 128  28  28}       fp32 ->  fp32 K: 3x3 | S: 2x2 | P: 1 1 1 1 DW(128) MFLOPS:  1.81 Rate:2937
   8 [ 3.37% :    1.2 ms]   Convolution idx:   29 shape: {1 128  28  28} -> {1 256  28  28}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS: 51.38 Rate:42671
   9 [ 1.32% :    0.5 ms]   Convolution idx:   32 shape: {1 256  28  28} -> {1 256  28  28}       fp32 ->  fp32 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(256) MFLOPS:  3.61 Rate:7655
  10 [ 6.45% :    2.3 ms]   Convolution idx:   35 shape: {1 256  28  28} -> {1 256  28  28}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:102.76 Rate:44564
  11 [ 0.78% :    0.3 ms]   Convolution idx:   38 shape: {1 256  28  28} -> {1 256  14  14}       fp32 ->  fp32 K: 3x3 | S: 2x2 | P: 1 1 1 1 DW(256) MFLOPS:  0.90 Rate:3259
  12 [ 3.27% :    1.2 ms]   Convolution idx:   41 shape: {1 256  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS: 51.38 Rate:43954
  13 [ 1.01% :    0.4 ms]   Convolution idx:   44 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(512) MFLOPS:  1.81 Rate:4989
  14 [ 6.57% :    2.3 ms]   Convolution idx:   47 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:102.76 Rate:43767
  15 [ 0.96% :    0.3 ms]   Convolution idx:   50 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(512) MFLOPS:  1.81 Rate:5266
  16 [ 6.44% :    2.3 ms]   Convolution idx:   53 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:102.76 Rate:44659
  17 [ 1.01% :    0.4 ms]   Convolution idx:   56 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(512) MFLOPS:  1.81 Rate:5003
  18 [ 6.44% :    2.3 ms]   Convolution idx:   59 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:102.76 Rate:44678
  19 [ 0.98% :    0.3 ms]   Convolution idx:   62 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(512) MFLOPS:  1.81 Rate:5174
  20 [ 6.36% :    2.3 ms]   Convolution idx:   65 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:102.76 Rate:45249
  21 [ 1.02% :    0.4 ms]   Convolution idx:   68 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(512) MFLOPS:  1.81 Rate:4976
  22 [ 6.61% :    2.4 ms]   Convolution idx:   71 shape: {1 512  14  14} -> {1 512  14  14}       fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:102.76 Rate:43523
  23 [ 0.61% :    0.2 ms]   Convolution idx:   74 shape: {1 512  14  14} -> {1 512   7   7}       fp32 ->  fp32 K: 3x3 | S: 2x2 | P: 1 1 1 1 DW(512) MFLOPS:  0.45 Rate:2062
  24 [ 3.36% :    1.2 ms]   Convolution idx:   77 shape: {1 512   7   7} -> {1 1024   7   7}      fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS: 51.38 Rate:42853
  25 [ 0.74% :    0.3 ms]   Convolution idx:   80 shape: {1 1024   7   7} -> {1 1024   7   7}     fp32 ->  fp32 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(1024) MFLOPS:  0.90 Rate:3397
  26 [ 7.65% :    2.7 ms]   Convolution idx:   81 shape: {1 1024   7   7} -> {1 1024   7   7}     fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:102.76 Rate:37588
  27 [ 0.08% :    0.0 ms]       Pooling idx:   84 shape: {1 1024   7   7} -> {1 1024   1   1}     fp32 ->  fp32 K: 7x7 | S: 1x1 | P: 0 0 0 0         Avg
  28 [ 1.07% :    0.4 ms]   Convolution idx:   85 shape: {1 1024   1   1} -> {1 1000   1   1}     fp32 ->  fp32 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  2.05 Rate:5360
total time: 422.97 ms. avg time: 42.30 ms. min time: 35.73 ms.
```

## 精度 Profiler

精度 Profiler，用于 CPU 后端运行网络模型后，导出每一层的 input/ouput tensor data，用于分析输出结果异常的问题。

### 使用方法

- 程序执行前，添加环境变量 `export TG_DEBUG_DATA=1`，启用精度 Profiler 功能；
- 删除环境变量 `unset TG_DEBUG_DATA`， 关闭精度 Profiler 功能。

### logo 信息

数据导出后，在程序执行的当前路径下生成 `./output` 文件夹。

```bash
$ ./tm_classification -m models/squeezenet.tmfile -i images/cat.jpg
model file : models/squeezenet.tmfile
image file : images/cat.jpg
label_file : (null)
img_h, img_w, scale[3], mean[3] : 227 227 , 1.000 1.000 1.000, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 4402.85 ms, max_time 4402.85 ms, min_time 4402.85 ms
--------------------------------------
0.273199, 281
0.267552, 282
0.181004, 278
0.081799, 285
0.072407, 151
--------------------------------------
$ ls ./output
conv1-conv1-bn-conv1-scale-relu1_in_blob_data.txt
conv1-conv1-bn-conv1-scale-relu1_out_blob_data.txt
conv2_1-dw-conv2_1-dw-bn-conv2_1-dw-scale-relu2_1-dw_in_blob_data.txt
conv2_1-dw-conv2_1-dw-bn-conv2_1-dw-scale-relu2_1-dw_out_blob_data.txt
conv2_1-sep-conv2_1-sep-bn-conv2_1-sep-scale-relu2_1-sep_in_blob_data.txt
conv2_1-sep-conv2_1-sep-bn-conv2_1-sep-scale-relu2_1-sep_out_blob_data.txt
conv2_2-dw-conv2_2-dw-bn-conv2_2-dw-scale-relu2_2-dw_in_blob_data.txt
conv2_2-dw-conv2_2-dw-bn-conv2_2-dw-scale-relu2_2-dw_out_blob_data.txt
conv2_2-sep-conv2_2-sep-bn-conv2_2-sep-scale-relu2_2-sep_in_blob_data.txt
conv2_2-sep-conv2_2-sep-bn-conv2_2-sep-scale-relu2_2-sep_out_blob_data.txt
conv3_1-dw-conv3_1-dw-bn-conv3_1-dw-scale-relu3_1-dw_in_blob_data.txt
conv3_1-dw-conv3_1-dw-bn-conv3_1-dw-scale-relu3_1-dw_out_blob_data.txt
conv3_1-sep-conv3_1-sep-bn-conv3_1-sep-scale-relu3_1-sep_in_blob_data.txt
conv3_1-sep-conv3_1-sep-bn-conv3_1-sep-scale-relu3_1-sep_out_blob_data.txt
conv3_2-dw-conv3_2-dw-bn-conv3_2-dw-scale-relu3_2-dw_in_blob_data.txt
conv3_2-dw-conv3_2-dw-bn-conv3_2-dw-scale-relu3_2-dw_out_blob_data.txt
conv3_2-sep-conv3_2-sep-bn-conv3_2-sep-scale-relu3_2-sep_in_blob_data.txt
conv3_2-sep-conv3_2-sep-bn-conv3_2-sep-scale-relu3_2-sep_out_blob_data.txt
conv4_1-dw-conv4_1-dw-bn-conv4_1-dw-scale-relu4_1-dw_in_blob_data.txt
conv4_1-dw-conv4_1-dw-bn-conv4_1-dw-scale-relu4_1-dw_out_blob_data.txt
conv4_1-sep-conv4_1-sep-bn-conv4_1-sep-scale-relu4_1-sep_in_blob_data.txt
conv4_1-sep-conv4_1-sep-bn-conv4_1-sep-scale-relu4_1-sep_out_blob_data.txt
conv4_2-dw-conv4_2-dw-bn-conv4_2-dw-scale-relu4_2-dw_in_blob_data.txt
conv4_2-dw-conv4_2-dw-bn-conv4_2-dw-scale-relu4_2-dw_out_blob_data.txt
conv4_2-sep-conv4_2-sep-bn-conv4_2-sep-scale-relu4_2-sep_in_blob_data.txt
conv4_2-sep-conv4_2-sep-bn-conv4_2-sep-scale-relu4_2-sep_out_blob_data.txt
conv5_1-dw-conv5_1-dw-bn-conv5_1-dw-scale-relu5_1-dw_in_blob_data.txt
conv5_1-dw-conv5_1-dw-bn-conv5_1-dw-scale-relu5_1-dw_out_blob_data.txt
conv5_1-sep-conv5_1-sep-bn-conv5_1-sep-scale-relu5_1-sep_in_blob_data.txt
conv5_1-sep-conv5_1-sep-bn-conv5_1-sep-scale-relu5_1-sep_out_blob_data.txt
conv5_2-dw-conv5_2-dw-bn-conv5_2-dw-scale-relu5_2-dw_in_blob_data.txt
conv5_2-dw-conv5_2-dw-bn-conv5_2-dw-scale-relu5_2-dw_out_blob_data.txt
conv5_2-sep-conv5_2-sep-bn-conv5_2-sep-scale-relu5_2-sep_in_blob_data.txt
conv5_2-sep-conv5_2-sep-bn-conv5_2-sep-scale-relu5_2-sep_out_blob_data.txt
conv5_3-dw-conv5_3-dw-bn-conv5_3-dw-scale-relu5_3-dw_in_blob_data.txt
conv5_3-dw-conv5_3-dw-bn-conv5_3-dw-scale-relu5_3-dw_out_blob_data.txt
conv5_3-sep-conv5_3-sep-bn-conv5_3-sep-scale-relu5_3-sep_in_blob_data.txt
conv5_3-sep-conv5_3-sep-bn-conv5_3-sep-scale-relu5_3-sep_out_blob_data.txt
conv5_4-dw-conv5_4-dw-bn-conv5_4-dw-scale-relu5_4-dw_in_blob_data.txt
conv5_4-dw-conv5_4-dw-bn-conv5_4-dw-scale-relu5_4-dw_out_blob_data.txt
conv5_4-sep-conv5_4-sep-bn-conv5_4-sep-scale-relu5_4-sep_in_blob_data.txt
conv5_4-sep-conv5_4-sep-bn-conv5_4-sep-scale-relu5_4-sep_out_blob_data.txt
conv5_5-dw-conv5_5-dw-bn-conv5_5-dw-scale-relu5_5-dw_in_blob_data.txt
conv5_5-dw-conv5_5-dw-bn-conv5_5-dw-scale-relu5_5-dw_out_blob_data.txt
conv5_5-sep-conv5_5-sep-bn-conv5_5-sep-scale-relu5_5-sep_in_blob_data.txt
conv5_5-sep-conv5_5-sep-bn-conv5_5-sep-scale-relu5_5-sep_out_blob_data.txt
conv5_6-dw-conv5_6-dw-bn-conv5_6-dw-scale-relu5_6-dw_in_blob_data.txt
conv5_6-dw-conv5_6-dw-bn-conv5_6-dw-scale-relu5_6-dw_out_blob_data.txt
conv5_6-sep-conv5_6-sep-bn-conv5_6-sep-scale-relu5_6-sep_in_blob_data.txt
conv5_6-sep-conv5_6-sep-bn-conv5_6-sep-scale-relu5_6-sep_out_blob_data.txt
conv6-dw-conv6-dw-bn-conv6-dw-scale-relu6-dw_in_blob_data.txt
conv6-dw-conv6-dw-bn-conv6-dw-scale-relu6-dw_out_blob_data.txt
conv6-sep-conv6-sep-bn-conv6-sep-scale-relu6-sep_in_blob_data.txt
conv6-sep-conv6-sep-bn-conv6-sep-scale-relu6-sep_out_blob_data.txt
fc7_in_blob_data.txt
fc7_out_blob_data.txt
pool6_in_blob_data.txt
pool6_out_blob_data.txt
```

## Naive Profiler

Naive Profiler，用于关闭 CPU 性能算子，后端计算只使用 Naive C 实现的 reference op，用于对比分析性能算子的计算结果。
 
### 使用方法

- 程序执行前，添加环境变量 `export TG_DEBUG_REF=1`，启用 Naive Profiler 功能；
- 删除环境变量 `unset TG_DEBUG_REF`， 关闭精度 Naive Profiler 功能。
