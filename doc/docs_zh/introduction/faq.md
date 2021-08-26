# FAQ 常见问题

## 模型支持情况？

### 支持模型

已支持：TensorFlow、TFLite、PyTorch ( ONNX )、MXNet、Caffe、Darknet、OneFlow、PaddlePaddle

计划中：TensorFlow2 ( MLIR )、PyTorch ( TorchScript )

### 支持算子

请参考 [Tengine Support Operators List ](https://github.com/OAID/Tengine/wiki/Tengine-Support-Operators-List)。

## 如何获取模型中间结果？

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

## 如何获取模型各个layer耗时？

### 使用方法

- 程序执行前，添加环境变量 `export TG_DEBUG_TIME=1`，启用性能 Profiler 功能；
- 删除环境变量 `unset TG_DEBUG_TIME`， 关闭性能 Profiler 功能。

### logo 信息

```
 0 [ 7.48% :  0.7 ms]   Convolution idx:  5 shape: {1   3 100 100} -> {1   8  50  50}     int8 K: 3x3 | S: 2x2 | P: 0 1 0 1         MFLOPS:  1.08 Rate:1519
 1 [ 6.66% :  0.6 ms]   Convolution idx:  8 shape: {1   8  50  50} -> {1   8  50  50}     int8 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(  8) MFLOPS:  0.36 Rate:569
 2 [ 9.54% :  0.9 ms]   Convolution idx: 11 shape: {1   8  50  50} -> {1  16  50  50}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  0.64 Rate:706
 3 [ 3.99% :  0.4 ms]   Convolution idx: 14 shape: {1  16  50  50} -> {1  16  25  25}     int8 K: 3x3 | S: 2x2 | P: 0 1 0 1 DW( 16) MFLOPS:  0.18 Rate:475
 4 [ 6.77% :  0.6 ms]   Convolution idx: 17 shape: {1  16  25  25} -> {1  32  25  25}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  0.64 Rate:995
 5 [ 6.90% :  0.7 ms]   Convolution idx: 20 shape: {1  32  25  25} -> {1  32  25  25}     int8 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW( 32) MFLOPS:  0.36 Rate:549
 6 [ 4.20% :  0.4 ms]   Convolution idx: 23 shape: {1  32  25  25} -> {1  32  25  25}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  1.28 Rate:3207
 7 [ 1.42% :  0.1 ms]   Convolution idx: 26 shape: {1  32  25  25} -> {1  32  13  13}     int8 K: 3x3 | S: 2x2 | P: 1 1 1 1 DW( 32) MFLOPS:  0.10 Rate:721
 8 [ 2.36% :  0.2 ms]   Convolution idx: 29 shape: {1  32  13  13} -> {1  64  13  13}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  0.69 Rate:3092
 9 [ 3.43% :  0.3 ms]   Convolution idx: 32 shape: {1  64  13  13} -> {1  64  13  13}     int8 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW( 64) MFLOPS:  0.19 Rate:597
10 [ 3.98% :  0.4 ms]   Convolution idx: 35 shape: {1  64  13  13} -> {1  64  13  13}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  1.38 Rate:3663
11 [ 0.80% :  0.1 ms]   Convolution idx: 38 shape: {1  64  13  13} -> {1  64   7   7}     int8 K: 3x3 | S: 2x2 | P: 1 1 1 1 DW( 64) MFLOPS:  0.06 Rate:741
12 [ 2.24% :  0.2 ms]   Convolution idx: 41 shape: {1  64   7   7} -> {1 128   7   7}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  0.80 Rate:3771
13 [ 1.59% :  0.2 ms]   Convolution idx: 44 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(128) MFLOPS:  0.11 Rate:747
14 [ 4.21% :  0.4 ms]   Convolution idx: 47 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  1.61 Rate:4015
15 [ 1.53% :  0.1 ms]   Convolution idx: 50 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(128) MFLOPS:  0.11 Rate:778
16 [ 4.41% :  0.4 ms]   Convolution idx: 53 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  1.61 Rate:3833
17 [ 1.66% :  0.2 ms]   Convolution idx: 56 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(128) MFLOPS:  0.11 Rate:715
18 [ 4.16% :  0.4 ms]   Convolution idx: 59 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  1.61 Rate:4065
19 [ 1.52% :  0.1 ms]   Convolution idx: 62 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(128) MFLOPS:  0.11 Rate:784
20 [ 4.46% :  0.4 ms]   Convolution idx: 65 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  1.61 Rate:3786
21 [ 1.59% :  0.2 ms]   Convolution idx: 68 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(128) MFLOPS:  0.11 Rate:748
22 [ 4.37% :  0.4 ms]   Convolution idx: 71 shape: {1 128   7   7} -> {1 128   7   7}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  1.61 Rate:3869
23 [ 0.54% :  0.1 ms]   Convolution idx: 74 shape: {1 128   7   7} -> {1 128   4   4}     int8 K: 3x3 | S: 2x2 | P: 1 1 1 1 DW(128) MFLOPS:  0.04 Rate:722
24 [ 2.88% :  0.3 ms]   Convolution idx: 77 shape: {1 128   4   4} -> {1 256   4   4}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  1.05 Rate:3825
25 [ 1.02% :  0.1 ms]   Convolution idx: 80 shape: {1 256   4   4} -> {1 256   4   4}     int8 K: 3x3 | S: 1x1 | P: 1 1 1 1 DW(256) MFLOPS:  0.07 Rate:761
26 [ 5.54% :  0.5 ms]   Convolution idx: 81 shape: {1 256   4   4} -> {1 256   4   4}     int8 K: 1x1 | S: 1x1 | P: 0 0 0 0         MFLOPS:  2.10 Rate:3986
27 [ 0.11% :  0.0 ms]       Pooling idx: 84 shape: {1 256   4   4} -> {1 256   1   1}     int8 K: 4x4 | S: 1x1 | P: 0 0 0 0         Avg
28 [ 0.27% :  0.0 ms] FullyConnected idx: 85 shape: {1 256   1   1} -> {1 131   1   1}    int8
29 [ 0.40% :  0.0 ms]       Softmax idx: 86 shape: {1 131   1   1} -> {1 131   1   1}     int8
```

## 如何屏蔽性能算子？

Naive Profiler，用于关闭 CPU 性能算子，后端计算只使用 Naive C 实现的 reference op，用于对比分析性能算子的计算结果。
 
### 使用方法

- 程序执行前，添加环境变量 `export TG_DEBUG_REF=1`，启用 Naive Profiler 功能；
- 删除环境变量 `unset TG_DEBUG_REF`， 关闭精度 Naive Profiler 功能。
