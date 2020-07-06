# FAQ 常见问题

## 模型支持情况？

### 支持模型

已支持：TensorFlow、TFLite、PyTorch ( ONNX )、MXNet、Caffe、Darknet

计划中：TensorFlow2 ( MLIR )、PyTorch ( TorchScript )

### 支持算子

请参考 [Tengine Support Operators List ](https://github.com/OAID/Tengine/wiki/Tengine-Support-Operators-List)。

## 如何获取模型中间结果？

创建 cmake 工程时添加 `-DTENGINE_DEBUG_DATA=ON` 变量，将在每一层的 input、output tensor 数据保存到 `./output` 中，以 mobilenet_v1.tmfile 为例：

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

创建 cmake 工程时添加 `-DTENGINE_DEBUG_TIME=ON` 变量，将在网络模型运行过程中打印每一层的耗时情况，以 mobilenet_v1.tmfile 为例：

```bash
$ ./tm_classification -m models/squeezenet.tmfile -i images/cat.jpg
Convolution              13.29 ms  conv1-conv1/bn-conv1/scale-relu1
Convolution              16.78 ms  conv2_1/dw-conv2_1/dw/bn-conv2_1/dw/scale-relu2_1/dw
Convolution              32.15 ms  conv2_1/sep-conv2_1/sep/bn-conv2_1/sep/scale-relu2_1/sep
Convolution               7.85 ms  conv2_2/dw-conv2_2/dw/bn-conv2_2/dw/scale-relu2_2/dw
Convolution              26.75 ms  conv2_2/sep-conv2_2/sep/bn-conv2_2/sep/scale-relu2_2/sep
Convolution              15.02 ms  conv3_1/dw-conv3_1/dw/bn-conv3_1/dw/scale-relu3_1/dw
Convolution              59.70 ms  conv3_1/sep-conv3_1/sep/bn-conv3_1/sep/scale-relu3_1/sep
Convolution               3.48 ms  conv3_2/dw-conv3_2/dw/bn-conv3_2/dw/scale-relu3_2/dw
Convolution              28.39 ms  conv3_2/sep-conv3_2/sep/bn-conv3_2/sep/scale-relu3_2/sep
Convolution               8.57 ms  conv4_1/dw-conv4_1/dw/bn-conv4_1/dw/scale-relu4_1/dw
Convolution              61.16 ms  conv4_1/sep-conv4_1/sep/bn-conv4_1/sep/scale-relu4_1/sep
Convolution               2.21 ms  conv4_2/dw-conv4_2/dw/bn-conv4_2/dw/scale-relu4_2/dw
Convolution              31.55 ms  conv4_2/sep-conv4_2/sep/bn-conv4_2/sep/scale-relu4_2/sep
Convolution               4.19 ms  conv5_1/dw-conv5_1/dw/bn-conv5_1/dw/scale-relu5_1/dw
Convolution              63.83 ms  conv5_1/sep-conv5_1/sep/bn-conv5_1/sep/scale-relu5_1/sep
Convolution               3.96 ms  conv5_2/dw-conv5_2/dw/bn-conv5_2/dw/scale-relu5_2/dw
Convolution              65.00 ms  conv5_2/sep-conv5_2/sep/bn-conv5_2/sep/scale-relu5_2/sep
Convolution               4.95 ms  conv5_3/dw-conv5_3/dw/bn-conv5_3/dw/scale-relu5_3/dw
Convolution              65.26 ms  conv5_3/sep-conv5_3/sep/bn-conv5_3/sep/scale-relu5_3/sep
Convolution               4.02 ms  conv5_4/dw-conv5_4/dw/bn-conv5_4/dw/scale-relu5_4/dw
Convolution              64.07 ms  conv5_4/sep-conv5_4/sep/bn-conv5_4/sep/scale-relu5_4/sep
Convolution               4.21 ms  conv5_5/dw-conv5_5/dw/bn-conv5_5/dw/scale-relu5_5/dw
Convolution              69.94 ms  conv5_5/sep-conv5_5/sep/bn-conv5_5/sep/scale-relu5_5/sep
Convolution               1.89 ms  conv5_6/dw-conv5_6/dw/bn-conv5_6/dw/scale-relu5_6/dw
Convolution              31.51 ms  conv5_6/sep-conv5_6/sep/bn-conv5_6/sep/scale-relu5_6/sep
Convolution               3.37 ms  conv6/dw-conv6/dw/bn-conv6/dw/scale-relu6/dw
Convolution              63.23 ms  conv6/sep-conv6/sep/bn-conv6/sep/scale-relu6/sep
Pooling                   0.23 ms  pool6
Convolution               1.53 ms  fc7

model file : models/squeezenet.tmfile
image file : images/cat.jpg
label_file : (null)
img_h, img_w, scale[3], mean[3] : 227 227 , 1.000 1.000 1.000, 104.0 116.7 122.7
Repeat 1 times, thread 1, avg time 480.62 ms, max_time 480.62 ms, min_time 480.62 ms
--------------------------------------
0.273199, 281
0.267552, 282
0.181004, 278
0.081799, 285
0.072407, 151
--------------------------------------
```

## Todo......

