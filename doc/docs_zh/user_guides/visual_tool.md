# 模型可视化工具

### 简介

Netron 是常用的机器学习模型可视化工具。

### 目的

适配 Netron 项目，使其支持解析 tmfile，可视化 Tengine 模型。

###  Tengine模型

Tengine 模型为后缀 ”.tmfile”文件，由 Tengine: Covert Tool 通过其他训练框架转换得到，存储数据格式为二进制。

###  原理介绍

1. Netron 是基于 Node.js 开发的 Electron 应用程序，使用的语言是 javascript；
2. Electron 应用程序是使用 javascript 开发的跨平台、应用程序框架；
3. Netron 在解析模型文件后，读取到
   a) 模型信息，Model Properties；
   b) 模型输入、输出，Model Inputs/Outputs，包含输入数据尺寸；
   c) 模型绘图，左侧显示模型结构；
   d) 节点信息，Node Properties，Attributes，Inputs, Outputs等；

## Model Properties

进入Netron界面后，点左上角图标或点击灰色节点（如 图1 中红色标记所示），弹出右侧边栏：Model Properties。

| ![img](https://raw.githubusercontent.com/OAID/Tengine/tengine-lite/doc/docs_zh/images/clip_image002.jpg) |
| ------------------------------------------------------------ |
| 图1  模型信息 Model Properties                               |

（1）  MODEL PROPERTIES

a)  format：解析到 Tengine 模型文件时显示 Tengine V2.0；
b)  source: 源模型格式，如通过 Caffe 转换成 tmfile，则显示 Caffe；如通过TensorFlow 转换成 tmfile，则显示 TensorFlow； 

（2）  INPUTS

a)  data:

name: 输入 tensor 的名称，如此处为 data; 
type: 数据类型，此处为 FP32 格式；维度信息，此模型为 [10,3,227,227]；

（3）  OUTPUTS

a)  prob:

name: 输出 tensor 的名称，如此处为 prob; 
type: 数据类型，此处为 FP32 格式；维度信息位置，须经过 infershape 后由 Tengine 计算得到输出尺寸。

##  模型绘图

Tengine 中，模型通过 tensor 连接。

节点 Node 连线形成网络，并根据不同算子类型显示不同颜色。如 ”layer” 类型节点显示为蓝色，”Activation” 相关节点显示为深红色，”Normalize” 相关节点显示为深绿色。
Convolution 算子默认显示 weight 和 bias 维度信息。

| ![img](https://raw.githubusercontent.com/OAID/Tengine/tengine-lite/doc/docs_zh/images/clip_image004.jpg) |
| ------------------------------------------------------------ |
| 图2  模型绘图                                                |

## 节点信息

节点为 Node，每个节点包含一个算子 Operator。

算子具有类型type、名称name、属性ATTRIBUTES及输入INPUTS、输出OUTPUTS。

| ![img](https://raw.githubusercontent.com/OAID/Tengine/tengine-lite/doc/docs_zh/images/clip_image006.jpg) |
| ------------------------------------------------------------ |
| 图2  模型绘图                                                |

点击绘图区的Node，右侧弹出该节点的详细信息，其中包括：

（1）  NODE PROPERTIES:

a)  type: 算子类型，如Convolution算子则显示Convolution;

b)  name: 节点名称，如节点名为conv-relu_conv1（绘图区被选中、红色标记的Convolution节点）;

（2）  ATTRIBUTIES: 有参数的算子会显示，无参数的算子不显示；根据算子类型的不同，显示不同的ATTRIBUTES列表；如【5 不同算子的Attributes】根据不同算子类型有详细列表。

（3）  INPUTS: 显示该节点的输入，其中：

a)   input：显示输入tensor名称，即前一个 Node 的输出；

b)   weight/bias/…：为输入的其他参数，如 weight，bias等。在Tengine中，weight、bias等作为 Node，以输出 tensor 的形式，传递数据给其对应的节点。

（4）  OUTPUTS: 输出tensor：

此处conv-relu_conv1节点的输出实际为Convolution后Relu的输出，其中 Relu 节点在模型转换时被融合进Convolution节点，此处不影响计算；

此输出 tensor 对应下一个 Node 的输入。

## 不同算子的Attributes

目前提供92个 Node 类型（即算子类型，但包括了对 INPUT 和 Const 的处理）的解析。

### 算子列表

其中无参数算子如下表：

| 编号 | 算子              | 分类       |
| ---- | ----------------- | ---------- |
| 0    | Accuracy          | /          |
| 4    | Const             | /          |
| 8    | DropOut           | Dropout    |
| 12   | INPUT             | INPUT      |
| 17   | Prelu             | Activation |
| 21   | ReLU6             | Activation |
| 29   | Split             | Shape      |
| 33   | Logistic          | Activation |
| 36   | TanH              | Activation |
| 37   | Sigmoid           | Activation |
| 39   | FusedbnScaleRelu  | Activation |
| 46   | Max               | Layer      |
| 47   | Min               | Layer      |
| 62   | Noop              | Layer      |
| 68   | Absval            | Data       |
| 74   | BroadMul          | Layer      |
| 81   | Reverse           | Shape      |
| 83   | Ceil              | Layer      |
| 84   | SquaredDifference | Layer      |
| 85   | Round             | Layer      |
| 86   | ZerosLike         | Layer      |
| 90   | Mean              | Layer      |
| 91   | MatMul            | Layer      |
| 94   | Shape             | Shape      |
| 95   | Where             | /          |
| 97   | Mish              | Activation |
| 98   | Num               | Layer      |

（表中“分类”一栏对算子进行了分类，与其显示颜色有关，“/”代表未知分类。）

有参数算子如下表：

| 编号 | 算子                 | 分类          |
| ---- | -------------------- | ------------- |
| 1    | BatchNormalization   | Normalization |
| 2    | BilinearResize       | Shape         |
| 3    | Concat               | Shape         |
| 5    | Convolution          | Layer         |
| 6    | DeConvolution        | Layer         |
| 7    | DetectionOutput      | Layer         |
| 9    | Eltwise              | /             |
| 10   | Flatten              | Shape         |
| 11   | FullyConnected       | Layer         |
| 13   | LRN                  | Normalization |
| 14   | Normalize            | Normalization |
| 15   | Permute              | Shape         |
| 16   | Pooling              | Pool          |
| 18   | PriorBox             | /             |
| 19   | Region               | /             |
| 20   | ReLU                 | Activation    |
| 22   | Reorg                | Shape         |
| 23   | Reshape              | Shape         |
| 24   | RoiPooling           | Pool          |
| 25   | RPN                  | /             |
| 26   | Scale                | Layer         |
| 27   | Slice                | Shape         |
| 28   | SoftMax              | Activation    |
| 30   | DetectionPostProcess | Layer         |
| 31   | Gemm                 | /             |
| 32   | Generic              | /             |
| 34   | LSTM                 | Layer         |
| 35   | RNN                  | Layer         |
| 38   | Squeeze              | Shape         |
| 40   | Pad                  | Layer         |
| 41   | StridedSlice         | Shape         |
| 42   | ArgMax               | Layer         |
| 43   | ArgMin               | Layer         |
| 44   | TopKV2               | Layer         |
| 45   | Reduction            | /             |
| 48   | GRU                  | Layer         |
| 49   | Addn                 | /             |
| 50   | SwapAxis             | Shape         |
| 51   | Upsample             | Data          |
| 52   | SpaceToBatchND       | Shape         |
| 53   | BatchToSpaceND       | Shape         |
| 54   | Resize               | Data          |
| 55   | ShuffleChannel       | Shape         |
| 56   | Crop                 | Shape         |
| 57   | ROIAlign             | /             |
| 58   | Psroipooling         | Pool          |
| 59   | Unary                | /             |
| 60   | Expanddims           | Shape         |
| 61   | Bias                 | Layer         |
| 63   | Threshold            | Activation    |
| 64   | Hardsigmoid          | Activation    |
| 65   | Embed                | Transform     |
| 66   | InstanceNorm         | Normalization |
| 67   | MVN                  | /             |
| 69   | Cast                 | /             |
| 70   | HardSwish            | Activation    |
| 71   | Interp               | Layer         |
| 72   | SELU                 | Activation    |
| 73   | ELU                  | Activation    |
| 75   | Logical              | Layer         |
| 76   | Gather               | Data          |
| 77   | Transpose            | Transform     |
| 78   | Comparison           | Layer         |
| 79   | SpaceToDepth         | Shape         |
| 80   | DepthToSpace         | Shape         |
| 82   | SparseToDense        | Shape         |
| 87   | Clip                 | Layer         |
| 88   | Unsqueeze            | Transform     |
| 89   | ReduceL2             | Layer         |
| 92   | Expand               | Layer         |
| 93   | Scatter              | Layer         |
| 96   | Tile                 | Layer         |

（表中“分类”一栏对算子进行了分类，与其显示颜色有关，“/”代表未知分类。）

###  有参数算子属性列表

####  BatchNormalization

| 参数           | 数据类型 | 说明        |
| -------------- | -------- | ----------- |
| rescale_factor | float32  | 默认值  1   |
| eps            | float32  | 默认值 1e-5 |
| caffe_flavor   | int32    | 默认值 0    |

####  BilinearResize

| 参数    | 数据类型 | 说明                              |
| ------- | -------- | --------------------------------- |
| scale_x | float32  | 水平方向变换因子                  |
| scale_y | float32  | 垂直方向变换因子                  |
| type    | int32    | 0: NEAREST_NEIGHBOR    1: BILIEAR |

####  Concat

| 参数 | 数据类型 | 说明                                                       |
| ---- | -------- | ---------------------------------------------------------- |
| axis | int32    | 合并操作轴，支持“0，1，2，3”，NCHW 默认为1， NHWC 默认为3. |

####  Convolution

| 参数           | 数据类型 | 说明                                                    |
| -------------- | -------- | ------------------------------------------------------- |
| kernel_h       | int32    | 垂直方向 Kernel 大小，默认值为1                         |
| kernel_w       | int32    | 水平方向 Kernel 大小，默认值为1                         |
| stride_h       | int32    | 垂直方向 Stride 大小，默认值为1                         |
| stride_w       | int32    | 水平方向 Stride 大小，默认值为1                         |
| dilation_h     | int32    | 垂直方向空洞因子值，默认值为1                           |
| dilation_w     | int32    | 水平方向空洞因子值,  默认值为1                          |
| input_channel  | int32    | 输入特征图通道数（creat_graph后）                       |
| output_channel | int32    | 输出特征图通道数                                        |
| group          | int32    | 分组数，默认值为  1                                     |
| activation     | int32    | 是否和Relu合并，0：RELU   1: RELU1 6: RELU6，默认值为-1 |
| pad_h0         | int32    | top padding rows，默认值为0                             |
| pad_w0         | int32    | left padding columns，默认值为0                         |
| pad_h1         | int32    | bottom padding rows，默认值为0                          |
| pad_w1         | int32    | right padding columns，默认值为0                        |

#### DeConvolution

| 参数       | 数据类型 | 说明                                         |
| ---------- | -------- | -------------------------------------------- |
| num_output | int32    | 输出元素个数                                 |
| kernel_h   | int32    | 垂直方向 Kernel 大小                         |
| kernel_w   | int32    | 水平方向 Kernel 大小                         |
| stride_h   | int32    | 垂直方向 Stride 大小                         |
| stride_w   | int32    | 水平方向 Stride 大小                         |
| pad_w0     | int32    | left padding columns                         |
| pad_h0     | int32    | top padding rows                             |
| pad_w1     | int32    | right padding columns                        |
| pad_h1     | int32    | bottom padding rows                          |
| dilation_h | int32    | 垂直方向空洞因子值                           |
| dilation_w | int32    | 水平方向空洞因子值                           |
| group      | int32    | 分组数，默认值为  1                          |
| activation | int32    | 是否和Relu合并，0：RELU   1: RELU1  6: RELU6 |

#### DetectionOutput

| 参数                 | 数据类型 | 说明                              |
| -------------------- | -------- | --------------------------------- |
| num_classes          | int32    | 检测类别数                        |
| keep_top_k           | int32    | NMS操作后， bounding box 个数     |
| nms_top_k            | int32    | NMS操作前，置信度高的预测框的个数 |
| confidence_threshold | float32  | 置信度阈值                        |
| nms_threshold        | float32  | 非极大值抑制阈值                  |

#### Eltwise

| 参数         | 数据类型 | 说明                                                         |
| ------------ | -------- | ------------------------------------------------------------ |
| type         | uint32   | 0: ELT_PROD  1: ELT_PROD_SCALAR  2: ELT_SUM  3: ELT_SUM_SCALAR  4: ELT_SUB  5: ELT_SUB_SCALAR  6: ELT_MAX  7: ELT_RSQRT  8: ELT_DIV  9: ELT_LOG  10: ELT_EXP  11: ELT_SQRT  12: ELT_FLOOR  13: ELT_SQUARE  14: ELT_POW  15:  ELT_POWER |
| caffe_flavor | int32    | 是否支持caffe 格式 1：表示caffe 框架计算模式                 |

####  Flatten

| 参数     | 数据类型 | 说明   |
| -------- | -------- | ------ |
| axis     | int32    | 起始轴 |
| end_axis | int32    | 终止轴 |

#### FullyConnected

| 参数       | 数据类型 | 说明           |
| ---------- | -------- | -------------- |
| num_output | int32    | 输出特征图大小 |

#### LRN

| 参数        | 数据类型 | 说明           |
| ----------- | -------- | -------------- |
| local_size  | int32    | 归一化区域大小 |
| alpha       | float32  | 默认为*1e-05*  |
| beta        | float32  | 默认为0.75     |
| norm_region | int32    | Norm 范围      |
| k           | float32  | 默认为2        |

#### Normalize

| 参数           | 数据类型 | 说明                         |
| -------------- | -------- | ---------------------------- |
| across_spatial | int32    | 表示是否对整个图片进行归一化 |
| channel_shared | int32    | 表示  scale 是否相同         |

#### Permute

| 参数   | 数据类型 | 说明             |
| ------ | -------- | ---------------- |
| flag   | int32    | 未使用           |
| order0 | int32    | permute 之前的轴 |
| order1 | int32    | permute 之前的轴 |
| order2 | int32    | permute 之前的轴 |
| order3 | int32    | permute 之前的轴 |

#### Pooling

| 参数         | 数据类型 | 说明                                                 |
| ------------ | -------- | ---------------------------------------------------- |
| alg          | int32    | 说明 pooling的计算方法，0 :MaxPooling   1:AvgPooling |
| kernel_h     | int32    | 垂直方向 Kernel 大小                                 |
| kernel_w     | int32    | 水平方向 Kernel 大小                                 |
| stride_h     | int32    | 垂直方向 Stride 大小                                 |
| stride_w     | int32    | 水平方向 Stride 大小                                 |
| global       | int32    | 1：Global Pooling 标志                               |
| caffe_flavor | int32    | 1：Caffe 框架特殊处理标志                            |
| pad_h0       | int32    | top padding columns                                  |
| pad_w0       | int32    | left padding rows                                    |
| pad_h1       | int32    | bottom padding columns                               |
| pad_w1       | int32    | right padding rows                                   |

#### PriorBox

| 参数                   | 数据类型     | 说明                                         |
| ---------------------- | ------------ | -------------------------------------------- |
| offset_vf_min_size     | tm_uoffset_t | offset of TM2_Vector_floats  <min_sizes>     |
| offset_vf_max_size     | tm_uoffset_t | offset of TM2_Vector_floats  <max_sizes>     |
| offset_vf_variance     | tm_uoffset_t | offset of TM2_Vector_floats  <variances>     |
| offset_vf_aspect_ratio | tm_uoffset_t | offset of TM2_Vector_floats  <aspect_ratios> |
| flip                   | int32        | 是否翻转，默认值为  0                        |
| clip                   | int32        | 是否裁剪，默认值为  0                        |
| img_size               | int32        | 候选框大小                                   |
| img_h                  | int32        | 候选框在 height 上的偏移                     |
| img_w                  | int32        | 候选框在 width 上的偏移                      |
| step_w                 | float32      | 候选框在 width 上的步长                      |
| step_h                 | float32      | 候选框在  height 上的步长                    |
| offset                 | float32      | 候选框中心位移                               |
| num_priors             | int32        | 默认候选框个数                               |
| out_dim                | int32        | 输出个数                                     |

#### Region

| 参数                 | 数据类型     | 说明                                  |
| -------------------- | ------------ | ------------------------------------- |
| num_classes          | int32        | 检测类别总数                          |
| side                 | int32        | NULL                                  |
| num_box              | int32        | 候选框数                              |
| coords               | int32        | 坐标个数                              |
| confidence_threshold | float32      | 置信度阈值                            |
| nms_threshold        | float32      | 非极大值抑制阈值                      |
| offset_vf_biases     | tm_uoffset_t | offset of TM2_Vector_floats  <biases> |

#### ReLU

| 参数           | 数据类型 | 说明                                |
| -------------- | -------- | ----------------------------------- |
| negative_slope | float32  | 对标准的ReLU函数进行变化，默认值为0 |

#### Reorg

| 参数   | 数据类型 | 说明     |
| ------ | -------- | -------- |
| Stride | int32    | 步进大小 |

#### Reshape

| 参数     | 数据类型 | 说明              |
| -------- | -------- | ----------------- |
| dim_0    | int32    | Batch             |
| dim_1    | int32    | Channel           |
| dim_2    | int32    | Height            |
| dim_3    | int32    | Width             |
| dim_size | int32    | Dim 大小          |
| axis     | int32    | 指定 reshape 维度 |

#### RoiPooling

| 参数          | 数据类型 | 说明                                           |
| ------------- | -------- | ---------------------------------------------- |
| pooled_h      | int32    | 池化高度                                       |
| pooled_w      | int32    | 池化宽度                                       |
| spatial_scale | float32  | 用于将  cords 从输入比例转换为池化时使用的比例 |

#### RPN

| 参数                    | 数据类型     | 说明                                          |
| ----------------------- | ------------ | --------------------------------------------- |
| offset_vf_ratios        | tm_uoffset_t | pointer to TM2_Vector_floats  <ratios>        |
| offset_vf_anchor_scales | tm_uoffset_t | pointer to  TM2_Vector_floats <anchor_scales> |
| feat_stride             | int32        | 特征值步进大小                                |
| basesize                | int32        | 基础尺寸                                      |
| min_size                | int32        | 最小尺寸                                      |
| per_nms_topn            | int32        | NMS操作后， bounding box 个数                 |
| post_nms_topn           | int32        | NMS操作前，置信度高的预测框的个数             |
| nms_thresh              | float32      | 非极大值抑制阈值                              |
| offset_va_anchors       | tm_uoffset_t | offset of TM2_Vector_anchors  <anchors>       |

#### Scale

| 参数      | 数据类型 | 说明       |
| --------- | -------- | ---------- |
| axis      | int32    | 操作轴     |
| num_axes  | int32    | 缩放的比例 |
| bias_term | int32    | 缩放的偏置 |

#### Slice

| 参数                   | 数据类型     | 说明                                                         |
| ---------------------- | ------------ | ------------------------------------------------------------ |
| axis                   | int32        | 操作轴                                                       |
| offset_vi_slice_points | tm_uoffset_t | offset of TM2_Vector_dims  <slice_points>  各个轴的起始维度，大小等于轴数 |
| offset_vi_begins       | tm_uoffset_t | offset of TM2_Vector_dims  <begins>                          |
| offset_vi_sizes        | tm_uoffset_t | offset of TM2_Vector_dims  <sizes>  各个轴的截止维度,  大小等于轴数 |
| iscaffe                | int32        | True: 表明是 caffe 框架中的  slice                           |
| ismxnet                | int32        | True: 表明是  mxnet 框架中的slice                            |
| begin                  | int32        | 各个轴上切片的起始索引值                                     |
| end                    | int32        | 各个轴上切片的结束索引值                                     |

#### SoftMax

| 参数 | 数据类型 | 说明   |
| ---- | -------- | ------ |
| axis | int32    | 操作轴 |

#### DetectionPostProcess

| 参数                      | 数据类型     | 说明                         |
| ------------------------- | ------------ | ---------------------------- |
| max_detections            | int32        | 最大检测数量                 |
| max_classes_per_detection | int32        | 每个检测框中的最大分类类别数 |
| nms_score_threshold       | float32      | 非极大值抑制得分阈值         |
| nms_iou_threshold         | float32      | 非极大值抑制IOU阈值          |
| num_classes               | int32        | 检测类别总数                 |
| offset_vf_scales          | tm_uoffset_t | Scale参数                    |

#### Gemm

| 参数   | 数据类型 | 说明              |
| ------ | -------- | ----------------- |
| alpha  | float32  | 生成矩阵A         |
| beta   | float32  | 生成矩阵B         |
| transA | int32    | 矩阵A是否转置变换 |
| transB | int32    | 矩阵B是否转置变换 |

#### Generic

| 参数            | 数据类型     | 说明                  |
| --------------- | ------------ | --------------------- |
| max_input_num   | int32        | 最大输入  Tensor 个数 |
| max_output_num  | int32        | 最小输入  Tensor 个数 |
| offset_s_opname | tm_uoffset_t | Operator Name 索引    |

#### LSTM

| 参数           | 数据类型 | 说明                |
| -------------- | -------- | ------------------- |
| forget_bias    | float32  | 未使用              |
| clip           | float32  | 未使用              |
| output_len     | int32    | 输出长度            |
| sequence_len   | int32    | 序列长度            |
| input_size     | int32    | 输入大小            |
| hidden_size    | int32    | 隐藏层大小          |
| cell_size      | int32    | 单元大小            |
| has_peephole   | int32    | 是否支持  peephole  |
| has_projection | int32    | 是否支持 projection |
| has_clip       | int32    | 是否支持 clip       |
| has_bias       | int32    | 是否支持 bias       |
| has_init_state | int32    | 是否支持 init_state |
| forget_act     | int32    | 未使用              |
| input_act      | int32    | 未使用              |
| output_act     | int32    | 未使用              |
| cellin_act     | int32    | 未使用              |
| cellout_act    | int32    | 未使用              |
| mxnet_flag     | int32    | 未使用              |

#### RNN

| 参数           | 数据类型 | 说明                |
| -------------- | -------- | ------------------- |
| clip           | float32  | 裁剪值              |
| output_len     | int32    | 输出长度            |
| sequence_len   | int32    | 序列长度            |
| input_size     | int32    | 输入大小            |
| hidden_size    | int32    | 隐藏层大小          |
| has_clip       | int32    | 是否支持  clip      |
| has_bias       | int32    | 是否支持 bias       |
| has_init_state | int32    | 是否支持 init state |
| activation     | int32    | 激活层类别          |

#### Squeeze

| 参数  | 数据类型 | 说明    |
| ----- | -------- | ------- |
| dim_0 | int32    | Batch   |
| dim_1 | int32    | Channel |
| dim_2 | int32    | Height  |
| dim_3 | int32    | Width   |

#### Pad

| 参数    | 数据类型 | 说明                                              |
| ------- | -------- | ------------------------------------------------- |
| pad_n_0 | int32    | 未使用，默认为0                                   |
| pad_n_1 | int32    | 未使用，默认为0                                   |
| pad_c_0 | int32    | 未使用，默认为0                                   |
| pad_c_1 | int32    | 未使用，默认为0                                   |
| pad_h_0 | int32    | top padding rows                                  |
| pad_h_1 | int32    | bottom padding rows                               |
| pad_w_0 | int32    | left padding columns                              |
| pad_w_1 | int32    | right padding columns                             |
| mode    | int32    | 0: CONSTANT   1: REFLECT   2: SYMMETRIC   3. EDGE |
| value   | float32  | 当  mode 为CONSTANT时，设置的常量值               |

#### StridedSlice

| 参数     | 数据类型 | 说明               |
| -------- | -------- | ------------------ |
| begine_n | int32    | Batch 起始索引     |
| end_n    | int32    | Batch 结束索引     |
| stride_n | int32    | Batch Slice 步进   |
| begine_c | int32    | Channel 起始索引   |
| end_c    | int32    | Channel 结束索引   |
| stride_c | int32    | Channel Slice 步进 |
| begine_h | int32    | Height 起始索引    |
| end_h    | int32    | Height 结束索引    |
| stride_h | int32    | Height Slice 步进  |
| begine_w | int32    | Width 起始索引     |
| end_w    | int32    | Width 结束索引     |
| stride_w | int32    | Width Slice 步进   |

#### ArgMax

| 参数 | 数据类型 | 说明             |
| ---- | -------- | ---------------- |
| axis | int32    | 操作轴,默认值为0 |

#### ArgMin

| 参数 | 数据类型 | 说明             |
| ---- | -------- | ---------------- |
| axis | int32    | 操作轴,默认值为0 |

#### TopKV2

| 参数   | 数据类型 | 说明                           |
| ------ | -------- | ------------------------------ |
| k      | int32    | top 的个数                     |
| Sorted | int32    | true: 降序排列 false: 升序排序 |

#### Reduction

| 参数    | 数据类型 | 说明           |
| ------- | -------- | -------------- |
| dim_0   | int32    | Batch          |
| dim_1   | int32    | Channel        |
| dim_2   | int32    | Height         |
| dim_3   | int32    | Width          |
| type    | int32    | 类别           |
| keepdim | int32    | 指定  dim 不变 |

#### GRU

| 参数               | 数据类型 | 说明                    |
| ------------------ | -------- | ----------------------- |
| clip               | float32  | Clip 值                 |
| output_len         | int32    | 输出长度                |
| sequence_len       | int32    | 序列长度                |
| input_size         | int32    | 输入大小                |
| hidden_size        | int32    | 隐藏层大小              |
| has_clip           | int32    | 是否支持 clip           |
| has_gate_bias      | int32    | 是否支持 gate_bias      |
| has_candidate_bias | int32    | 是否支持 candidate_bias |
| has_init_state     | int32    | 是否支持 init_state     |
| mxnet_flag         | int32    | 未使用                  |

#### Addn

| 参数 | 数据类型 | 说明             |
| ---- | -------- | ---------------- |
| axis | int32    | 操作轴,默认值为0 |

#### SwapAxis

| 参数  | 数据类型 | 说明        |
| ----- | -------- | ----------- |
| dim_0 | int32    | 待交换的轴0 |
| dim_1 | int32    | 待交换的轴1 |

#### Upsample

| 参数  | 数据类型 | 说明     |
| ----- | -------- | -------- |
| scale | int32    | 采样因子 |

#### SpaceToBatchND

| 参数       | 数据类型 | 说明                  |
| ---------- | -------- | --------------------- |
| dilation_x | int32    | Width 膨胀值          |
| dilation_y | int32    | Height 膨胀值         |
| pad_top    | int32    | top padding rows      |
| pad_bottom | int32    | bottom padding rows   |
| pad_left   | int32    | left padding columns  |
| pad_right  | int32    | right padding columns |

#### BatchToSpaceND

| 参数        | 数据类型 | 说明               |
| ----------- | -------- | ------------------ |
| dilation_x  | int32    | Width 膨胀值       |
| dilation_y  | int32    | Height 膨胀值      |
| crop_top    | int32    | top crop rows      |
| crop_bottom | int32    | bottom crop rows   |
| crop_left   | int32    | left crop columns  |
| crop_right  | int32    | right crop columns |

#### Resize

| 参数    | 数据类型 | 说明                             |
| ------- | -------- | -------------------------------- |
| scale_x | float32  | 水平方向变换因子                 |
| scale_y | float32  | 垂直方向变换因子                 |
| type    | int32    | 0: NEAREST_NEIGHBOR   1: BILIEAR |

#### ShuffleChannel

| 参数  | 数据类型 | 说明     |
| ----- | -------- | -------- |
| group | int32    | group 值 |

#### Crop

| 参数        | 数据类型 | 说明                                               |
| ----------- | -------- | -------------------------------------------------- |
| num_args    | int32    | 参数数目                                           |
| offset_c    | int32    | C 维度方向offset                                   |
| offset_h    | int32    | 垂直方向上方offset                                 |
| offset_w    | int32    | 垂直方向左方offset                                 |
| crop_h      | int32    | 输出垂直方向大小                                   |
| crop_w      | int32    | 输出水平方向大小                                   |
| center_crop | bool     | True: 中心crop False: 按照offset crop，默认为false |
| axis        | int32    | 操作轴，默认值为1，用于Caffe 框架                  |
| flag        | int32    | 未使用                                             |

#### ROIAlign

| 参数          | 数据类型 | 说明                 |
| ------------- | -------- | -------------------- |
| pooled_width  | int32    | 池化后的输出宽度     |
| pooled_height | int32    | 池化后的输出高度     |
| spatial_scale | int32    | 乘法性质空间标尺因子 |

#### Psroipooling

| 参数          | 数据类型 | 说明                 |
| ------------- | -------- | -------------------- |
| pooled_w      | int32    | 池化后的输出宽度     |
| pooled_h      | int32    | 池化后的输出高度     |
| spatial_scale | float32  | 乘法性质空间标尺因子 |
| output_dim    | int32    | 输出  dims 大小      |

#### Unary

| 参数 | 数据类型 | 说明                                                         |
| ---- | -------- | ------------------------------------------------------------ |
| type | int32    | 0: UNARY_ABS  1: UNARY_NEG  2: UNARY_FLOOR  3: UNARY_CEIL  4: UNARY_SQUARE  5: UNARY_SQRT  6: UNARY_RSQRT  7: UNARY_EXP  8: UNARY_LOG  9: UNARY_SIN  10: UNARY_COS  11: UNARY_TAN  12: UNARY_ASIN  13: UNARY_ACOS  14: UNARY_ATAN  15: UNARY_RECIPROCAL  16: UNARY_TANH |

#### Expanddims

| 参数 | 数据类型 | 说明   |
| ---- | -------- | ------ |
| axis | int32    | 操作轴 |

#### Bias

| 参数      | 数据类型 | 说明          |
| --------- | -------- | ------------- |
| bias_size | int32    | Bias 参数个数 |

#### Threshold

| 参数      | 数据类型 | 说明 |
| --------- | -------- | ---- |
| Threshold | float32  | 阈值 |

#### Hardsigmoid

| 参数  | 数据类型 | 说明       |
| ----- | -------- | ---------- |
| alpha | float32  | alpha 因子 |
| beta  | float32  | 偏移参数   |

#### Embed

| 参数             | 数据类型 | 说明                                  |
| ---------------- | -------- | ------------------------------------- |
| num_output       | int32    | 输出元素个数                          |
| input_dim        | int32    | 输入数据长度                          |
| bias_term        | int32    | 1 : 表示有bias                        |
| weight_data_size | int32    | Weight 数据长度 必须小于等于input_dim |

#### InstanceNorm

| 参数 | 数据类型 | 说明   |
| ---- | -------- | ------ |
| eps  | float32  | Eps 值 |

#### MVN

| 参数               | 数据类型 | 说明                              |
| ------------------ | -------- | --------------------------------- |
| across_channels    | int32    | 1：跨channel                      |
| normalize_variance | int32    | 0：求和方式    1：求方差方式      |
| eps                | float32  | normalize_variance = 1,用到的因子 |

#### Cast

| 参数      | 数据类型 | 说明                                           |
| --------- | -------- | ---------------------------------------------- |
| type_from | int32    | 0为int32 1: float32 2: float16 3:int8 4: uint8 |
| type_to   | int32    | 0为int32 1: float32 2: float16 3:int8 4: uint8 |

#### HardSwish

| 参数  | 数据类型 | 说明              |
| ----- | -------- | ----------------- |
| alpha | float32  | 乘法因子 默认为1  |
| beta  | float32  | 移位参数，默认为3 |

#### Interp

| 参数          | 数据类型 | 说明              |
| ------------- | -------- | ----------------- |
| resize_type   | int32    | 类型，未使用      |
| width_scale   | float32  | Width 缩放因子    |
| height_scale  | float32  | Height 缩放因子   |
| output_width  | int32    | 输出  Width 大小  |
| output_height | int32    | 输出  Height 大小 |

#### SELU

| 参数   | 数据类型 | 说明                        |
| ------ | -------- | --------------------------- |
| alpha  | float32  | SeLU 激活函数中的  α 的值   |
| lambda | float32  | 表示SeLU激活函数中的 λ 的值 |

#### ELU

| 参数  | 数据类型 | 说明                |
| ----- | -------- | ------------------- |
| alpha | float32  | alpha 因子，默认为1 |

#### Logical

| 参数 | 数据类型 | 说明         |
| ---- | -------- | ------------ |
| type | int32    | 逻辑处理类型 |

#### Gather

| 参数        | 数据类型 | 说明          |
| ----------- | -------- | ------------- |
| axis        | int32    | 操作轴        |
| indices_num | int32    | Index  的个数 |

#### Transpose

| 参数 | 数据类型 | 说明               |
| ---- | -------- | ------------------ |
| dim0 | int32    | Transpose 之前的轴 |
| dim1 | int32    | Transpose 之前的轴 |
| dim2 | int32    | Transpose 之前的轴 |
| dim3 | int32    | Transpose 之前的轴 |

#### Comparison

| 参数 | 数据类型 | 说明         |
| ---- | -------- | ------------ |
| type | int32    | 比较操作类型 |

#### SpaceToDepth

| 参数       | 数据类型 | 说明                                  |
| ---------- | -------- | ------------------------------------- |
| block_size | int32    | 水平方向&&垂直方向移动到 C 方向的倍数 |

#### DepthToSpace

| 参数       | 数据类型 | 说明                                 |
| ---------- | -------- | ------------------------------------ |
| block_size | int32    | C 方向移动到水平方向&&垂直方向的倍数 |

#### SparseToDense

| 参数               | 数据类型 | 说明              |
| ------------------ | -------- | ----------------- |
| output_shape_size0 | int32    | 输出  Height 大小 |
| output_shape_size1 | int32    | 输出  Width 大小  |
| default_value      | int32    | 默认  Value       |

#### Clip

| 参数 | 数据类型 | 说明           |
| ---- | -------- | -------------- |
| max  | float    | 截断操作最大值 |
| min  | float    | 截断操作最小值 |

#### Unsqueeze

| 参数             | 数据类型     | 说明             |
| ---------------- | ------------ | ---------------- |
| offset_vi_axises | tm_uoffset_t | 操作轴偏移量数组 |

#### ReduceL2

| 参数    | 数据类型 | 说明           |
| ------- | -------- | -------------- |
| axis    | int32    | 操作轴         |
| keepdim | int32    | 保留的维度大小 |

#### Expand

| 参数           | 数据类型     | 说明         |
| -------------- | ------------ | ------------ |
| offset_v_shape | tm_uoffset_t | 输出维度数组 |

#### Scatter                 

| 参数    | 数据类型  | 说明           |
| ------- | --------- | -------------- |
| axis    | int32     | 操作轴         |
| is_onnx | tm_bool_t | 是否为ONNX算子 |

#### Tile

| 参数           | 数据类型     | 说明                      |
| -------------- | ------------ | ------------------------- |
| offset_vi_flag | tm_uoffset_t | caffe: 0, onnx: 1         |
| offset_vi_reps | tm_uoffset_t | 用于  tile 补齐操作的数据 |
