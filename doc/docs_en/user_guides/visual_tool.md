# Model Visualization Tool

### Introduction

Netron is a commonly used visualization tool for machine learning models.

### Purpose

Adapt Netron project to support parsing tmfile and visualizing Tengine model.

###  Tengine Model

Tengine model is a suffix ".tmfile" file, which is converted by Tengine-Covert-Tool through other training frameworks, and the stored data format is binary.

###  Principle Introduction

1. Netron is an Electron application developed based on Node.js, and its language is javascript；

2. Electron application is a cross-platform and application framework developed using javascript；

3. After parsing the model file, Netron reads

   a)   Model information，Model Properties；

   b)   Model input and output，Include size of input data；

   c)   Model drawing, the left side shows the model structure；

   d)   Node information，Node Properties，Attributes，Inputs, Outputs, etc.

 

## Model Properties

After entering the Netron interface, click the icon in the upper left corner or click the gray node (as shown by the red mark in Figure 1) to pop up the right sidebar: Model Properties.

 

| ![img](https://raw.githubusercontent.com/BUG1989/tengine-docs/main/images/clip_image002.jpg) |
| ------------------------------------------------------------ |
| Figure 1. Model Properties                                   |

 

（1）  MODEL PROPERTIES

a)  format：Tmfile v2.0；

b)  source:  Source model format, if converted into tmfile by Caffe, displays Caffe; TensorFlow is displayed if it is converted to tmfile through TensorFlow； 

（2）  INPUTS

a)  data:

name: The name of the graph input tensor. The name here is "data"; 

type: Data type, here is FP32 format; Dimension information is [10,3,227,227] in this model；

（3）  OUTPUTS

a)  prob:

name: The name of the graph output tensor. The name here is "prob"; 

type: Data type, here is FP32 format; Dimension information of tensors; Tensors' size is calculated by Tengine after passing through infershape.

 

##  Model Drawing

In Tengine, the models are connected by tensors.

Nodes are connected to form a network and different colors are displayed according to different operator types. For example, "layer" type nodes are displayed in blue, "Activation" related nodes are displayed in deep red and "Normalize" related nodes are displayed in dark green.

The Convolution operator displays weight and bias dimension information by default.

| ![img](https://raw.githubusercontent.com/BUG1989/tengine-docs/main/images/clip_image004.jpg) |
| ------------------------------------------------------------ |
| Figure 2. Model Drawing                                      |

 



## Node Information

Each node contains an Operator.

Operators have type type, name name, ATTRIBUTES, INPUTS and OUTPUTS.

| ![img](https://raw.githubusercontent.com/BUG1989/tengine-docs/main/images/clip_image006.jpg) |
| ------------------------------------------------------------ |
| Figure 3. Node Information                                   |

 

Click Node in the drawing area, and the detailed information of the node will pop up on the right side, including:

（1）  NODE PROPERTIES:

a)  type: Operator type, such as Convolution operator, displays Convolution;

b)  name: Node name, for example, the node name is conv-relu_conv1 (the Convolution node with the drawing area selected and marked in red);

（2） ATTRIBUTIES: operators with parameters will be displayed, but operators without parameters will not be displayed; According to different operator types, different ATTRIBUTES lists are displayed; For example, [5 Attributes of Different Operators] has a detailed list according to different operator types.

（3）  INPUTS: displays the inputs of this node, including:

a)   input：The name of the node input tensor.

b)   weight/bias/…：Input parameters of the node.

（4）  OUTPUTS:  displays the outputs of this node, including:

The name of the node output tensor；

Conv and relu merge here。





## Attributes of different operators

At present, there are 92 Node types (i.e., operator types, including the processing of INPUT and Const).

### Operator List

The nonparametric operators are listed in the following table：

| ID   | OPERATOR          | CATALOG    |
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

| ID   | OPERATOR             | CATALOG       |
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

(The column "CATALOG" in the table classifies the operators, which is related to their display colors, and "/"represents unknown classification. )

 

###  Attribute List of Parametric Operators

####  BatchNormalization

| Parameter      | Data Type | Description     |
| -------------- | --------- | --------------- |
| rescale_factor | float32   | default:   1    |
| eps            | float32   | default:   1e-5 |
| caffe_flavor   | int32     | default:   0    |

 

####  BilinearResize

| Parameter | Data Type | Description                                |
| --------- | --------- | ------------------------------------------ |
| scale_x   | float32   | Horizontal direction transformation factor |
| scale_y   | float32   | Vertical direction transformation factor   |
| type      | int32     | 0: NEAREST_NEIGHBOR    1: BILIEAR          |

 

####  Concat

| Parameter | Data Type | Description                                                  |
| --------- | --------- | ------------------------------------------------------------ |
| axis      | int32     | Merging operation axes supports "0, 1, 2, 3". The default value of NCHW is 1, and the default value of NHWC is 3. |

 

####  Convolution

| Parameter      | Data Type | Description                                                  |
| -------------- | --------- | ------------------------------------------------------------ |
| kernel_h       | int32     | Kernel size in vertical direction, the default value is 1    |
| kernel_w       | int32     | Horizontal Kernel size, the default value is 1               |
| stride_h       | int32     | Vertical Stride size, the default value is 1                 |
| stride_w       | int32     | Horizontal Stride size, the default value is 1               |
| dilation_h     | int32     | Vertical hole factor value, the default value is 1           |
| dilation_w     | int32     | Horizontal hole factor value, the default value is 1         |
| input_channel  | int32     | The channel number of input feature                          |
| output_channel | int32     | The channel number of output feature                         |
| group          | int32     | Group number. Default is 1                                   |
| activation     | int32     | Merge Relu or not. -1: Don't merge; 0: Relu; 1:Relu1; 6:Relu6. Default is -1 |
| pad_h0         | int32     | Top padding rows. Default is 0                               |
| pad_w0         | int32     | Left padding columns. Default is 0                           |
| pad_h1         | int32     | Bottom padding rows. Default is 0                            |
| pad_w1         | int32     | Right padding columns. Default is 0                          |

 

#### DeConvolution

| Parameter  | Data Type | Description                                                  |
| ---------- | --------- | ------------------------------------------------------------ |
| num_output | int32     | Number of output elements                                    |
| kernel_h   | int32     | Kernel size in vertical direction                            |
| kernel_w   | int32     | Kernel size in horizontal direction                          |
| stride_h   | int32     | Stride size in vertical direction                            |
| stride_w   | int32     | Stride size in horizontal direction                          |
| pad_w0     | int32     | Left padding columns                                         |
| pad_h0     | int32     | Top padding rows                                             |
| pad_w1     | int32     | Right padding columns                                        |
| pad_h1     | int32     | Bottom padding rows                                          |
| dilation_h | int32     | Dilation size in vertical direction                          |
| dilation_w | int32     | Dilation size in horizontal direction                        |
| group      | int32     | Group number. Default is 1                                   |
| activation | int32     | Merge Relu or not. -1: Don't merge; 0: Relu; 1:Relu1; 6:Relu6. Default is -1 |

 

#### DetectionOutput

| Parameter            | Data Type | Description                                                  |
| -------------------- | --------- | ------------------------------------------------------------ |
| num_classes          | int32     | Number of detection categories                               |
| keep_top_k           | int32     | After NMS operation, the number of bounding box nms_top_kint32 |
| nms_top_k            | int32     | Number of prediction frames with high confidence before NMS operation |
| confidence_threshold | float32   | Confidence threshold                                         |
| nms_threshold        | float32   | Non-maximum suppression threshold                            |

 

#### Eltwise

| Parameter    | Data Type | Description                                                  |
| ------------ | --------- | ------------------------------------------------------------ |
| type         | uint32    | 0: ELT_PROD  1: ELT_PROD_SCALAR  2: ELT_SUM  3: ELT_SUM_SCALAR  4: ELT_SUB  5: ELT_SUB_SCALAR  6: ELT_MAX  7: ELT_RSQRT  8: ELT_DIV  9: ELT_LOG  10: ELT_EXP  11: ELT_SQRT  12: ELT_FLOOR  13: ELT_SQUARE  14: ELT_POW  15:  ELT_POWER |
| caffe_flavor | int32     | Whether caffe Format 1 is supported: indicates the caffe framework calculation mode |

 

####  Flatten

| Parameter | Data Type | Description      |
| --------- | --------- | ---------------- |
| axis      | int32     | Starting axis    |
| end_axis  | int32     | Termination axis |

 

#### FullyConnected

| Parameter  | Data Type | Description             |
| ---------- | --------- | ----------------------- |
| num_output | int32     | Output feature map size |

 

#### LRN

| Parameter   | Data Type | Description          |
| ----------- | --------- | -------------------- |
| local_size  | int32     | Normalized area size |
| alpha       | float32   | *Default is 1e-05*   |
| beta        | float32   | Default is 0.75      |
| norm_region | int32     | Range of norm        |
| k           | float32   | Default is 2         |

 

#### Normalize

| 参数           | Data Type | Description                                      |
| -------------- | --------- | ------------------------------------------------ |
| across_spatial | int32     | Indicates whether to normalize the whole picture |
| channel_shared | int32     | Indicates whether scale is the same              |

 

#### Permute

| Parameter | Data Type | Description           |
| --------- | --------- | --------------------- |
| flag      | int32     | Reservation parameter |
| order0    | int32     | Axis before permute   |
| order1    | int32     | Axis before permute   |
| order2    | int32     | Axis before permute   |
| order3    | int32     | Axis before permute   |

 

#### Pooling

| Parameter    | Data Type | Description                         |
| ------------ | --------- | ----------------------------------- |
| alg          | int32     | 0 :MaxPooling   1:AvgPooling        |
| kernel_h     | int32     | Kernel size in vertical direction   |
| kernel_w     | int32     | Kernel size in horizontal direction |
| stride_h     | int32     | Stride size in vertical direction   |
| stride_w     | int32     | Stride size in horizontal direction |
| global       | int32     | 1：Global Pooling                   |
| caffe_flavor | int32     | 1：for caffe's special case         |
| pad_h0       | int32     | Top padding columns                 |
| pad_w0       | int32     | Left padding rows                   |
| pad_h1       | int32     | Bottom padding columns              |
| pad_w1       | int32     | Right padding rows                  |

 

#### PriorBox

| Parameter              | Data Type    | Description                                  |
| ---------------------- | ------------ | -------------------------------------------- |
| offset_vf_min_size     | tm_uoffset_t | Offset of TM2_Vector_floats  <min_sizes>     |
| offset_vf_max_size     | tm_uoffset_t | Offset of TM2_Vector_floats  <max_sizes>     |
| offset_vf_variance     | tm_uoffset_t | Offset of TM2_Vector_floats  <variances>     |
| offset_vf_aspect_ratio | tm_uoffset_t | Offset of TM2_Vector_floats  <aspect_ratios> |
| flip                   | int32        | Flip or not, the default value is 0          |
| clip                   | int32        | clip or not, the default value is 0          |
| img_size               | int32        | Candidate box size                           |
| img_h                  | int32        | Offset of candidate box in height            |
| img_w                  | int32        | Offset of candidate box in width             |
| step_h                 | float32      | Step size of candidate box on height         |
| step_w                 | float32      | Step size of candidate box on width          |
| offset                 | float32      | Center displacement of candidate frame       |
| num_priors             | int32        | Default number of candidate boxes            |
| out_dim                | int32        | Output number                                |

 

#### Region

| Parameter            | Data Type    | Description                           |
| -------------------- | ------------ | ------------------------------------- |
| num_classes          | int32        | Total number of detection categories  |
| side                 | int32        | NULL                                  |
| num_box              | int32        | Number of candidate boxes             |
| coords               | int32        | Number of coordinates                 |
| confidence_threshold | float32      | Confidence threshold                  |
| nms_threshold        | float32      | Non-maximum suppression threshold     |
| offset_vf_biases     | tm_uoffset_t | Offset of TM2_Vector_floats  <biases> |

 

#### ReLU

| Parameter      | Data Type | Description                                                  |
| -------------- | --------- | ------------------------------------------------------------ |
| negative_slope | float32   | Change the standard ReLU function, and the default value is 0 |

 

#### Reorg

| Parameter | Data Type | Description |
| --------- | --------- | ----------- |
| Stride    | int32     | Step size   |

 

#### Reshape

| Parameter | Data Type | Description                   |
| --------- | --------- | ----------------------------- |
| dim_0     | int32     | Batch                         |
| dim_1     | int32     | Channel                       |
| dim_2     | int32     | Height                        |
| dim_3     | int32     | Width                         |
| dim_size  | int32s    | Dim size                      |
| axis      | int32     | Specify the reshape dimension |

 

#### RoiPooling

| Parameter     | Data Type | Description                                          |
| ------------- | --------- | ---------------------------------------------------- |
| pooled_h      | int32     | Pool height                                          |
| pooled_w      | int32     | Pool width                                           |
| spatial_scale | float32   | Used to convert cords from input scale to pool scale |

 

#### RPN

| Parameter               | Data Type    | Description                                                  |
| ----------------------- | ------------ | ------------------------------------------------------------ |
| offset_vf_ratios        | tm_uoffset_t | pointer to TM2_Vector_floats  <ratios>                       |
| offset_vf_anchor_scales | tm_uoffset_t | pointer to  TM2_Vector_floats <anchor_scales>                |
| feat_stride             | int32        | Eigenvalue step size                                         |
| basesize                | int32        | Foundation size                                              |
| min_size                | int32        | minimum size                                                 |
| per_nms_topn            | int32        | The number of bounding box after NMS operation               |
| post_nms_topn           | int32        | Number of prediction frames with high confidence before NMS operation |
| nms_thresh              | float32      | Non-maximum suppression threshold                            |
| offset_va_anchors       | tm_uoffset_t | Offset of TM2_Vector_anchors  <anchors>                      |

 

#### Scale

| Parameter | Data Type | Description       |
| --------- | --------- | ----------------- |
| axis      | int32     | Operating shaft   |
| num_axes  | int32     | Scale of scaling  |
| bias_term | int32     | Offset of scaling |

 

#### Slice

| Parameter              | Data Type    | Description                                                  |
| ---------------------- | ------------ | ------------------------------------------------------------ |
| axis                   | int32        | Operating shaft                                              |
| offset_vi_slice_points | tm_uoffset_t | Offset of TM2_Vector_dims  <slice_points>  ;The starting dimension of each axis is equal to the number of axes |
| offset_vi_begins       | tm_uoffset_t | offset of TM2_Vector_dims  <begins>                          |
| offset_vi_sizes        | tm_uoffset_t | offset of TM2_Vector_dims  <sizes> The cut-off dimension of each axis is equal to the number of axes |
| iscaffe                | int32        | True: Slice corresponding to Caffe                           |
| ismxnet                | int32        | True: Slice corresponding to MxNet                           |
| begin                  | int32        | The starting index value of the slice on each axis           |
| end                    | int32        | End index value of slice on each axis                        |

 

#### SoftMax

| Parameter | Data Type | Description     |
| --------- | --------- | --------------- |
| axis      | int32     | Operating shaft |

 

#### DetectionPostProcess

| Parameter                 | Data Type    | Description                                                  |
| ------------------------- | ------------ | ------------------------------------------------------------ |
| max_detections            | int32        | Maximum number of detections                                 |
| max_classes_per_detection | int32        | Maximum number of classification categories in each detection frame |
| nms_score_threshold       | float32      | Non-maximum inhibition score threshold                       |
| nms_iou_threshold         | float32      | Non-maximum suppression IOU threshold                        |
| num_classes               | int32        | Total number of detection categories                         |
| offset_vf_scales          | tm_uoffset_t | Scale parameter                                              |

 

#### Gemm

| Parameter | Data Type | Description            |
| --------- | --------- | ---------------------- |
| alpha     | float32   | Matrix A               |
| beta      | float32   | Matrix B               |
| transA    | int32     | Is matrix A transposed |
| transB    | int32     | Is matrix B transposed |

 

#### Generic

| Parameter       | Data Type    | Description                    |
| --------------- | ------------ | ------------------------------ |
| max_input_num   | int32        | Maximum number of input Tensor |
| max_output_num  | int32        | Minimum number of input Tensor |
| offset_s_opname | tm_uoffset_t | Operator Name index            |

 

#### LSTM

| Parameter      | Data Type | Description             |
| -------------- | --------- | ----------------------- |
| forget_bias    | float32   | Reservation parameter   |
| clip           | float32   | Reservation parameter   |
| output_len     | int32     | Output length           |
| sequence_len   | int32     | Sequence length         |
| input_size     | int32     | Enter the size          |
| hidden_size    | int32     | Hide layer size         |
| cell_size      | int32     | Unit size               |
| has_peephole   | int32     | Is peephole supported   |
| has_projection | int32     | Is projection supported |
| has_clip       | int32     | Is clip supported       |
| has_bias       | int32     | Is bias supported       |
| has_init_state | int32     | Is init_state supported |
| forget_act     | int32     | Reservation parameter   |
| input_act      | int32     | Reservation parameter   |
| output_act     | int32     | Reservation parameter   |
| cellin_act     | int32     | Reservation parameter   |
| cellout_act    | int32     | Reservation parameter   |
| mxnet_flag     | int32     | Reservation parameter   |

 

#### RNN

| Parameter      | Data Type | Description               |
| -------------- | --------- | ------------------------- |
| clip           | float32   | Clip value                |
| output_len     | int32     | Output length             |
| sequence_len   | int32     | Sequence length           |
| input_size     | int32     | Input size                |
| hidden_size    | int32     | Hidden size               |
| has_clip       | int32     | Is clip supported         |
| has_bias       | int32     | Is bias supported         |
| has_init_state | int32     | Is init_state supported   |
| activation     | int32     | Activation layer category |

 

#### Squeeze

| Parameter | Data Type | Description |
| --------- | --------- | ----------- |
| dim_0     | int32     | Batch       |
| dim_1     | int32     | Channel     |
| dim_2     | int32     | Height      |
| dim_3     | int32     | Width       |

 

#### Pad

| Parameter | Data Type | Description                                       |
| --------- | --------- | ------------------------------------------------- |
| pad_n_0   | int32     | Reservation parameter. Default is 0               |
| pad_n_1   | int32     | Reservation parameter. Default is 0               |
| pad_c_0   | int32     | Reservation parameter. Default is 0               |
| pad_c_1   | int32     | Reservation parameter. Default is 0               |
| pad_h_0   | int32     | Top padding rows                                  |
| pad_h_1   | int32     | Bottom padding rows                               |
| pad_w_0   | int32     | Left padding columns                              |
| pad_w_1   | int32     | Right padding columns                             |
| mode      | int32     | 0: CONSTANT   1: REFLECT   2: SYMMETRIC   3. EDGE |
| value     | float32   | Set the CONSTANT value of when mode is constant   |

#### StridedSlice

| Parameter | Data Type | Description         |
| --------- | --------- | ------------------- |
| begine_n  | int32     | Batch start index   |
| end_n     | int32     | Batch end index     |
| stride_n  | int32     | Batch Slice step    |
| begine_c  | int32     | Channel start index |
| end_c     | int32     | Channel end index   |
| stride_c  | int32     | Channel Slice step  |
| begine_h  | int32     | Height start index  |
| end_h     | int32     | Height end index    |
| stride_h  | int32     | Height Slice step   |
| begine_w  | int32     | Width start index   |
| end_w     | int32     | Width end index     |
| stride_w  | int32     | Width Slice step    |

 

#### ArgMax

| Parameter | Data Type | Description                            |
| --------- | --------- | -------------------------------------- |
| axis      | int32     | Operation axis, the default value is 0 |

 

#### ArgMin

| Parameter | Data Type | Description                            |
| --------- | --------- | -------------------------------------- |
| axis      | int32     | Operation axis, the default value is 0 |

 

#### TopKV2

| Parameter | Data Type | Description                                    |
| --------- | --------- | ---------------------------------------------- |
| k         | int32     | The first k numbers                            |
| Sorted    | int32     | True: descending sort;  False: ascending order |

 

#### Reduction

| Parameter | Data Type | Description                        |
| --------- | --------- | ---------------------------------- |
| dim_0     | int32     | Batch                              |
| dim_1     | int32     | Channel                            |
| dim_2     | int32     | Height                             |
| dim_3     | int32     | Width                              |
| type      | int32     | Catalog                            |
| keepdim   | int32     | Specifies that dim does not change |

#### GRU

| Parameter          | Data Type | Description                |
| ------------------ | --------- | -------------------------- |
| clip               | float32   | Clip value                 |
| output_len         | int32     | Output length              |
| sequence_len       | int32     | Sequence length            |
| input_size         | int32     | Input length               |
| hidden_size        | int32     | Hidden length              |
| has_clip           | int32     | Is clip supported          |
| has_gate_bias      | int32     | Is bias supported          |
| has_candidate_bias | int32     | Is andidate_bias supported |
| has_init_state     | int32     | Is init_state supported    |
| mxnet_flag         | int32     | Reservation parameter      |

 

#### Addn

| Parameter | Data Type | Description                            |
| --------- | --------- | -------------------------------------- |
| axis      | int32     | Operation axis, the default value is 0 |

 

#### SwapAxis

| Parameter | Data Type | Description          |
| --------- | --------- | -------------------- |
| dim_0     | int32     | Axis 0 to be swapped |
| dim_1     | int32     | Axis 1 to be swapped |

 

#### Upsample

| Parameter | Data Type | Description    |
| --------- | --------- | -------------- |
| scale     | int32     | Scaling factor |

 

#### SpaceToBatchND

| Parameter  | Data Type | Description            |
| ---------- | --------- | ---------------------- |
| dilation_x | int32     | Width  expansion value |
| dilation_y | int32     | Height expansion value |
| pad_top    | int32     | Top padding rows       |
| pad_bottom | int32     | Bottom padding rows    |
| pad_left   | int32     | Left padding columns   |
| pad_right  | int32     | Right padding columns  |

 

#### BatchToSpaceND

| Parameter   | Data Type | Description            |
| ----------- | --------- | ---------------------- |
| dilation_x  | int32     | Width expansion value  |
| dilation_y  | int32     | Height expansion value |
| crop_top    | int32     | Top crop rows          |
| crop_bottom | int32     | Bottom crop rows       |
| crop_left   | int32     | Left crop columns      |
| crop_right  | int32     | Right crop columns     |

 

#### Resize

| Parameter | Data Type | Description                                |
| --------- | --------- | ------------------------------------------ |
| scale_x   | float32   | Horizontal direction transformation factor |
| scale_y   | float32   | Vertical direction transformation factor   |
| type      | int32     | 0: NEAREST_NEIGHBOR   1: BILIEAR           |

#### ShuffleChannel

| Parameter | Data Type | Description  |
| --------- | --------- | ------------ |
| group     | int32     | group number |

 

#### Crop

| Parameter   | Data Type | Description                                                  |
| ----------- | --------- | ------------------------------------------------------------ |
| num_args    | int32     | Number of parameters                                         |
| offset_c    | int32     | C dimension direction offset                                 |
| offset_h    | int32     | H dimension direction offset                                 |
| offset_w    | int32     | W dimension direction offset                                 |
| crop_h      | int32     | Output vertical size                                         |
| crop_w      | int32     | Output horizontal size                                       |
| center_crop | bool      | Center_crop or not; 0 : Not                                  |
| axis        | int32     | Operation axis, the default value is 1, which is used for Caffe framework |
| flag        | int32     | Reservation parameter                                        |

 

#### ROIAlign

| Parameter     | Data Type | Description                           |
| ------------- | --------- | ------------------------------------- |
| pooled_width  | int32     | Output width after pooling            |
| pooled_height | int32     | Output height after pooling           |
| spatial_scale | int32     | Scale factor of normal property space |

 

#### Psroipooling

| Parameter     | Data Type | Description                                   |
| ------------- | --------- | --------------------------------------------- |
| pooled_w      | int32     | Output width after pooling                    |
| pooled_h      | int32     | Output height after pooling                   |
| spatial_scale | float32   | Scale factor of multiplicative property space |
| output_dim    | int32     | Output dims size                              |

 

#### Unary

| Parameter | Data Type | Description                                                  |
| --------- | --------- | ------------------------------------------------------------ |
| type      | int32     | 0: UNARY_ABS  1: UNARY_NEG  2: UNARY_FLOOR  3: UNARY_CEIL  4: UNARY_SQUARE  5: UNARY_SQRT  6: UNARY_RSQRT  7: UNARY_EXP  8: UNARY_LOG  9: UNARY_SIN  10: UNARY_COS  11: UNARY_TAN  12: UNARY_ASIN  13: UNARY_ACOS  14: UNARY_ATAN  15: UNARY_RECIPROCAL  16: UNARY_TANH |

 

#### Expanddims

| Parameter | Data Type | Description     |
| --------- | --------- | --------------- |
| axis      | int32     | Operating shaft |

 

#### Bias

| Parameter | Data Type | Description               |
| --------- | --------- | ------------------------- |
| bias_size | int32     | Number of Bias parameters |

 

#### Threshold

| Parameter | Data Type | Description     |
| --------- | --------- | --------------- |
| Threshold | float32   | Threshold value |

 

#### Hardsigmoid

| Parameter | Data Type | Description      |
| --------- | --------- | ---------------- |
| alpha     | float32   | Alpha factor     |
| beta      | float32   | Offset parameter |

 

#### Embed

| Parameter        | Data Type | Description                                                  |
| ---------------- | --------- | ------------------------------------------------------------ |
| num_output       | int32     | Number of output elements                                    |
| input_dim        | int32     | input length                                                 |
| bias_term        | int32     | bias or not; 0 : Not                                         |
| weight_data_size | int32     | The Weight data length must be less than or equal to input_dim |

#### InstanceNorm

| Parameter | Data Type | Description |
| --------- | --------- | ----------- |
| eps       | float32   | Eps value   |

 

#### MVN

| Parameter          | Data Type | Description                            |
| ------------------ | --------- | -------------------------------------- |
| across_channels    | int32     | 1：Cross channel                       |
| normalize_variance | int32     | 0: summation method 1: variance method |
| eps                | float32   | normalize_variance = 1                 |

 

#### Cast

| Parameter | Data Type | Description                                   |
| --------- | --------- | --------------------------------------------- |
| type_from | int32     | 0:int32 1: float32 2: float16 3:int8 4: uint8 |
| type_to   | int32     | 0:int32 1: float32 2: float16 3:int8 4: uint8 |

 

#### HardSwish

| Parameter | Data Type | Description                             |
| --------- | --------- | --------------------------------------- |
| alpha     | float32   | The multiplication factor defaults to 1 |
| beta      | float32   | Shift parameter, the default is 3       |

 

#### Interp

| Parameter     | Data Type | Description           |
| ------------- | --------- | --------------------- |
| resize_type   | int32     | Reservation parameter |
| width_scale   | float32   | Width scaling factor  |
| height_scale  | float32   | Height scaling factor |
| output_width  | int32     | Output size of width  |
| output_height | int32     | Output size of height |

 

#### SELU

| Parameter | Data Type | Description                                               |
| --------- | --------- | --------------------------------------------------------- |
| alpha     | float32   | SeLU activates the value of α in the function             |
| lambda    | float32   | Represents the value of λ in the SeLU activation function |

 

#### ELU

| Parameter | Data Type | Description                    |
| --------- | --------- | ------------------------------ |
| alpha     | float32   | Alpha factor, the default is 1 |

#### Logical

| Parameter | Data Type | Description             |
| --------- | --------- | ----------------------- |
| type      | int32     | Logical processing type |

 

#### Gather

| Parameter   | Data Type | Description     |
| ----------- | --------- | --------------- |
| axis        | int32     | Operating shaft |
| indices_num | int32     | Number of Index |

 

#### Transpose

| Parameter | Data Type | Description           |
| --------- | --------- | --------------------- |
| dim0      | int32     | Axis before Transpose |
| dim1      | int32     | Axis before Transpose |
| dim2      | int32     | Axis before Transpose |
| dim3      | int32     | Axis before Transpose |

 

#### Comparison

| Parameter | Data Type | Description             |
| --------- | --------- | ----------------------- |
| type      | int32     | Compare operation types |

 

#### SpaceToDepth

| Parameter  | Data Type | Description                                                  |
| ---------- | --------- | ------------------------------------------------------------ |
| block_size | int32     | Horizontal direction & & vertical direction moves to the multiple of C direction |

 

#### DepthToSpace

| Parameter  | Data Type | Description                                                  |
| ---------- | --------- | ------------------------------------------------------------ |
| block_size | int32     | C direction moves to horizontal direction & & multiple of vertical direction |

 

#### SparseToDense

| Parameter          | Data Type | Description           |
| ------------------ | --------- | --------------------- |
| output_shape_size0 | int32     | Output size of height |
| output_shape_size1 | int32     | Output size of width  |
| default_value      | int32     | Default  Value        |

 

#### Clip

| Parameter | Data Type | Description                           |
| --------- | --------- | ------------------------------------- |
| max       | float     | Maximum value of truncation operation |
| min       | float     | Minimum value of truncation operation |

 

#### Unsqueeze

| Parameter        | Data Type    | Description                 |
| ---------------- | ------------ | --------------------------- |
| offset_vi_axises | tm_uoffset_t | Operation axis offset array |

 

#### ReduceL2

| Parameter | Data Type | Description             |
| --------- | --------- | ----------------------- |
| axis      | int32     | Operating shaft         |
| keepdim   | int32     | Retained dimension size |

 

#### Expand

| Parameter      | Data Type    | Description            |
| -------------- | ------------ | ---------------------- |
| offset_v_shape | tm_uoffset_t | Output dimension array |

#### Scatter                 

| Parameter | Data Type | Description             |
| --------- | --------- | ----------------------- |
| axis      | int32     | Operating shaft         |
| is_onnx   | tm_bool_t | Is onnx operator or not |

 

#### Tile

| Parameter      | Data Type    | Description                             |
| -------------- | ------------ | --------------------------------------- |
| offset_vi_flag | tm_uoffset_t | caffe: 0, onnx: 1                       |
| offset_vi_reps | tm_uoffset_t | Data used for tile completion operation |

 