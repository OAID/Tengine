/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: jingyou@openailab.com
 */

#ifndef __TM2_FORMAT_H__
#define __TM2_FORMAT_H__

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TM2_FILE_VER_MAIN       2
#define TM2_FILE_VER_SUB        0
#define TM2_FILE_VER_COMPILE    0

#define TM2_OP_VER              1

#define TM2_NOT_SET             0x00

/* Type define */
typedef uint32_t tm_uoffset_t;                  /* offset is 4-byte unsigned integer */
typedef uint32_t tm_size_t;                     /* size is 4-byte unsigned integer */
typedef uint8_t tm_bool_t;                      /* bool is 1-byte unsigned integer */

/* Operator strings */
#define TM2_OPSTR_ACCURACY                      "Accuracy"
#define TM2_OPSTR_BATCHNORMALIZATION            "BatchNormalization"
#define TM2_OPSTR_BILINEARRESIZE                "Resize"
#define TM2_OPSTR_CONCAT                        "Concat"
#define TM2_OPSTR_CONST                         "Const"
#define TM2_OPSTR_CONVOLUTION                   "Convolution"
#define TM2_OPSTR_DECONVOLUTION                 "Deconvolution"
#define TM2_OPSTR_DETECTIONOUTPUT               "DetectionOutput"
#define TM2_OPSTR_DROPOUT                       "Dropout"
#define TM2_OPSTR_ELTWISE                       "Eltwise"
#define TM2_OPSTR_FLATTEN                       "Flatten"
#define TM2_OPSTR_FULLYCONNECTED                "FullyConnected"
#define TM2_OPSTR_INPUTOP                       "InputOp"
#define TM2_OPSTR_LRN                           "LRN"
#define TM2_OPSTR_NORMALIZE                     "Normalize"
#define TM2_OPSTR_PERMUTE                       "Permute"
#define TM2_OPSTR_POOLING                       "Pooling"
#define TM2_OPSTR_PRELU                         "PReLU"
#define TM2_OPSTR_PRIORBOX                      "PriorBox"
#define TM2_OPSTR_REGION                        "Region"
#define TM2_OPSTR_RELU                          "ReLu"
#define TM2_OPSTR_RELU6                         "ReLu6"
#define TM2_OPSTR_REORG                         "Reorg"
#define TM2_OPSTR_RESHAPE                       "Reshape"
#define TM2_OPSTR_ROIPOOLING                    "ROIPooling"
#define TM2_OPSTR_RPN                           "RPN"
#define TM2_OPSTR_SCALE                         "Scale"
#define TM2_OPSTR_SLICE                         "Slice"
#define TM2_OPSTR_SOFTMAX                       "Softmax"
#define TM2_OPSTR_SPLIT                         "Split"
#define TM2_OPSTR_DETECTIONPOSTPROCESS          "DetectionPostProcess"
#define TM2_OPSTR_GEMM                          "Gemm"
#define TM2_OPSTR_GENERIC                       "Generic"
#define TM2_OPSTR_LOGISTIC                      "Logistic"
#define TM2_OPSTR_LSTM                          "LSTM"
#define TM2_OPSTR_RNN                           "RNN"
#define TM2_OPSTR_TANH                          "Tanh"
#define TM2_OPSTR_SIGMOID                       "Sigmoid"
#define TM2_OPSTR_SQUEEZE                       "Squeeze"
#define TM2_OPSTR_PAD                           "Pad"
#define TM2_OPSTR_STRIDEDSLICE                  "StridedSlice"
#define TM2_OPSTR_REDUCTION                     "Reduction"
#define TM2_OPSTR_ARGMAX                        "ArgMax"
#define TM2_OPSTR_ARGMIN                        "ArgMin"
#define TM2_OPSTR_TOPKV2                        "TopKV2"
#define TM2_OPSTR_MAX                           "Maximum"
#define TM2_OPSTR_MIN                           "Minimum"
#define TM2_OPSTR_ADDN                          "Addn"
#define TM2_OPSTR_SWAPAXIS                      "SwapAxis"
#define TM2_OPSTR_GRU                           "GRU"
#define TM2_OPSTR_FUSEDBNSCALERELU              "Fused.BNScaleReLu"
#define TM2_OPSTR_UPSAMPLE                      "Upsample"
#define TM2_OPSTR_SHUFFLECHANNEL                "ShuffleChannel"
#define TM2_OPSTR_RESIZE                        "Resize"
#define TM2_OPSTR_SPACETOBATCHND                "SpaceToBatchND"
#define TM2_OPSTR_BATCHTOSPACEND                "BatchToSpaceND"
#define TM2_OPSTR_CROP                          "Crop"
#define TM2_OPSTR_PSROIPOOLING                  "Psroipooling"
#define TM2_OPSTR_ROIALIGN                      "Roialign"
#define TM2_OPSTR_EXPANDDIMS                    "Expanddims"
#define TM2_OPSTR_UNARY                         "Unary"
#define TM2_OPSTR_BIAS                          "Bias"
#define TM2_OPSTR_NOOP                          "Noop"
#define TM2_OPSTR_THRESHOLD                     "Threshold"
#define TM2_OPSTR_HARDSIGMOID                   "Hardsigmoid"
#define TM2_OPSTR_EMBED                         "Embedding"
#define TM2_OPSTR_INSTANCENORM                  "InstanceNorm"
#define TM2_OPSTR_MVN                           "MVN"
#define TM2_OPSTR_ABSVAL                        "Absval"
#define TM2_OPSTR_CAST                          "Cast"
#define TM2_OPSTR_HARDSWISH                     "HardSwish"
#define TM2_OPSTR_INTERP                        "Interp"
#define TM2_OPSTR_SELU                          "Selu"
#define TM2_OPSTR_ELU                           "Elu"
#define TM2_OPSTR_BROADMUL                      "BroadMul"
#define TM2_OPSTR_LOGICAL                       "Logical"
#define TM2_OPSTR_GATHER                        "Gather"
#define TM2_OPSTR_TRANSPOSE                     "Transpose"
#define TM2_OPSTR_REVERSE                       "Reverse"
#define TM2_OPSTR_COMPARISON                    "Comparison"
#define TM2_OPSTR_SPACETODEPTH                  "SpaceToDepth"
#define TM2_OPSTR_DEPTHTOSPACE                  "DepthToSpace"
#define TM2_OPSTR_SQUAREDDIFFERENCE             "SquaredDifference"
#define TM2_OPSTR_SPARSETODENSE                 "SparseToDense"
#define TM2_OPSTR_CEIL                          "Ceil"
#define TM2_OPSTR_ROUND                         "Round"
#define TM2_OPSTR_ZEROSLIKE                     "ZerosLike"
#define TM2_OPSTR_CLIP                          "Clip"
#define TM2_OPSTR_UNSQUEEZE                     "Unsqueeze"
#define TM2_OPSTR_REDUCEL2                      "ReduceL2"
#define TM2_OPSTR_MEAN                          "Mean"
#define TM2_OPSTR_MATMUL                        "MatMul"
#define TM2_OPSTR_MISH                          "Mish"
#define TM2_OPSTR_L2NORMALIZATION               "L2Normalization"
#define TM2_OPSTR_RELU1                         "ReLU1"
#define TM2_OPSTR_SHAPE                         "Shape"
#define TM2_OPSTR_LOGSOFTMAX                    "LogSoftmax"
#define TM2_OPSTR_SCATTER                       "Scatter"
#define TM2_OPSTR_TILE                          "Tile"
#define TM2_OPSTR_L2POOL                        "L2Pool"
#define TM2_OPSTR_SOFTPLUS 						"Softplus"
#define TM2_OPSTR_RECIPROCAL 					"Reciprocal"
/* Operator types */
#define TM2_OPTYPE_ACCURACY                       0 /* No Param                 */
#define TM2_OPTYPE_BATCHNORMALIZATION             1 /* TM2_BatchNormParam       */
#define TM2_OPTYPE_BILINEARRESIZE                 2 /* TM2_ResizeParam          */
#define TM2_OPTYPE_CONCAT                         3 /* TM2_ConcatParam          */
#define TM2_OPTYPE_CONST                          4 /* No Param                 */
#define TM2_OPTYPE_CONVOLUTION                    5 /* TM2_ConvParam            */
#define TM2_OPTYPE_DECONVOLUTION                  6 /* TM2_DeconvParam          */
#define TM2_OPTYPE_DETECTIONOUTPUT                7 /* TM2_DetectionOutputParam */
#define TM2_OPTYPE_DROPOUT                        8 /* No Param                 */
#define TM2_OPTYPE_ELTWISE                        9 /* TM2_EltwiseParam         */
#define TM2_OPTYPE_FLATTEN                       10 /* TM2_FlattenParam         */
#define TM2_OPTYPE_FULLYCONNECTED                11 /* TM2_FCParam              */
#define TM2_OPTYPE_INPUTOP                       12 /* No Param                 */
#define TM2_OPTYPE_LRN                           13 /* TM2_LRNParam             */
#define TM2_OPTYPE_NORMALIZE                     14 /* TM2_NormalizeParam       */
#define TM2_OPTYPE_PERMUTE                       15 /* TM2_PermuteParam         */
#define TM2_OPTYPE_POOLING                       16 /* TM2_PoolParam            */
#define TM2_OPTYPE_PRELU                         17 /* No Param                 */
#define TM2_OPTYPE_PRIORBOX                      18 /* TM2_PriorBoxParam        */
#define TM2_OPTYPE_REGION                        19 /* TM2_RegionParam          */
#define TM2_OPTYPE_RELU                          20 /* TM2_ReLuParam            */
#define TM2_OPTYPE_RELU6                         21 /* No Param                 */
#define TM2_OPTYPE_REORG                         22 /* TM2_ReorgParam           */
#define TM2_OPTYPE_RESHAPE                       23 /* TM2_ReshapeParam         */
#define TM2_OPTYPE_ROIPOOLING                    24 /* TM2_ROIPoolingParam      */
#define TM2_OPTYPE_RPN                           25 /* TM2_RPNParam             */
#define TM2_OPTYPE_SCALE                         26 /* TM2_ScaleParam           */
#define TM2_OPTYPE_SLICE                         27 /* TM2_SliceParam           */
#define TM2_OPTYPE_SOFTMAX                       28 /* TM2_SoftmaxParam         */
#define TM2_OPTYPE_SPLIT                         29 /* No Param                 */
#define TM2_OPTYPE_DETECTIONPOSTPROCESS          30 /* TM2_DetectionPostProcessParam */
#define TM2_OPTYPE_GEMM                          31 /* TM2_GemmParam            */
#define TM2_OPTYPE_GENERIC                       32 /* TM2_GenericParam         */
#define TM2_OPTYPE_LOGISTIC                      33 /* No Param                 */
#define TM2_OPTYPE_LSTM                          34 /* TM2_LstmParam            */
#define TM2_OPTYPE_RNN                           35 /* TM2_RnnParam             */
#define TM2_OPTYPE_TANH                          36 /* No Param                 */
#define TM2_OPTYPE_SIGMOID                       37 /* No Param                 */
#define TM2_OPTYPE_SQUEEZE                       38 /* TM2_SqueezeParam         */
#define TM2_OPTYPE_FUSEDBNSCALERELU              39 /* No Param                 */
#define TM2_OPTYPE_PAD                           40 /* TM2_PadParam             */
#define TM2_OPTYPE_STRIDEDSLICE                  41 /* TM2_StrideSliceParam     */
#define TM2_OPTYPE_ARGMAX                        42 /* TM2_ArgmaxParam          */
#define TM2_OPTYPE_ARGMIN                        43 /* TM2_ArgminParam          */
#define TM2_OPTYPE_TOPKV2                        44 /* TM2_TopkV2Param          */
#define TM2_OPTYPE_REDUCTION                     45 /* TM2_ReductionParam       */
#define TM2_OPTYPE_MAX                           46 /* No Param                 */
#define TM2_OPTYPE_MIN                           47 /* No Param                 */
#define TM2_OPTYPE_GRU                           48 /* TM2_GruParam             */
#define TM2_OPTYPE_ADDN                          49 /* TM2_AddNParam            */
#define TM2_OPTYPE_SWAPAXIS                      50 /* TM2_SwapAixsParam        */
#define TM2_OPTYPE_UPSAMPLE                      51 /* TM2_UpsampleParam        */
#define TM2_OPTYPE_SPACETOBATCHND                52
#define TM2_OPTYPE_BATCHTOSPACEND                53
#define TM2_OPTYPE_RESIZE                        54
#define TM2_OPTYPE_SHUFFLECHANNEL                55 /* TM2_ShuffleChannelPara   */
#define TM2_OPTYPE_CROP                          56 /* TM2_CropParam            */
#define TM2_OPTYPE_ROIALIGN                      57
#define TM2_OPTYPE_PSROIPOOLING                  58
#define TM2_OPTYPE_UNARY                         59
#define TM2_OPTYPE_EXPANDDIMS                    60
#define TM2_OPTYPE_BIAS                          61
#define TM2_OPTYPE_NOOP                          62
#define TM2_OPTYPE_THRESHOLD                     63
#define TM2_OPTYPE_HARDSIGMOID                   64
#define TM2_OPTYPE_EMBED                         65
#define TM2_OPTYPE_INSTANCENORM                  66
#define TM2_OPTYPE_MVN                           67
#define TM2_OPTYPE_ABSVAL                        68
#define TM2_OPTYPE_CAST                          69
#define TM2_OPTYPE_HARDSWISH                     70
#define TM2_OPTYPE_INTERP                        71
#define TM2_OPTYPE_SELU                          72
#define TM2_OPTYPE_ELU                           73
#define TM2_OPTYPE_BROADMUL                      74
#define TM2_OPTYPE_LOGICAL                       75
#define TM2_OPTYPE_GATHER                        76
#define TM2_OPTYPE_TRANSPOSE                     77
#define TM2_OPTYPE_COMPARISON                    78
#define TM2_OPTYPE_SPACETODEPTH                  79
#define TM2_OPTYPE_DEPTHTOSPACE                  80
#define TM2_OPTYPE_REVERSE                       81
#define TM2_OPTYPE_SPARSETODENSE                 82
#define TM2_OPTYPE_CEIL                          83
#define TM2_OPTYPE_SQUAREDDIFFERENCE             84
#define TM2_OPTYPE_ROUND                         85
#define TM2_OPTYPE_ZEROSLIKE                     86
#define TM2_OPTYPE_CLIP                          87
#define TM2_OPTYPE_UNSQUEEZE                     88
#define TM2_OPTYPE_REDUCEL2                      89
#define TM2_OPTYPE_MEAN                          90
#define TM2_OPTYPE_MATMUL                        91
#define TM2_OPTYPE_EXPAND                        92
#define TM2_OPTYPE_SCATTER                       93
#define TM2_OPTYPE_SHAPE                         94
#define TM2_OPTYPE_WHERE                         95
#define TM2_OPTYPE_TILE                          96
#define TM2_OPTYPE_MISH                          97
#define TM2_OPTYPE_L2POOL                        98
#define TM2_OPTYPE_LOGSOFTMAX                    99
#define TM2_OPTYPE_RELU1                        100
#define TM2_OPTYPE_L2NORMALIZATION              101
#define TM2_OPTYPE_SOFTPLUS 					102
#define TM2_OPTYPE_RECIPROCAL 					103
#define TM2_OPTYPE_NUM 							104

/* --------------------- -------- TM objects -------------------------------- */

typedef struct
{
    uint16_t ver_main; /* main version of Tengine model file format */
    uint16_t ver_sub; /* sub version of Tengine model file format */
    uint16_t ver_compile; /* compile version of Tengine model file format */
    tm_uoffset_t offset_root; /* offset of root table (TM2_Model) */
} TM2_Header;

/* Root table of Tengine model */
typedef struct
{
    int32_t orig_format; /* format of original model */
    int32_t sub_format; /* sub format for DLA model */
    tm_uoffset_t offset_vo_subgraphs; /* offset of TM2_Vector_offsets <offsets of subgraphs> */
    tm_uoffset_t offset_s_mname; /* offset of string <model name> */
} TM2_Model;

/* Only 1 subgraph is supported currently */
typedef struct
{
    uint32_t subgraph_id; /* subgraph id */
    int32_t graph_layout; /* actual data layout */
    int32_t model_layout; /* data layout of original model */
    tm_uoffset_t offset_vi_input_indices; /* offset of TM2_Vector_indices <indices of input nodes> */
    tm_uoffset_t offset_vi_output_indices; /* offset of TM2_Vector_indices <indices of output nodes> */
    tm_uoffset_t offset_vo_seq_nodes; /* offset of TM2_Vector_offsets <nodes> */
    tm_uoffset_t offset_vo_tensors; /* offset of TM2_Vector_offsets <tensors> */
    tm_uoffset_t offset_vo_buffers; /* offset of TM2_Vector_offsets <buffers> */
    tm_uoffset_t offset_s_sname; /* offset of string <subgraph name> */
    tm_uoffset_t offset_vo_sub_info; /* offset of TM2_Vector_offsets <sub graph infomation> */
} TM2_Subgraph;

typedef struct
{
    uint32_t subgraph_id; /* sub graph idx */
    uint32_t input_wait_count; /* input wait count */
    int32_t data_type;         /* FP32 FP16 U8 INT8 */
    tm_uoffset_t offset_vi_node_list;     /* offset of TM2_Vector_indices <indices of node list> */
    tm_uoffset_t offset_vi_input_tensor;  /* offset of TM2_Vector_indices <indices of input node> */
    tm_uoffset_t offset_vi_output_tensor; /* offset of TM2_Vector_indices <indices of output node> */
    tm_uoffset_t offset_s_device_name;    /* offset of string <device name> */
} TM2_Sub_Info;

typedef struct
{
    tm_uoffset_t offset_s_attrname; /* offset of string <attr name> */
    tm_uoffset_t offset_s_attrval; /* offset of string <attr value> */
    int32_t attr_type;
} TM2_Attr;

typedef struct
{
    uint32_t node_id; /* node id */
    tm_uoffset_t offset_vi_input_tensors; /* offset of TM2_Vector_indices <indices of input tensors> */
    tm_uoffset_t offset_vi_output_tensors; /* offset of TM2_Vector_indices <indices of output tensors> */
    tm_uoffset_t offset_t_operator; /* offset of table  <operator> */
    tm_uoffset_t offset_s_nname; /* offset of string <node name> */
    tm_uoffset_t offset_vo_attrs; /* offset of TM2_Vector_offsets <attrs> */
    tm_bool_t dynamic_shape;
} TM2_Node;

typedef struct
{
    uint32_t op_ver; /* version of operator */
    uint32_t operator_type; /* operator type */
    tm_uoffset_t offset_t_param; /* offset of table <operator param> */
} TM2_Operator;

typedef struct
{
    int32_t zero_point;
    float scale;
    int32_t width;
} TM2_QuantParam;

typedef struct
{
    uint32_t tensor_id;
    uint32_t buffer_id;
    tm_uoffset_t offset_vd_dims; /* offset of TM2_Vector_dims <dims> */
    tm_uoffset_t offset_s_tname; /* offset of string <tensor name> */
    tm_uoffset_t offect_vo_quantparams; /* offset of TM2_Vector_offsets <quant params> */
    int32_t layout;
    int32_t type;
    int32_t data_type;
} TM2_Tensor;

typedef struct
{
    tm_size_t size; /* buffer size */
    tm_uoffset_t offset_data; /* offset of buffer data */
} TM2_Buffer;

typedef struct
{
    tm_size_t size; /* string size */
    tm_uoffset_t offset_data; /* offset of string data */
} TM2_String;

/* ------------------------ ------- Vectors --------------------------------- */

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    tm_uoffset_t offsets[0];
} TM2_Vector_offsets;

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    uint32_t indices[0];
} TM2_Vector_indices;

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    int32_t dims[0];
} TM2_Vector_dims;

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    float data[0];
} TM2_Vector_floats;

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    float data[0][4]; /* x0, y0, x1, y1 */
} TM2_Vector_anchors;

/* -------------------- ------- Operator params ----------------------------- */

typedef struct
{
    int32_t max_input_num;
    int32_t max_output_num;
    tm_uoffset_t offset_s_opname; /* offset of string <op name> */
} TM2_GenericParam;

typedef struct
{
    float rescale_factor;
    float eps;
    int32_t caffe_flavor;
} TM2_BatchNormParam;

typedef struct
{
    int32_t axis;
} TM2_ConcatParam;

typedef struct
{
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t input_channel;
    int32_t output_channel;
    int32_t group;
    int32_t activation;
    int32_t pad_h0; /* top padding rows */
    int32_t pad_w0; /* left padding columns */
    int32_t pad_h1; /* bottom padding rows */
    int32_t pad_w1; /* right padding columns */
} TM2_ConvParam;

typedef struct
{
    int32_t num_output;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_w0;
    int32_t pad_h0;
    int32_t pad_w1;
    int32_t pad_h1;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t group;
    int32_t activation;
    int32_t output_pad_h0;
    int32_t output_pad_w0;
} TM2_DeconvParam;

typedef struct
{
    int32_t num_classes;
    int32_t keep_top_k;
    int32_t nms_top_k;
    float confidence_threshold;
    float nms_threshold;
} TM2_DetectionOutputParam;

typedef struct
{
    uint32_t type;
    int32_t caffe_flavor;
    float shift;
    float power;
    float scale;
} TM2_EltwiseParam;

typedef struct
{
    int32_t num_output;
} TM2_FCParam;

typedef struct
{
    int32_t axis;
    int32_t end_axis;
} TM2_FlattenParam;

typedef struct
{
    int32_t local_size;
    float alpha;
    float beta;
    int32_t norm_region;
    float k;
    float bias;
    tm_bool_t is_onnx;
} TM2_LRNParam;

typedef struct
{
    int32_t across_spatial;
    int32_t channel_shared;
} TM2_NormalizeParam;

typedef struct
{
    int32_t flag;
    int32_t order0;
    int32_t order1;
    int32_t order2;
    int32_t order3;
} TM2_PermuteParam;

typedef struct
{
    uint32_t alg;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t global;
    int32_t caffe_flavor;
    int32_t pad_h0; /* top padding rows */
    int32_t pad_w0; /* left padding columns */
    int32_t pad_h1; /* bottom padding rows */
    int32_t pad_w1; /* right padding columns */
} TM2_PoolParam;

typedef struct
{
    tm_uoffset_t offset_vf_min_size; /* offset of TM2_Vector_floats <min_sizes> */
    tm_uoffset_t offset_vf_max_size; /* offset of TM2_Vector_floats <max_sizes> */
    tm_uoffset_t offset_vf_variance; /* offset of TM2_Vector_floats <variances> */
    tm_uoffset_t offset_vf_aspect_ratio; /* offset of TM2_Vector_floats <aspect_ratios> */
    int32_t flip;
    int32_t clip;
    int32_t img_size;
    int32_t img_h;
    int32_t img_w;
    float step_w;
    float step_h;
    float offset;
    int32_t num_priors;
    int32_t out_dim;
} TM2_PriorBoxParam;

typedef struct
{
    int32_t num_classes;
    int32_t side;
    int32_t num_box;
    int32_t coords;
    float confidence_threshold;
    float nms_threshold;
    tm_uoffset_t offset_vf_biases; /* offset of TM2_Vector_floats <biases> */
} TM2_RegionParam;

typedef struct
{
    float negative_slope;
} TM2_ReLuParam;

typedef struct
{
    int32_t stride;
} TM2_ReorgParam;

typedef struct
{
    int32_t is_mxnet;
    int32_t reverse;
    tm_uoffset_t offset_re_shape;
} TM2_ReshapeParam;

typedef struct
{
    float scale_x;
    float scale_y;
    int type;
} TM2_ResizeParam;

typedef struct
{
    int32_t pooled_h;
    int32_t pooled_w;
    float spatial_scale;
} TM2_ROIPoolingParam;

typedef struct
{
    tm_uoffset_t offset_vf_ratios; /* pointer to TM2_Vector_floats <ratios> */
    tm_uoffset_t offset_vf_anchor_scales; /* pointer to TM2_Vector_floats <anchor_scales> */
    int32_t feat_stride;
    int32_t basesize;
    int32_t min_size;
    int32_t per_nms_topn;
    int32_t post_nms_topn;
    float nms_thresh;
    tm_uoffset_t offset_va_anchors; /* offset of TM2_Vector_anchors <anchors> */
} TM2_RPNParam;

typedef struct
{
    int32_t axis;
    int32_t num_axes;
    int32_t bias_term;
} TM2_ScaleParam;

typedef struct
{
    int32_t axis;
    tm_uoffset_t offset_vi_slice_points; /* offset of TM2_Vector_dims <slice_points> */
    tm_uoffset_t offset_vi_begins; /* offset of TM2_Vector_dims <begins> */
    tm_uoffset_t offset_vi_sizes; /* offset of TM2_Vector_dims <sizes> */
    int32_t iscaffe;
    int32_t ismxnet;
    int32_t isonnx;
    int32_t begin;
    int32_t end;
} TM2_SliceParam;

typedef struct
{
    int32_t axis;
} TM2_SoftmaxParam;

typedef struct
{
    int32_t max_detections;
    int32_t max_classes_per_detection;
    float nms_score_threshold;
    float nms_iou_threshold;
    int32_t num_classes;
    tm_uoffset_t offset_vf_scales; /* y_scale, x_scale, h_scale, w_scale */
} TM2_DetectionPostProcessParam;

typedef struct
{
    float alpha;
    float beta;
    int32_t transA;
    int32_t transB;
} TM2_GemmParam;

typedef struct
{
    float forget_bias;
    float clip;
    int32_t output_len;
    int32_t sequence_len;
    int32_t input_size;
    int32_t hidden_size;
    int32_t cell_size;
    int32_t has_peephole;
    int32_t has_projection;
    int32_t has_clip;
    int32_t has_bias;
    int32_t has_init_state;
    int32_t forget_act;
    int32_t input_act;
    int32_t output_act;
    int32_t cellin_act;
    int32_t cellout_act;
    int32_t mxnet_flag;
} TM2_LstmParam;

typedef struct
{
    float clip;
    int32_t output_len;
    int32_t sequence_len;
    int32_t input_size;
    int32_t hidden_size;
    int32_t has_clip;
    int32_t has_bias;
    int32_t has_init_state;
    int32_t activation;
} TM2_RnnParam;

typedef struct
{
    int32_t dim_0;
    int32_t dim_1;
    int32_t dim_2;
    int32_t dim_3;
} TM2_SqueezeParam;

typedef struct
{
    int32_t axis;
    int32_t keepdims;
} TM2_ArgMaxParam;

typedef struct
{
    int32_t axis;
    int32_t keepdims;
} TM2_ArgMinParam;

typedef struct
{
    int32_t k;
    int32_t sorted;
} TM2_TopKV2Param;

typedef struct
{
    int32_t begin_n;
    int32_t end_n;
    int32_t stride_n;
    int32_t begin_c;
    int32_t end_c;
    int32_t stride_c;
    int32_t begin_h;
    int32_t end_h;
    int32_t stride_h;
    int32_t begin_w;
    int32_t end_w;
    int32_t stride_w;
} TM2_StridedSliceParam;

typedef struct
{
    int32_t pad_n_0;
    int32_t pad_n_1;
    int32_t pad_c_0;
    int32_t pad_c_1;
    int32_t pad_h_0;
    int32_t pad_h_1;
    int32_t pad_w_0;
    int32_t pad_w_1;
    int32_t mode;
    float value;
} TM2_PadParam;

typedef struct
{
    int32_t dim_0;
    int32_t dim_1;
    int32_t dim_2;
    int32_t dim_3;
    int32_t type;
    int32_t keepdim;
} TM2_ReductionParam;

typedef struct
{
    float clip;
    int32_t output_len;
    int32_t sequence_len;
    int32_t input_size;
    int32_t hidden_size;
    int32_t has_clip;
    int32_t has_gate_bias;
    int32_t has_candidate_bias;
    int32_t has_init_state;
    int32_t mxnet_flag;
} TM2_GRUParam;

typedef struct
{
    int32_t axis;
} TM2_AddnParam;

typedef struct
{
    int32_t dim_0;
    int32_t dim_1;
} TM2_SwapAxisParam;

typedef struct
{
    int32_t axis;
    int32_t split_dim;
    // int32_t squeeze_axis;
    tm_bool_t is_caffe;
    tm_bool_t is_onnx;
    tm_uoffset_t offset_split_sizes;
} TM2_SplitParam;

typedef struct
{
    float scale;
} TM2_UpsampleParam;

typedef struct
{
    int group;
} TM2_ShuffleChannelParam;

typedef struct
{
    int32_t dilation_x;
    int32_t dilation_y;
    int32_t pad_top;
    int32_t pad_bottom;
    int32_t pad_left;
    int32_t pad_right;

} TM2_SpaceToBatchNDParam;

typedef struct
{
    int32_t dilation_x;
    int32_t dilation_y;
    int32_t crop_top;
    int32_t crop_bottom;
    int32_t crop_left;
    int32_t crop_right;

} TM2_BatchToSpaceNDParam;

typedef struct
{
    int32_t num_args;
    int32_t offset_c;
    int32_t offset_h;
    int32_t offset_w;
    int32_t crop_h;
    int32_t crop_w;
    bool center_crop;
    int32_t axis;
    int32_t flag;
} TM2_CropParam;

typedef struct
{
    int32_t pooled_width;
    int32_t pooled_height;
    float spatial_scale;
} TM2_RoialignParam;

typedef struct
{
    int32_t pooled_w;
    int32_t pooled_h;
    float spatial_scale;
    int32_t output_dim;
} TM2_PsroipoolingParam;

typedef struct
{
    int32_t axis;
} TM2_ExpanddimsParam;

typedef struct
{
    int32_t type;
} TM2_UnaryParam;

typedef struct
{
    int32_t bias_size;
} TM2_BiasParam;

typedef struct
{
    float threshold;
} TM2_ThresholdParam;

typedef struct
{
    float alpha;
    float beta;
} TM2_HardsigmoidParam;

typedef struct
{
    int32_t num_output;
    int32_t input_dim;
    int32_t bias_term;
    int32_t weight_data_size;
} TM2_EmbedParam;

typedef struct
{
    float eps;
} TM2_InstanceNormParam;

typedef struct
{
    int32_t across_channels;
    int32_t normalize_variance;
    float eps;
} TM2_MVNParam;

typedef struct
{
    int32_t type_from;
    int32_t type_to;
} TM2_CastParam;

typedef struct
{
    float alpha;
    float beta;
} TM2_HardSwishParam;

typedef struct
{
    int32_t resize_type;    // 1=nearest  2=bilinear  3=bicubic
    float width_scale;
    float height_scale;
    int32_t output_width;
    int32_t output_height;
} TM2_InterpParam;

typedef struct
{
    float alpha;
    float lambda;
} TM2_SeluParam;

typedef struct
{
    float alpha;
} TM2_EluParam;

typedef struct
{
    uint32_t type;
} TM2_LogicalParam;

typedef struct
{
    int32_t axis;
    int32_t indices_num;
    tm_bool_t is_onnx;
} TM2_GatherParam;
typedef struct
{
    tm_uoffset_t offset_tr_shape;
} TM2_TransposeParam;
typedef struct
{
    int32_t type;
} TM2_ComparisonParam;
typedef struct
{
    int block_size;
} TM2_SpaceToDepthParam;

typedef struct
{
    int block_size;
} TM2_DepthToSpaceParam;

typedef struct
{
    int output_shape_size0;
    int output_shape_size1;
    int default_value;
} TM2_SparseToDenseParam;

typedef struct
{
    float max;
    float min;
} TM2_ClipParam;

typedef struct
{
    tm_uoffset_t offset_vi_axises;
} TM2_UnsqueezeParam;

typedef struct
{
    int32_t axis;
    int32_t keepdim;
} TM2_ReduceL2Param;

typedef struct
{
    int32_t axis;
} TM2_LogSoftmaxParam;

typedef struct
{
    int32_t axis;
    tm_bool_t is_onnx;
} TM2_ScatterParam;

typedef struct
{
    int32_t paddingType;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
} TM2_L2PoolParam;

typedef struct
{
    int32_t frame_flag;
    int32_t reps_size;
    tm_uoffset_t offset_reps;
} TM2_TileParam;

#ifdef __cplusplus
}
#endif

#endif
