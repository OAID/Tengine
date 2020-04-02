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
 * Copyright (c) 2018, Open AI Lab
 * Author: jingyou@openailab.com
 */
#ifndef __TM1_FORMAT_H__
#define __TM1_FORMAT_H__

#include "tm_generate.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TM_FILE_VER_MAIN 0
#define TM_FILE_VER_SUB 1
#define TM_FILE_VER_COMPILE 0

#define NOT_SET 0x00

/* Type define */
typedef uint32_t tm_uoffset_t; /* offset is 4-byte unsigned integer */
typedef uint32_t tm_size_t; /* size is 4-byte unsigned integer */
typedef uint8_t tm_bool_t; /* bool is 1-byte unsigned integer */

/* Data types */
#define TM_DT_FLOAT32 0
#define TM_DT_FLOAT16 1
#define TM_DT_INT32 2
#define TM_DT_INT8 3

/* Operator strings */
#define OP_STR_ACCURACY "Accuracy"
#define OP_STR_BATCHNORMALIZATION "BatchNormalization"
#define OP_STR_BILINEARRESIZE "BilinearResize"
#define OP_STR_CONCAT "Concat"
#define OP_STR_CONST "Const"
#define OP_STR_CONVOLUTION "Convolution"
#define OP_STR_DECONVOLUTION "Deconvolution"
#define OP_STR_DETECTIONOUTPUT "DetectionOutput"
#define OP_STR_DROPOUT "Dropout"
#define OP_STR_ELTWISE "Eltwise"
#define OP_STR_FLATTEN "Flatten"
#define OP_STR_FULLYCONNECTED "FullyConnected"
#define OP_STR_INPUTOP "InputOp"
#define OP_STR_LRN "LRN"
#define OP_STR_NORMALIZE "Normalize"
#define OP_STR_PERMUTE "Permute"
#define OP_STR_POOLING "Pooling"
#define OP_STR_PRELU "PReLU"
#define OP_STR_PRIORBOX "PriorBox"
#define OP_STR_REGION "Region"
#define OP_STR_RELU "ReLu"
#define OP_STR_RELU6 "ReLu6"
#define OP_STR_REORG "Reorg"
#define OP_STR_RESHAPE "Reshape"
#define OP_STR_ROIPOOLING "ROIPooling"
#define OP_STR_RPN "RPN"
#define OP_STR_SCALE "Scale"
#define OP_STR_SLICE "Slice"
#define OP_STR_SOFTMAX "Softmax"
#define OP_STR_SPLIT "Split"

/* Operator types */
#define TM_OPTYPE_ACCURACY 0 /* No Param                */
#define TM_OPTYPE_BATCHNORMALIZATION 1 /* TM_BatchNormParam       */
#define TM_OPTYPE_BILINEARRESIZE 2 /* TM_ResizeParam          */
#define TM_OPTYPE_CONCAT 3 /* TM_ConcatParam          */
#define TM_OPTYPE_CONST 4 /* No Param                */
#define TM_OPTYPE_CONVOLUTION 5 /* TM_ConvParam            */
#define TM_OPTYPE_DECONVOLUTION 6 /* TM_DeconvParam          */
#define TM_OPTYPE_DETECTIONOUTPUT 7 /* TM_DetectionOutputParam */
#define TM_OPTYPE_DROPOUT 8 /* No Param                */
#define TM_OPTYPE_ELTWISE 9 /* TM_EltwiseParam         */
#define TM_OPTYPE_FLATTEN 10 /* TM_FlattenParam         */
#define TM_OPTYPE_FULLYCONNECTED 11 /* TM_FCParam              */
#define TM_OPTYPE_INPUTOP 12 /* No Param                */
#define TM_OPTYPE_LRN 13 /* TM_LRNParam             */
#define TM_OPTYPE_NORMALIZE 14 /* TM_NormalizeParam       */
#define TM_OPTYPE_PERMUTE 15 /* TM_PermuteParam         */
#define TM_OPTYPE_POOLING 16 /* TM_PoolParam            */
#define TM_OPTYPE_PRELU 17 /* No Param                */
#define TM_OPTYPE_PRIORBOX 18 /* TM_PriorBoxParam        */
#define TM_OPTYPE_REGION 19 /* TM_RegionParam          */
#define TM_OPTYPE_RELU 20 /* TM_ReLuParam            */
#define TM_OPTYPE_RELU6 21 /* No Param                */
#define TM_OPTYPE_REORG 22 /* TM_ReorgParam           */
#define TM_OPTYPE_RESHAPE 23 /* TM_ReshapeParam         */
#define TM_OPTYPE_ROIPOOLING 24 /* TM_ROIPoolingParam      */
#define TM_OPTYPE_RPN 25 /* TM_RPNParam             */
#define TM_OPTYPE_SCALE 26 /* TM_ScaleParam           */
#define TM_OPTYPE_SLICE 27 /* TM_SliceParam           */
#define TM_OPTYPE_SOFTMAX 28 /* TM_SoftmaxParam         */
#define TM_OPTYPE_SPLIT 29 /* No Param                */
#define TM_OPTYPE_NUM 30

/* --------------------- -------- TM objects -------------------------------- */

typedef struct
{
    uint16_t ver_main; /* main version of Tengine model file format */
    uint16_t ver_sub; /* sub version of Tengine model file format */
    uint16_t ver_compile; /* compile version of Tengine model file format */
    tm_uoffset_t offset_root; /* offset of root table (TM_Model) */
} TM_Header;

/* Root table of Tengine model */
typedef struct
{
    tm_uoffset_t offset_vo_subgraphs; /* offset of TM_Vector_offsets <offsets of subgraphs> */
    tm_uoffset_t offset_s_mname; /* offset of string <model name> */
} TM_Model;

/* Only 1 subgraph is supported currently */
typedef struct
{
    uint32_t subgraph_id; /* subgraph id */
    tm_uoffset_t offset_vi_input_indices; /* offset of TM_Vector_indices <indices of input nodes> */
    tm_uoffset_t offset_vi_output_indices; /* offset of TM_Vector_indices <indices of output nodes> */
    tm_uoffset_t offset_vo_seq_nodes; /* offset of TM_Vector_offsets <nodes> */
    tm_uoffset_t offset_vo_tensors; /* offset of TM_Vector_offsets <tensors> */
    tm_uoffset_t offset_vo_buffers; /* offset of TM_Vector_offsets <buffers> */
    tm_uoffset_t offset_s_sname; /* offset of string <subgraph name> */
} TM_Subgraph;

typedef struct
{
    uint32_t node_id; /* node id */
    tm_uoffset_t offset_vi_input_tensors; /* offset of TM_Vector_indices <indices of input tensors> */
    tm_uoffset_t offset_vi_output_tensors; /* offset of TM_Vector_indices <indices of output tensors> */
    tm_uoffset_t offset_t_operator; /* offset of table  <operator> */
    tm_uoffset_t offset_s_nname; /* offset of string <node name> */
    tm_bool_t dynamic_shape;
} TM_Node;

typedef struct
{
    uint32_t operator_type; /* operator type */
    tm_uoffset_t offset_t_param; /* offset of table <operator param> */
    tm_uoffset_t offset_t_quantop; /* offset of table <quant op> */
} TM_Operator;

typedef struct
{
    tm_bool_t enabled;
    tm_bool_t quant_input;
    tm_bool_t dequant_output;
    tm_uoffset_t offset_vo_quantparam; /* offset of TM_Vector_offsets <quant param> */
} TM_Quantop;

typedef struct
{
    uint32_t tensor_id;
    uint32_t buffer_id;
    tm_uoffset_t offset_vd_dims; /* offset of TM_Vector_dims <dims> */
    tm_uoffset_t offset_s_tname; /* offset of string <tensor name> */
    uint8_t type;
    uint8_t data_type;
} TM_Tensor;

typedef struct
{
    tm_size_t size; /* buffer size */
    tm_uoffset_t offset_data; /* offset of buffer data */
} TM_Buffer;

typedef struct
{
    tm_size_t size; /* string size */
    tm_uoffset_t offset_data; /* offset of string data */
} TM_String;

/* ------------------------ ------- Vectors --------------------------------- */

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    tm_uoffset_t offsets[0];
} TM_Vector_offsets;

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    uint32_t indices[0];
} TM_Vector_indices;

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    int32_t dims[0];
} TM_Vector_dims;

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    float data[0];
} TM_Vector_floats;

typedef struct
{
    tm_size_t v_num; /* number of vector elements */
    float data[0][4]; /* x0, y0, x1, y1 */
} TM_Vector_anchors;

/* -------------------- ------- Operator params ----------------------------- */

typedef struct
{
    int32_t quant_method;
    int32_t quant_width;
    int32_t quant_zero;
    float min;
    float max;
    float scale;
    float float_zero;
    tm_bool_t data_quanted;
} TM_QuantParam;

typedef struct
{
    float rescale_factor;
    float eps;
    int32_t caffe_flavor;
} TM_BatchNormParam;

typedef struct
{
    int32_t axis;
} TM_ConcatParam;

typedef struct
{
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t output_channel;
    int32_t group;
    int32_t pads[4];
    int32_t activation;
} TM_ConvParam;

typedef struct
{
    int32_t kernel_size;
    int32_t stride;
    int32_t pad;
    int32_t num_output;
    int32_t dilation;
} TM_DeconvParam;

typedef struct
{
    int32_t num_classes;
    int32_t keep_top_k;
    int32_t nms_top_k;
    float confidence_threshold;
    float nms_threshold;
} TM_DetectionOutputParam;

typedef struct
{
    uint32_t type;
    int32_t caffe_flavor;
} TM_EltwiseParam;

typedef struct
{
    int32_t num_output;
} TM_FCParam;

typedef struct
{
    int32_t axis;
    int32_t end_axis;
} TM_FlattenParam;

typedef struct
{
    int32_t local_size;
    float alpha;
    float beta;
    int32_t norm_region;
    float k;
} TM_LRNParam;

typedef struct
{
    int32_t across_spatial;
    int32_t channel_shared;
} TM_NormalizeParam;

typedef struct
{
    int32_t flag;
    int32_t order0;
    int32_t order1;
    int32_t order2;
    int32_t order3;
} TM_PermuteParam;

typedef struct
{
    uint32_t alg;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t global;
    int32_t caffe_flavor;
    int32_t kernel_shape[2]; /* kernel along each axis (H, W) */
    int32_t strides[2]; /* stride along each axis (H, W) */
    int32_t pads[4]; /* [x1_begin, x2_begin, x1_end, x2_end] for each axis */
} TM_PoolParam;

typedef struct
{
    tm_uoffset_t offset_vf_min_size; /* offset of TM_Vector_floats <min_sizes> */
    tm_uoffset_t offset_vf_max_size; /* offset of TM_Vector_floats <max_sizes> */
    tm_uoffset_t offset_vf_variance; /* offset of TM_Vector_floats <variances> */
    tm_uoffset_t offset_vf_aspect_ratio; /* offset of TM_Vector_floats <aspect_ratios> */
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
} TM_PriorBoxParam;

typedef struct
{
    int32_t num_classes;
    int32_t side;
    int32_t num_box;
    int32_t coords;
    float confidence_threshold;
    float nms_threshold;
    tm_uoffset_t offset_vf_biases; /* offset of TM_Vector_floats <biases> */
} TM_RegionParam;

typedef struct
{
    float negative_slope;
} TM_ReLuParam;

typedef struct
{
    int32_t stride;
} TM_ReorgParam;

typedef struct
{
    int32_t is_mxnet;
    int32_t reverse;
    tm_uoffset_t offset_re_shape;
} TM_ReshapeParam;



typedef struct
{
    float scale_x;
    float scale_y;
} TM_ResizeParam;

typedef struct
{
    int32_t pooled_h;
    int32_t pooled_w;
    float spatial_scale;
} TM_ROIPoolingParam;

typedef struct
{
    tm_uoffset_t offset_vf_ratios; /* pointer to TM_Vector_floats <ratios> */
    tm_uoffset_t offset_vf_anchor_scales; /* pointer to TM_Vector_floats <anchor_scales> */
    int32_t feat_stride;
    int32_t basesize;
    int32_t min_size;
    int32_t per_nms_topn;
    int32_t post_nms_topn;
    float nms_thresh;
    tm_uoffset_t offset_va_anchors; /* offset of TM_Vector_anchors <anchors> */
} TM_RPNParam;

typedef struct
{
    int32_t axis;
    int32_t num_axes;
    int32_t bias_term;
} TM_ScaleParam;

typedef struct
{
    int32_t axis;
} TM_SliceParam;

typedef struct
{
    int32_t axis;
} TM_SoftmaxParam;

#ifdef __cplusplus
}
#endif

#endif
