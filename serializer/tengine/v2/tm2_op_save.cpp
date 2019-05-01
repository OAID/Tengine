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
 * Copyright (c) 2019, Open AI Lab
 * Author: jingyou@openailab.com
 */
#include <string.h>

#include "tm2_format.h"
#include "tm2_op_serializer.hpp"

namespace TEngine {

namespace TMSerializer2 {

inline void SetTmOperator(TM2_Operator* tm_op, const uint32_t op_type, const tm_uoffset_t offset)
{
    tm_op->op_ver = TM2_OP_VER;
    tm_op->operator_type = op_type;
    tm_op->offset_t_param = offset;
}

tm_uoffset_t SaveTmAccuracyOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ACCURACY, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmBatchNormOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    BatchNormParam* p = (dynamic_cast<BatchNorm*>(op))->GetParam();
    TM2_BatchNormParam tm_param;
    tm_param.rescale_factor = p->rescale_factor;
    tm_param.eps = p->eps;
    tm_param.caffe_flavor = p->caffe_flavor;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_BATCHNORMALIZATION,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_BatchNormParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmConcatOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ConcatParam* p = (dynamic_cast<Concat*>(op))->GetParam();
    TM2_ConcatParam tm_param;
    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CONCAT,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ConcatParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmConstOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CONST, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmConvOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ConvParam* p = (dynamic_cast<Convolution*>(op))->GetParam();
    TM2_ConvParam tm_param;

    tm_param.kernel_h = p->kernel_h;
    tm_param.kernel_w = p->kernel_w;
    tm_param.stride_h = p->stride_h;
    tm_param.stride_w = p->stride_w;
    tm_param.dilation_h = p->dilation_h;
    tm_param.dilation_w = p->dilation_w;
    tm_param.input_channel = p->input_channel;
    tm_param.output_channel = p->output_channel;
    tm_param.group = p->group;
    tm_param.activation = p->activation;
    tm_param.pad_h0 = p->pad_h0;
    tm_param.pad_h1 = p->pad_h1;
    tm_param.pad_w0 = p->pad_w0;
    tm_param.pad_w1 = p->pad_w1;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CONVOLUTION,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ConvParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmDeconvOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    DeconvParam* p = (dynamic_cast<Deconvolution*>(op))->GetParam();
    TM2_DeconvParam tm_param;

    tm_param.kernel_h = p->kernel_h;
    tm_param.kernel_w = p->kernel_w;
    tm_param.stride_h = p->stride_h;
    tm_param.stride_w = p->stride_w;
    tm_param.pad_w0 = p->pad_w0;
    tm_param.pad_w1 = p->pad_w1;
    tm_param.pad_h0 = p->pad_h0;
    tm_param.pad_h1 = p->pad_h1;
    tm_param.num_output = p->num_output;
    tm_param.dilation_h = p->dilation_h;
    tm_param.dilation_w = p->dilation_w;
    tm_param.group = p->group;
    tm_param.activation = p->activation;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_DECONVOLUTION,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_DeconvParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmDetectionOutputOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    DetectionOutputParam* p = (dynamic_cast<DetectionOutput*>(op))->GetParam();
    TM2_DetectionOutputParam tm_param;
    tm_param.num_classes = p->num_classes;
    tm_param.keep_top_k = p->keep_top_k;
    tm_param.nms_top_k = p->nms_top_k;
    tm_param.confidence_threshold = p->confidence_threshold;
    tm_param.nms_threshold = p->nms_threshold;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_DETECTIONOUTPUT,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_DetectionOutputParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmDropoutOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_DROPOUT, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmEltwiseOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    EltwiseParam* p = (dynamic_cast<Eltwise*>(op))->GetParam();
    TM2_EltwiseParam tm_param;
    tm_param.type = p->type;
    tm_param.caffe_flavor = p->caffe_flavor;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ELTWISE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_EltwiseParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmFCOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    FCParam* p = (dynamic_cast<FullyConnected*>(op))->GetParam();
    TM2_FCParam tm_param;
    tm_param.num_output = p->num_output;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_FULLYCONNECTED,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_FCParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmFlattenOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    FlattenParam* p = (dynamic_cast<Flatten*>(op))->GetParam();
    TM2_FlattenParam tm_param;
    tm_param.axis = p->axis;
    tm_param.end_axis = p->end_axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_FLATTEN,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_FlattenParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmInputOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_INPUTOP, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmLRNOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    LRNParam* p = (dynamic_cast<LRN*>(op))->GetParam();
    TM2_LRNParam tm_param;
    tm_param.local_size = p->local_size;
    tm_param.alpha = p->alpha;
    tm_param.beta = p->beta;
    tm_param.norm_region = p->norm_region;
    tm_param.k = p->k;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_LRN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_LRNParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmNormalizeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    NormalizeParam* p = (dynamic_cast<Normalize*>(op))->GetParam();
    TM2_NormalizeParam tm_param;
    tm_param.across_spatial = p->across_spatial;
    tm_param.channel_shared = p->channel_shared;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_NORMALIZE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_NormalizeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmPermuteOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    PermuteParam* p = (dynamic_cast<Permute*>(op))->GetParam();
    TM2_PermuteParam tm_param;
    tm_param.flag = p->flag;
    tm_param.order0 = p->order0;
    tm_param.order1 = p->order1;
    tm_param.order2 = p->order2;
    tm_param.order3 = p->order3;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_PERMUTE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PermuteParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmPoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    PoolParam* p = (dynamic_cast<Pooling*>(op))->GetParam();
    TM2_PoolParam tm_param;
    tm_param.alg = p->alg;
    tm_param.kernel_h = p->kernel_h;
    tm_param.kernel_w = p->kernel_w;
    tm_param.stride_h = p->stride_h;
    tm_param.stride_w = p->stride_w;
    tm_param.global = p->global;
    tm_param.caffe_flavor = p->caffe_flavor;
    tm_param.pad_h0 = p->pad_h0;
    tm_param.pad_w0 = p->pad_w0;
    tm_param.pad_h1 = p->pad_h1;
    tm_param.pad_w1 = p->pad_w1;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_POOLING,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PoolParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmPreluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_PRELU, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmPriorBoxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    PriorBoxParam* p = (dynamic_cast<PriorBox*>(op))->GetParam();
    TM2_PriorBoxParam tm_param;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->min_size.size();
    TM2_Vector_floats* v_minsizes = ( TM2_Vector_floats* )malloc(vector_size);
    v_minsizes->v_num = p->min_size.size();
    for(unsigned int i = 0; i < p->min_size.size(); i++)
    {
        v_minsizes->data[i] = p->min_size[i];
    }
    tm_param.offset_vf_min_size = WriteTmObject(start_ptr, cur_pos, v_minsizes, vector_size);
    free(v_minsizes);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->max_size.size();
    TM2_Vector_floats* v_maxsizes = ( TM2_Vector_floats* )malloc(vector_size);
    v_maxsizes->v_num = p->max_size.size();
    for(unsigned int i = 0; i < p->max_size.size(); i++)
    {
        v_maxsizes->data[i] = p->max_size[i];
    }
    tm_param.offset_vf_max_size = WriteTmObject(start_ptr, cur_pos, v_maxsizes, vector_size);
    free(v_maxsizes);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->variance.size();
    TM2_Vector_floats* v_variance = ( TM2_Vector_floats* )malloc(vector_size);
    v_variance->v_num = p->variance.size();
    for(unsigned int i = 0; i < p->variance.size(); i++)
    {
        v_variance->data[i] = p->variance[i];
    }
    tm_param.offset_vf_variance = WriteTmObject(start_ptr, cur_pos, v_variance, vector_size);
    free(v_variance);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->aspect_ratio.size();
    TM2_Vector_floats* v_ratios = ( TM2_Vector_floats* )malloc(vector_size);
    v_ratios->v_num = p->aspect_ratio.size();
    for(unsigned int i = 0; i < p->aspect_ratio.size(); i++)
    {
        v_ratios->data[i] = p->aspect_ratio[i];
    }
    tm_param.offset_vf_aspect_ratio = WriteTmObject(start_ptr, cur_pos, v_ratios, vector_size);
    free(v_ratios);

    tm_param.flip = p->flip;
    tm_param.clip = p->clip;
    tm_param.img_size = p->img_size;
    tm_param.img_h = p->img_h;
    tm_param.img_w = p->img_w;
    tm_param.step_w = p->step_w;
    tm_param.step_h = p->step_h;
    tm_param.offset = p->offset;
    tm_param.num_priors = p->num_priors_;
    tm_param.out_dim = p->out_dim_;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_PRIORBOX,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PriorBoxParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmRegionOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    RegionParam* p = (dynamic_cast<Region*>(op))->GetParam();
    TM2_RegionParam tm_param;
    tm_param.num_classes = p->num_classes;
    tm_param.side = p->side;
    tm_param.num_box = p->num_box;
    tm_param.coords = p->coords;
    tm_param.confidence_threshold = p->confidence_threshold;
    tm_param.nms_threshold = p->nms_threshold;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->biases.size();
    TM2_Vector_floats* v_biases = ( TM2_Vector_floats* )malloc(vector_size);
    v_biases->v_num = p->biases.size();
    for(unsigned int i = 0; i < p->biases.size(); i++)
    {
        v_biases->data[i] = p->biases[i];
    }
    tm_param.offset_vf_biases = WriteTmObject(start_ptr, cur_pos, v_biases, vector_size);
    free(v_biases);

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_REGION,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_RegionParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReLuOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ReLuParam* p = (dynamic_cast<ReLu*>(op))->GetParam();
    TM2_ReLuParam tm_param;
    tm_param.negative_slope = p->negative_slope;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RELU, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReLuParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmRelu6Op(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RELU6, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReorgOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ReorgParam* p = (dynamic_cast<Reorg*>(op))->GetParam();
    TM2_ReorgParam tm_param;
    tm_param.stride = p->stride;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_REORG,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReorgParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReshapeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ReshapeParam* p = (dynamic_cast<Reshape*>(op))->GetParam();
    TM2_ReshapeParam tm_param;

    tm_param.dim_0 = p->dim_0;
    tm_param.dim_1 = p->dim_1;
    tm_param.dim_2 = p->dim_2;
    tm_param.dim_3 = p->dim_3;
    tm_param.dim_size = p->dim_size;
    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RESHAPE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReshapeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmResizeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ResizeParam* p = (dynamic_cast<Resize*>(op))->GetParam();
    TM2_ResizeParam tm_param;
    tm_param.scale_x = p->scale_w;
    tm_param.scale_y = p->scale_h;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_BILINEARRESIZE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ResizeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmROIPoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ROIPoolingParam* p = (dynamic_cast<ROIPooling*>(op))->GetParam();
    TM2_ROIPoolingParam tm_param;
    tm_param.pooled_h = p->pooled_h;
    tm_param.pooled_w = p->pooled_w;
    tm_param.spatial_scale = p->spatial_scale;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ROIPOOLING,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ROIPoolingParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmRPNOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    RPNParam* p = (dynamic_cast<RPN*>(op))->GetParam();
    TM2_RPNParam tm_param;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->ratios.size();
    TM2_Vector_floats* v_ratios = ( TM2_Vector_floats* )malloc(vector_size);
    v_ratios->v_num = p->ratios.size();
    for(unsigned int i = 0; i < p->ratios.size(); i++)
    {
        v_ratios->data[i] = p->ratios[i];
    }
    tm_param.offset_vf_ratios = WriteTmObject(start_ptr, cur_pos, v_ratios, vector_size);
    free(v_ratios);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->anchor_scales.size();
    TM2_Vector_floats* v_scales = ( TM2_Vector_floats* )malloc(vector_size);
    v_scales->v_num = p->anchor_scales.size();
    for(unsigned int i = 0; i < p->anchor_scales.size(); i++)
    {
        v_scales->data[i] = p->anchor_scales[i];
    }
    tm_param.offset_vf_anchor_scales = WriteTmObject(start_ptr, cur_pos, v_scales, vector_size);
    free(v_scales);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->anchors_.size() * 4;
    TM2_Vector_anchors* v_anchors = ( TM2_Vector_anchors* )malloc(vector_size);
    v_anchors->v_num = p->anchors_.size();
    for(unsigned int i = 0; i < p->anchors_.size(); i++)
    {
        v_anchors->data[i][0] = p->anchors_[i].x0;
        v_anchors->data[i][1] = p->anchors_[i].y0;
        v_anchors->data[i][2] = p->anchors_[i].x1;
        v_anchors->data[i][3] = p->anchors_[i].y1;
    }
    tm_param.offset_va_anchors = WriteTmObject(start_ptr, cur_pos, v_anchors, vector_size);
    free(v_anchors);

    tm_param.feat_stride = p->feat_stride;
    tm_param.basesize = p->basesize;
    tm_param.min_size = p->min_size;
    tm_param.per_nms_topn = p->per_nms_topn;
    tm_param.post_nms_topn = p->post_nms_topn;
    tm_param.nms_thresh = p->nms_thresh;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RPN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_RPNParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmScaleOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ScaleParam* p = (dynamic_cast<Scale*>(op))->GetParam();
    TM2_ScaleParam tm_param;
    tm_param.axis = p->axis;
    tm_param.num_axes = p->num_axes;
    tm_param.bias_term = p->bias_term;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SCALE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ScaleParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSliceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SliceParam* p = (dynamic_cast<Slice*>(op))->GetParam();
    TM2_SliceParam tm_param;

    tm_param.axis = p->axis;
    tm_param.iscaffe = p->iscaffe;

    if((p->slice_point_).size())
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * (p->slice_point_).size();
        TM2_Vector_dims* v_slice_points = ( TM2_Vector_dims* )malloc(vector_size);
        v_slice_points->v_num = (p->slice_point_).size();
        for(unsigned int i = 0; i < (p->slice_point_).size(); i++)
        {
            v_slice_points->dims[i] = p->slice_point_[i];
        }
        tm_param.offset_vi_slice_points = WriteTmObject(start_ptr, cur_pos, v_slice_points, vector_size);
        free(v_slice_points);
    }
    else
        tm_param.offset_vi_slice_points = TM2_NOT_SET;

    if((p->begin_).size())
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * (p->begin_).size();
        TM2_Vector_dims* v_begins = ( TM2_Vector_dims* )malloc(vector_size);
        v_begins->v_num = (p->begin_).size();
        for(unsigned int i = 0; i < (p->begin_).size(); i++)
        {
            v_begins->dims[i] = p->begin_[i];
        }
        tm_param.offset_vi_begins = WriteTmObject(start_ptr, cur_pos, v_begins, vector_size);
        free(v_begins);
    }
    else
        tm_param.offset_vi_begins = TM2_NOT_SET;

    if((p->size_).size())
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * (p->size_).size();
        TM2_Vector_dims* v_sizes = ( TM2_Vector_dims* )malloc(vector_size);
        v_sizes->v_num = (p->size_).size();
        for(unsigned int i = 0; i < (p->size_).size(); i++)
        {
            v_sizes->dims[i] = p->size_[i];
        }
        tm_param.offset_vi_sizes = WriteTmObject(start_ptr, cur_pos, v_sizes, vector_size);
        free(v_sizes);
    }
    else
        tm_param.offset_vi_sizes = TM2_NOT_SET;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SLICE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SliceParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSoftmaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SoftmaxParam* p = (dynamic_cast<Softmax*>(op))->GetParam();
    TM2_SoftmaxParam tm_param;
    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SOFTMAX,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SoftmaxParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSplitOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SPLIT, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmDetectionPostProcessOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    DetectionPostProcessParam* p = (dynamic_cast<DetectionPostProcess*>(op))->GetParam();
    TM2_DetectionPostProcessParam tm_param;

    tm_param.max_detections = p->max_detections;
    tm_param.max_classes_per_detection = p->max_classes_per_detection;
    tm_param.nms_score_threshold = p->nms_score_threshold;
    tm_param.nms_iou_threshold = p->nms_iou_threshold;
    tm_param.num_classes = p->num_classes;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->scales.size();
    TM2_Vector_floats* v_scales = ( TM2_Vector_floats* )malloc(vector_size);
    v_scales->v_num = p->scales.size();
    for(unsigned int i = 0; i < p->scales.size(); i++)
    {
        v_scales->data[i] = p->scales[i];
    }
    tm_param.offset_vf_scales = WriteTmObject(start_ptr, cur_pos, v_scales, vector_size);

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_DETECTIONPOSTPROCESS,
                   WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_DetectionPostProcessParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmGemmOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    GemmParam* p = (dynamic_cast<Gemm*>(op))->GetParam();
    TM2_GemmParam tm_param;

    tm_param.alpha = p->alpha;
    tm_param.beta = p->beta;
    tm_param.transA = p->transA;
    tm_param.transB = p->transB;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_GEMM,
                   WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_GemmParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmGenericOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    GenericParam* p = (dynamic_cast<Generic*>(op))->GetParam();
    TM2_GenericParam tm_param;

    tm_param.max_input_num = p->max_input_num;
    tm_param.max_output_num = p->max_output_num;

    TM2_String op_name;
    op_name.size = strlen(p->op_name) + 1;  // including trailing \0
    op_name.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, p->op_name, op_name.size);
    tm_param.offset_s_opname = WriteTmObject(start_ptr, cur_pos, &op_name, sizeof(TM2_String));

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_GENERIC,
                   WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_GenericParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmLogisticOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_LOGISTIC, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmLstmOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    LSTMParam* p = (dynamic_cast<LSTM*>(op))->GetParam();
    TM2_LstmParam tm_param;

    tm_param.forget_bias = p->forget_bias;
    tm_param.clip = p->clip;
    tm_param.output_len = p->output_len;
    tm_param.sequence_len = p->sequence_len;
    tm_param.input_size = p->input_size;
    tm_param.hidden_size = p->hidden_size;
    tm_param.cell_size = p->cell_size;
    tm_param.has_peephole = p->has_peephole;
    tm_param.has_projection = p->has_projection;
    tm_param.has_clip = p->has_clip;
    tm_param.has_bias = p->has_bias;
    tm_param.has_init_state = p->has_init_state;
    tm_param.forget_act = p->forget_act;
    tm_param.input_act = p->input_act;
    tm_param.output_act = p->output_act;
    tm_param.cellin_act = p->cellin_act;
    tm_param.cellout_act = p->cellout_act;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_LSTM,
                   WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_LstmParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmRnnOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    RNNParam* p = (dynamic_cast<RNN*>(op))->GetParam();
    TM2_RnnParam tm_param;

    tm_param.clip = p->clip;
    tm_param.output_len = p->output_len;
    tm_param.sequence_len = p->sequence_len;
    tm_param.input_size = p->input_size;
    tm_param.hidden_size = p->hidden_size;
    tm_param.has_clip = p->has_clip;
    tm_param.has_bias = p->has_bias;
    tm_param.has_init_state = p->has_init_state;
    tm_param.activation = p->activation;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RNN,
                   WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_RnnParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmTanhOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_TANH, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSigmoidOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SIGMOID, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSqueezeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SqueezeParam* p = (dynamic_cast<Squeeze*>(op))->GetParam();
    TM2_SqueezeParam tm_param;

    tm_param.dim_0 = p->dim_0;
    tm_param.dim_1 = p->dim_1;
    tm_param.dim_2 = p->dim_2;
    tm_param.dim_3 = p->dim_3;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SQUEEZE,
                   WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SqueezeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmFusedbnscalereluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_FUSEDBNSCALERELU, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

op_save_t SaveTmOpFunc(uint32_t op_type)
{
    switch(op_type)
    {
        case TM2_OPTYPE_ACCURACY:
            return SaveTmAccuracyOp;
        case TM2_OPTYPE_BATCHNORMALIZATION:
            return SaveTmBatchNormOp;
        case TM2_OPTYPE_BILINEARRESIZE:
            return SaveTmResizeOp;
        case TM2_OPTYPE_CONCAT:
            return SaveTmConcatOp;
        case TM2_OPTYPE_CONST:
            return SaveTmConstOp;
        case TM2_OPTYPE_CONVOLUTION:
            return SaveTmConvOp;
        case TM2_OPTYPE_DECONVOLUTION:
            return SaveTmDeconvOp;
        case TM2_OPTYPE_DETECTIONOUTPUT:
            return SaveTmDetectionOutputOp;
        case TM2_OPTYPE_DROPOUT:
            return SaveTmDropoutOp;
        case TM2_OPTYPE_ELTWISE:
            return SaveTmEltwiseOp;
        case TM2_OPTYPE_FLATTEN:
            return SaveTmFlattenOp;
        case TM2_OPTYPE_FULLYCONNECTED:
            return SaveTmFCOp;
        case TM2_OPTYPE_INPUTOP:
            return SaveTmInputOp;
        case TM2_OPTYPE_LRN:
            return SaveTmLRNOp;
        case TM2_OPTYPE_NORMALIZE:
            return SaveTmNormalizeOp;
        case TM2_OPTYPE_PERMUTE:
            return SaveTmPermuteOp;
        case TM2_OPTYPE_POOLING:
            return SaveTmPoolingOp;
        case TM2_OPTYPE_PRELU:
            return SaveTmPreluOp;
        case TM2_OPTYPE_PRIORBOX:
            return SaveTmPriorBoxOp;
        case TM2_OPTYPE_REGION:
            return SaveTmRegionOp;
        case TM2_OPTYPE_RELU:
            return SaveTmReLuOp;
        case TM2_OPTYPE_RELU6:
            return SaveTmRelu6Op;
        case TM2_OPTYPE_REORG:
            return SaveTmReorgOp;
        case TM2_OPTYPE_RESHAPE:
            return SaveTmReshapeOp;
        case TM2_OPTYPE_ROIPOOLING:
            return SaveTmROIPoolingOp;
        case TM2_OPTYPE_RPN:
            return SaveTmRPNOp;
        case TM2_OPTYPE_SCALE:
            return SaveTmScaleOp;
        case TM2_OPTYPE_SLICE:
            return SaveTmSliceOp;
        case TM2_OPTYPE_SOFTMAX:
            return SaveTmSoftmaxOp;
        case TM2_OPTYPE_SPLIT:
            return SaveTmSplitOp;
        case TM2_OPTYPE_DETECTIONPOSTPROCESS:
            return SaveTmDetectionPostProcessOp;
        case TM2_OPTYPE_GEMM:
            return SaveTmGemmOp;
        case TM2_OPTYPE_GENERIC:
            return SaveTmGenericOp;
        case TM2_OPTYPE_LOGISTIC:
            return SaveTmLogisticOp;
        case TM2_OPTYPE_LSTM:
            return SaveTmLstmOp;
        case TM2_OPTYPE_RNN:
            return SaveTmRnnOp;
        case TM2_OPTYPE_TANH:
            return SaveTmTanhOp;
        case TM2_OPTYPE_SIGMOID:
            return SaveTmSigmoidOp;
        case TM2_OPTYPE_SQUEEZE:
            return SaveTmSqueezeOp;
        case TM2_OPTYPE_FUSEDBNSCALERELU:
            return SaveTmFusedbnscalereluOp;
        default:
            LOG_ERROR() << "Operator #" << op_type << " not supported in tengine model yet\n";
            return nullptr;
    }
}

}    // namespace TMSerializer2

}    // namespace TEngine
