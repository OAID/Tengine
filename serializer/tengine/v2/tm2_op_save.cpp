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
    SetTmOperator(&tm_op, TM2_OPTYPE_CONCAT, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ConcatParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_CONVOLUTION, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ConvParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_ELTWISE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_EltwiseParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmFCOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    FCParam* p = (dynamic_cast<FullyConnected*>(op))->GetParam();
    TM2_FCParam tm_param;
    tm_param.num_output = p->num_output;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_FULLYCONNECTED, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_FCParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmFlattenOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    FlattenParam* p = (dynamic_cast<Flatten*>(op))->GetParam();
    TM2_FlattenParam tm_param;
    tm_param.axis = p->axis;
    tm_param.end_axis = p->end_axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_FLATTEN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_FlattenParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_PERMUTE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PermuteParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_POOLING, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PoolParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_PRIORBOX, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PriorBoxParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_REGION, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_RegionParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_REORG, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReorgParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReshapeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ReshapeParam* p = (dynamic_cast<Reshape*>(op))->GetParam();
    TM2_ReshapeParam tm_param;
    if(p->reverse)
        tm_param.reverse = 1;
    else
        tm_param.reverse = 0;
    if(p->is_mxnet)
        tm_param.is_mxnet = 1;
    else
        tm_param.is_mxnet = 0;
    
    if((p->re_shape).size())
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * (p->re_shape).size();
        TM2_Vector_dims* v_re_shape = ( TM2_Vector_dims* )malloc(vector_size);
        v_re_shape->v_num = (p->re_shape).size();
        for(unsigned int i = 0; i < (p->re_shape).size(); i++)
        {
            v_re_shape->dims[i] = p->re_shape[i];
        }
        tm_param.offset_re_shape = WriteTmObject(start_ptr, cur_pos, v_re_shape, vector_size);
        free(v_re_shape);
    }
    else{
        tm_param.offset_re_shape = TM2_NOT_SET;
    }


    TM2_Operator tm_op;
    tm_op.op_ver=2;
    SetTmOperator(&tm_op, TM2_OPTYPE_RESHAPE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReshapeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));

}

tm_uoffset_t SaveTmResizeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ResizeParam* p = (dynamic_cast<Resize*>(op))->GetParam();
    TM2_ResizeParam tm_param;
    tm_param.scale_x = p->scale_w;
    tm_param.scale_y = p->scale_h;
    tm_param.type = p->type;

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
    SetTmOperator(&tm_op, TM2_OPTYPE_SCALE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ScaleParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSliceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SliceParam* p = (dynamic_cast<Slice*>(op))->GetParam();
    TM2_SliceParam tm_param;

    tm_param.axis = p->axis;
    tm_param.iscaffe = p->iscaffe;
    tm_param.isonnx = p->isonnx;
    if(!tm_param.iscaffe){
        tm_param.begin = p->begin;
        tm_param.end = p->end;
    }
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
    SetTmOperator(&tm_op, TM2_OPTYPE_SLICE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SliceParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSoftmaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SoftmaxParam* p = (dynamic_cast<Softmax*>(op))->GetParam();
    TM2_SoftmaxParam tm_param;
    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SOFTMAX, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SoftmaxParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSplitOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SplitParam* p = (dynamic_cast<Split*>(op))->GetParam();
    TM2_SplitParam tm_param;
    if(p->is_caffe)
        tm_param.is_caffe = 1;
    else
        tm_param.is_caffe = 0;

    if(p->is_onnx){
        tm_param.is_onnx = 1;
    } else {
        tm_param.is_onnx = 0;
    }
    if(!p->is_caffe)
    {
        if(p->is_onnx)
            tm_param.axis = p->axis;
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * (p->split_sizes_).size();
        TM2_Vector_dims* v_split_sizes = ( TM2_Vector_dims* )malloc(vector_size);
        v_split_sizes->v_num = (p->split_sizes_).size();
        for(unsigned int i = 0; i < (p->split_sizes_).size(); i++)
        {
            v_split_sizes->dims[i] = p->split_sizes_[i];
        }
        tm_param.offset_split_sizes = WriteTmObject(start_ptr, cur_pos, v_split_sizes, vector_size);
        free(v_split_sizes);
        tm_param.split_dim = p->split_dim;
    }

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SPLIT, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SplitParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_GEMM, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_GemmParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmGenericOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    GenericParam* p = (dynamic_cast<Generic*>(op))->GetParam();
    TM2_GenericParam tm_param;

    tm_param.max_input_num = p->max_input_num;
    tm_param.max_output_num = p->max_output_num;

    TM2_String op_name;
    op_name.size = strlen(p->op_name) + 1;    // including trailing \0
    op_name.offset_data = WriteTmFileAlign1(start_ptr, cur_pos, p->op_name, op_name.size);
    tm_param.offset_s_opname = WriteTmObject(start_ptr, cur_pos, &op_name, sizeof(TM2_String));

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_GENERIC, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_GenericParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_LSTM, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_LstmParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_RNN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_RnnParam)));
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
    SetTmOperator(&tm_op, TM2_OPTYPE_SQUEEZE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SqueezeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmFusedbnscalereluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_FUSEDBNSCALERELU, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmMaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MAX, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmMinOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MIN, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmArgMaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ArgMaxParam* p = (dynamic_cast<ArgMax*>(op))->GetParam();
    TM2_ArgMaxParam tm_param;

    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ARGMAX, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ArgMaxParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmArgMinOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ArgMinParam* p = (dynamic_cast<ArgMin*>(op))->GetParam();
    TM2_ArgMinParam tm_param;

    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ARGMIN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ArgMinParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmTopKV2Op(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TopKV2Param* p = (dynamic_cast<TopKV2*>(op))->GetParam();
    TM2_TopKV2Param tm_param;

    tm_param.k = p->k;
    if(p->sorted)
        tm_param.sorted = 1;
    else
        tm_param.sorted = 0;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_TOPKV2, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_TopKV2Param)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmStridedSliceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    StridedSliceParam* p = (dynamic_cast<StridedSlice*>(op))->GetParam();
    TM2_StridedSliceParam tm_param;

    tm_param.begine_n = p->begin[0];
    tm_param.begine_c = p->begin[1];
    tm_param.begine_h = p->begin[2];
    tm_param.begine_w = p->begin[3];
    tm_param.end_n = p->end[0];
    tm_param.end_c = p->end[1];
    tm_param.end_h = p->end[2];
    tm_param.end_w = p->end[3];
    tm_param.stride_n = p->stride[0];
    tm_param.stride_c = p->stride[1];
    tm_param.stride_h = p->stride[2];
    tm_param.stride_w = p->stride[3];

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_STRIDEDSLICE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_StridedSliceParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmPadOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    PadParam* p = (dynamic_cast<Pad*>(op))->GetParam();
    TM2_PadParam tm_param;

    tm_param.mode = p->mode;
    tm_param.value = p->value;
    tm_param.pad_n_0 = p->pad_0_h;
    tm_param.pad_n_1 = p->pad_0_w;
    tm_param.pad_c_0 = p->pad_1_h;
    tm_param.pad_c_1 = p->pad_1_w;
    tm_param.pad_h_0 = p->pad_2_h;
    tm_param.pad_h_1 = p->pad_2_w;
    tm_param.pad_w_0 = p->pad_3_h;
    tm_param.pad_w_1 = p->pad_3_w;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_PAD, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PadParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReductionOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ReductionParam* p = (dynamic_cast<Reduction*>(op))->GetParam();
    TM2_ReductionParam tm_param;

    tm_param.dim_0 = p->dim_0;
    tm_param.dim_1 = p->dim_1;
    tm_param.dim_2 = p->dim_2;
    tm_param.dim_3 = p->dim_3;
    tm_param.keepdim = p->keepdim;
    tm_param.type = p->type;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_REDUCTION,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReductionParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSwapAxisOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SwapAxisParam* p = (dynamic_cast<SwapAxis*>(op))->GetParam();
    TM2_SwapAxisParam tm_param;

    tm_param.dim_0 = p->dim_0;
    tm_param.dim_1 = p->dim_1;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SWAPAXIS, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SwapAxisParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmAddnOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    AddnParam* p = (dynamic_cast<Addn*>(op))->GetParam();
    TM2_AddnParam tm_param;

    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ADDN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_AddnParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmGruOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    GRUParam* p = (dynamic_cast<GRU*>(op))->GetParam();
    TM2_GRUParam tm_param;

    tm_param.clip = p->clip;
    tm_param.output_len = p->output_len;
    tm_param.sequence_len = p->sequence_len;
    tm_param.input_size = p->input_size;
    tm_param.hidden_size = p->hidden_size;
    tm_param.has_clip = p->has_clip;
    tm_param.has_gate_bias = p->has_gate_bias;
    tm_param.has_candidate_bias = p->has_candidate_bias;
    tm_param.has_init_state = p->has_init_state;
    tm_param.mxnet_flag = p->mxnet_flag;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_GRU, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_GRUParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmUpsampleOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    UpsampleParam* p = (dynamic_cast<Upsample*>(op))->GetParam();
    TM2_UpsampleParam tm_param;

    tm_param.scale = p->scale;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_UPSAMPLE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_UpsampleParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmShuffleChannelOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ShuffleChannelParam* p = (dynamic_cast<ShuffleChannel*>(op))->GetParam();
    TM2_ShuffleChannelParam tm_param;

    tm_param.group = p->group;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SHUFFLECHANNEL, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ShuffleChannelParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSpaceToBatchNDOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SpaceToBatchNDParam* p = (dynamic_cast<SpaceToBatchND*>(op))->GetParam();
    TM2_SpaceToBatchNDParam tm_param;

    tm_param.dilation_x = p->dilation_x;
    tm_param.dilation_y = p->dilation_y;
    tm_param.pad_top = p->pad_top;
    tm_param.pad_bottom = p->pad_bottom;
    tm_param.pad_left = p->pad_left;
    tm_param.pad_right = p->pad_right;
				    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SPACETOBATCHND, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SpaceToBatchNDParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmBatchToSpaceNDOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    BatchToSpaceNDParam* p = (dynamic_cast<BatchToSpaceND*>(op))->GetParam();
    TM2_BatchToSpaceNDParam tm_param;

    tm_param.dilation_x = p->dilation_x;
    tm_param.dilation_y = p->dilation_y;
    tm_param.crop_top = p->crop_top;
    tm_param.crop_bottom = p->crop_bottom;
    tm_param.crop_left = p->crop_left;
    tm_param.crop_right = p->crop_right;
					    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_BATCHTOSPACEND, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_BatchToSpaceNDParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmCropOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    CropParam* p = (dynamic_cast<Crop*>(op))->GetParam();
    TM2_CropParam tm_param;

    tm_param.axis = p->axis;
    tm_param.center_crop = p->center_crop;
    tm_param.crop_h = p->crop_h;
    tm_param.crop_w = p->crop_w;
    tm_param.num_args = p->num_args;
    tm_param.offset_c = p->offset_c;
    tm_param.offset_h = p->offset_h;
    tm_param.offset_w = p->offset_w;
    tm_param.flag = p->flag;
					    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CROP, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_CropParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmUnaryOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    UnaryParam* p = (dynamic_cast<Unary*>(op))->GetParam();
    TM2_UnaryParam tm_param;

    tm_param.type = p->type;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_UNARY, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_UnaryParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmPsroipoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    PsroipoolingParam* p = (dynamic_cast<Psroipooling*>(op))->GetParam();
    TM2_PsroipoolingParam tm_param;

    tm_param.spatial_scale = p->spatial_scale;
    tm_param.pooled_w = p->pooled_w;
    tm_param.pooled_h = p->pooled_h;
    tm_param.output_dim = p->output_dim;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_PSROIPOOLING, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PsroipoolingParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmExpanddimsOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    ExpandDimsParam* p = (dynamic_cast<ExpandDims*>(op))->GetParam();
    TM2_ExpanddimsParam tm_param;

    tm_param.axis= p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_EXPANDDIMS, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ExpanddimsParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmRoialignOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    RoialignParam* p = (dynamic_cast<Roialign*>(op))->GetParam();
    TM2_RoialignParam tm_param;

    tm_param.spatial_scale = p->spatial_scale;
    tm_param.pooled_width = p->pooled_width;
    tm_param.pooled_height = p->pooled_height;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ROIALIGN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_RoialignParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));    
}

tm_uoffset_t SaveTmBiasOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    BiasParam* p = (dynamic_cast<Bias*>(op))->GetParam();
    TM2_BiasParam tm_param;

    tm_param.bias_size = p->bias_size;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_BIAS, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_BiasParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmThresholdOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ThresholdParam* p = (dynamic_cast<Threshold*>(op))->GetParam();
    TM2_ThresholdParam tm_param;

    tm_param.threshold = p->threshold;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_THRESHOLD, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ThresholdParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmNoopOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_NOOP, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmEmbedOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    EmbedParam* p = (dynamic_cast<Embed*>(op))->GetParam();
    TM2_EmbedParam tm_param;

    //tm_param.bias_term = p->bias_term;
    tm_param.input_dim = p->input_dim;
    tm_param.num_output = p->num_output;
    tm_param.weight_data_size = p->weight_data_size;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_EMBED, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_EmbedParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmHardsigmoidOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    HardsigmoidParam* p = (dynamic_cast<Hardsigmoid*>(op))->GetParam();
    TM2_HardsigmoidParam tm_param;

    tm_param.alpha = p->alpha;
    tm_param.beta = p->beta;
					    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_HARDSIGMOID, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_HardsigmoidParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmInstanceNormOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    InstanceNormParam* p = (dynamic_cast<InstanceNorm*>(op))->GetParam();
    TM2_InstanceNormParam tm_param;
    tm_param.eps = p->eps;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_INSTANCENORM, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_InstanceNormParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmMVNOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    MVNParam* p = (dynamic_cast<MVN*>(op))->GetParam();
    TM2_MVNParam tm_param;

    tm_param.across_channels = p->across_channels;
    tm_param.eps = p->eps;
    tm_param.normalize_variance = p->normalize_variance;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MVN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_MVNParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmAbsvalOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ABSVAL, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmCastOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    CastParam* p = (dynamic_cast<Cast*>(op))->GetParam();
    TM2_CastParam tm_param;
    tm_param.type_from = p->type_from;
    tm_param.type_to = p->type_to;
    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CAST, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_CastParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmHardSwishOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    HardswishParam* p = (dynamic_cast<Hardswish*>(op))->GetParam();
    TM2_HardSwishParam tm_param;
    tm_param.alpha = p->alpha;
    tm_param.beta = p->beta;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CAST, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_HardSwishParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));    
}
tm_uoffset_t SaveTmInterpOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    InterpParam* p = (dynamic_cast<Interp*>(op))->GetParam();
    TM2_InterpParam tm_param;
    tm_param.height_scale = p->height_scale;
    tm_param.output_height = p->output_height;
    tm_param.output_width = p->output_width;
    tm_param.resize_type = p->resize_type;
    tm_param.width_scale = p->width_scale;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_INTERP, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_InterpParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));    
}
tm_uoffset_t SaveTmSeluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    SeluParam* p = (dynamic_cast<Selu*>(op))->GetParam();
    TM2_SeluParam tm_param;
    tm_param.alpha = p->alpha;
    tm_param.lambda = p->lambda;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SELU, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SeluParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));  
}
tm_uoffset_t SaveTmEluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    EluParam* p = (dynamic_cast<Elu*>(op))->GetParam();
    TM2_EluParam tm_param;
    tm_param.alpha = p->alpha;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ELU, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_EluParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));  
}

tm_uoffset_t SaveTmBroadMulOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op){
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_BROADMUL, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmLogicalOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    LogicalParam* p = (dynamic_cast<Logical*>(op))->GetParam();
    TM2_LogicalParam tm_param;

    tm_param.type = p->type;
					    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_LOGICAL, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_LogicalParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmGatherOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    GatherParam* p = (dynamic_cast<Gather*>(op))->GetParam();
    TM2_GatherParam tm_param;

    tm_param.axis = p->axis;
    tm_param.indices_num = p->indices_num;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_GATHER, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_GatherParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmTransposeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TransposeParam* p = (dynamic_cast<Transpose*>(op))->GetParam();
    TM2_TransposeParam tm_param;
    if((p->tr_shape).size())
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * (p->tr_shape).size();
        TM2_Vector_dims* v_re_shape = ( TM2_Vector_dims* )malloc(vector_size);
        v_re_shape->v_num = (p->tr_shape).size();
        for(unsigned int i = 0; i < (p->tr_shape).size(); i++)
        {
            v_re_shape->dims[i] = p->tr_shape[i];
        }
        tm_param.offset_tr_shape = WriteTmObject(start_ptr, cur_pos, v_re_shape, vector_size);
        free(v_re_shape);
    }
    else{
        tm_param.offset_tr_shape = TM2_NOT_SET;
    }
    TM2_Operator tm_op;
    tm_op.op_ver=2;
    SetTmOperator(&tm_op, TM2_OPTYPE_TRANSPOSE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_TransposeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));  
}
tm_uoffset_t SaveTmComparisonOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ComparisonParam* p = (dynamic_cast<Comparison*>(op))->GetParam();
    TM2_ComparisonParam tm_param;

    tm_param.type = p->type;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MVN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ComparisonParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmReverseOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_REVERSE, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmSpaceToDepthOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SpaceToDepthParam* p = (dynamic_cast<SpaceToDepth*>(op))->GetParam();
    TM2_SpaceToDepthParam tm_param;

    tm_param.block_size = p->block_size;
					    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SPACETODEPTH, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SpaceToDepthParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmDepthToSpaceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    DepthToSpaceParam* p = (dynamic_cast<DepthToSpace*>(op))->GetParam();
    TM2_DepthToSpaceParam tm_param;

    tm_param.block_size = p->block_size;
					    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SPACETODEPTH, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_DepthToSpaceParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmSquaredDifferenceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SQUAREDDIFFERENCE, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmSparseToDenseOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SparseToDenseParam* p = (dynamic_cast<SparseToDense*>(op))->GetParam();
    TM2_SparseToDenseParam tm_param;

    tm_param.output_shape_size0 = p->output_shape_size0;
    tm_param.output_shape_size1 = p->output_shape_size1;
    tm_param.default_value = p->default_value;
					    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SPARSETODENSE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SparseToDenseParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmCeilOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CEIL, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmRoundOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ROUND, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmZerosLikeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ZEROSLIKE, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmClipOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ClipParam* p = (dynamic_cast<Clip*>(op))->GetParam();
    TM2_ClipParam tm_param;

    tm_param.max = p->max;
    tm_param.min = p->min;
					    
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CLIP, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ClipParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmMatMulOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MATMUL, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReduceL2Op(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ReduceL2Param* p = (dynamic_cast<ReduceL2*>(op))->GetParam();
    TM2_ReduceL2Param tm_param;

    tm_param.axis = p->axis;
    tm_param.keepdim = p->keepdim;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_REDUCEL2, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReduceL2Param)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmUnsqueezeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    UnsqueezeParam* p = (dynamic_cast<Unsqueeze*>(op))->GetParam();
    TM2_UnsqueezeParam tm_param;

    if((p->axises).size())
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * (p->axises).size();
        TM2_Vector_dims* v_axises = ( TM2_Vector_dims* )malloc(vector_size);
        v_axises->v_num = (p->axises).size();
        for(unsigned int i = 0; i < (p->axises).size(); i++)
        {
            v_axises->dims[i] = p->axises[i];
												            }
            tm_param.offset_vi_axises = WriteTmObject(start_ptr, cur_pos, v_axises, vector_size);
            free(v_axises);
        }
    else
        tm_param.offset_vi_axises = TM2_NOT_SET;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_UNSQUEEZE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_UnsqueezeParam)));
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
        case TM2_OPTYPE_SWAPAXIS:
            return SaveTmSwapAxisOp;
        case TM2_OPTYPE_GRU:
            return SaveTmGruOp;
        case TM2_OPTYPE_ADDN:
            return SaveTmAddnOp;
        case TM2_OPTYPE_MAX:
            return SaveTmMaxOp;
        case TM2_OPTYPE_MIN:
            return SaveTmMinOp;
        case TM2_OPTYPE_ARGMAX:
            return SaveTmArgMaxOp;
        case TM2_OPTYPE_ARGMIN:
            return SaveTmArgMinOp;
        case TM2_OPTYPE_TOPKV2:
            return SaveTmTopKV2Op;
        case TM2_OPTYPE_PAD:
            return SaveTmPadOp;
        case TM2_OPTYPE_STRIDEDSLICE:
            return SaveTmStridedSliceOp;
        case TM2_OPTYPE_REDUCTION:
            return SaveTmReductionOp;
        case TM2_OPTYPE_UPSAMPLE:
            return SaveTmUpsampleOp;
        case TM2_OPTYPE_SHUFFLECHANNEL:
            return SaveTmShuffleChannelOp;
        case TM2_OPTYPE_SPACETOBATCHND:
            return SaveTmSpaceToBatchNDOp;   
        case TM2_OPTYPE_BATCHTOSPACEND:
            return SaveTmBatchToSpaceNDOp;  
        case TM2_OPTYPE_RESIZE:
            return SaveTmResizeOp;	  
        case TM2_OPTYPE_CROP:
            return SaveTmCropOp;
        case TM2_OPTYPE_ROIALIGN:
            return SaveTmRoialignOp;
        case TM2_OPTYPE_PSROIPOOLING:
            return SaveTmPsroipoolingOp;
        case TM2_OPTYPE_EXPANDDIMS:
            return SaveTmExpanddimsOp;
        case TM2_OPTYPE_UNARY:
            return SaveTmUnaryOp;
        case TM2_OPTYPE_BIAS:
            return SaveTmBiasOp;  
        case TM2_OPTYPE_NOOP:
            return SaveTmNoopOp;  
        case TM2_OPTYPE_THRESHOLD:
            return SaveTmThresholdOp;  
        case TM2_OPTYPE_HARDSIGMOID:
            return SaveTmHardsigmoidOp;
        case TM2_OPTYPE_EMBED:
            return SaveTmEmbedOp;
        case TM2_OPTYPE_INSTANCENORM:
            return SaveTmInstanceNormOp;
        case TM2_OPTYPE_MVN:
            return SaveTmMVNOp; 
        case TM2_OPTYPE_ABSVAL:
            return SaveTmAbsvalOp;    
        case TM2_OPTYPE_CAST:
            return SaveTmCastOp;     
        case TM2_OPTYPE_HARDSWISH:
            return SaveTmHardSwishOp;     
        case TM2_OPTYPE_INTERP:
            return SaveTmInterpOp;    
        case TM2_OPTYPE_SELU:
            return SaveTmSeluOp;     
        case TM2_OPTYPE_ELU:
            return SaveTmEluOp;                           
        case TM2_OPTYPE_BROADMUL:
            return SaveTmBroadMulOp;
        case TM2_OPTYPE_LOGICAL:
            return SaveTmLogicalOp;   
        case TM2_OPTYPE_GATHER:
            return SaveTmGatherOp;                           
        case TM2_OPTYPE_TRANSPOSE:
            return SaveTmTransposeOp;
        case TM2_OPTYPE_COMPARISON:
            return SaveTmComparisonOp; 
        case TM2_OPTYPE_REVERSE:
            return SaveTmReverseOp;                                    
        case TM2_OPTYPE_SPACETODEPTH:
            return SaveTmSpaceToDepthOp;
        case TM2_OPTYPE_DEPTHTOSPACE:
            return SaveTmDepthToSpaceOp;
        case TM2_OPTYPE_SQUAREDDIFFERENCE:
            return SaveTmSquaredDifferenceOp;
        case TM2_OPTYPE_SPARSETODENSE:
            return SaveTmSparseToDenseOp;
        case TM2_OPTYPE_CEIL:
            return SaveTmCeilOp;
        case TM2_OPTYPE_ROUND:
            return SaveTmRoundOp;
        case TM2_OPTYPE_ZEROSLIKE:
            return SaveTmZerosLikeOp;
        case TM2_OPTYPE_CLIP:
            return SaveTmClipOp;    
	case TM2_OPTYPE_MATMUL:
    	    return SaveTmMatMulOp;	    
	case TM2_OPTYPE_REDUCEL2:
	    return SaveTmReduceL2Op;
	case TM2_OPTYPE_UNSQUEEZE:
	    return SaveTmUnsqueezeOp;
	default:
            LOG_ERROR() << "Operator #" << op_type << " not supported in tengine model yet\n";
            return nullptr;
    }
}

}    // namespace TMSerializer2

}    // namespace TEngine
