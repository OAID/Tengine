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
#include "tm1_op_serializer.hpp"

namespace TEngine {

namespace TMSerializer1 {

inline void SetTmOperator(TM_Operator* tm_op, const uint32_t op_type, const tm_uoffset_t offset1,
                          const tm_uoffset_t offset2)
{
    tm_op->operator_type = op_type;
    tm_op->offset_t_quantop = offset1;
    tm_op->offset_t_param = offset2;
}

static tm_uoffset_t SaveTmAccuracyOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_ACCURACY, NOT_SET, NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmBatchNormOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    BatchNormParam* p = (dynamic_cast<BatchNorm*>(op))->GetParam();
    TM_BatchNormParam tm_param;
    tm_param.rescale_factor = p->rescale_factor;
    tm_param.eps = p->eps;
    tm_param.caffe_flavor = p->caffe_flavor;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_BATCHNORMALIZATION, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_BatchNormParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmConcatOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ConcatParam* p = (dynamic_cast<Concat*>(op))->GetParam();
    TM_ConcatParam tm_param;
    tm_param.axis = p->axis;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_CONCAT, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_ConcatParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmConstOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_CONST, NOT_SET, NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmConvOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ConvParam* p = (dynamic_cast<Convolution*>(op))->GetParam();
    TM_ConvParam tm_param;
    tm_param.kernel_h = p->kernel_h;
    tm_param.kernel_w = p->kernel_w;
    tm_param.stride_h = p->stride_h;
    tm_param.stride_w = p->stride_w;
    tm_param.dilation_h = p->dilation_h;
    tm_param.dilation_w = p->dilation_w;
    tm_param.output_channel = p->output_channel;
    tm_param.activation = p->activation;
    tm_param.group = p->group;
    tm_param.pad_h = p->pad_h0;
    tm_param.pad_w = p->pad_w0;
    tm_param.pads[0] = p->pad_h0;
    tm_param.pads[1] = p->pad_w0;
    tm_param.pads[2] = p->pad_h1;
    tm_param.pads[3] = p->pad_w1;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_CONVOLUTION, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_ConvParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmDeconvOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    DeconvParam* p = (dynamic_cast<Deconvolution*>(op))->GetParam();
    TM_DeconvParam tm_param;

    tm_param.kernel_size = p->kernel_h;
    tm_param.stride = p->stride_h;
    tm_param.pad = p->pad_w0;
    tm_param.num_output = p->num_output;
    tm_param.dilation = p->dilation_h;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_DECONVOLUTION, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_DeconvParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmDetectionOutputOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    DetectionOutputParam* p = (dynamic_cast<DetectionOutput*>(op))->GetParam();
    TM_DetectionOutputParam tm_param;
    tm_param.num_classes = p->num_classes;
    tm_param.keep_top_k = p->keep_top_k;
    tm_param.nms_top_k = p->nms_top_k;
    tm_param.confidence_threshold = p->confidence_threshold;
    tm_param.nms_threshold = p->nms_threshold;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_DETECTIONOUTPUT, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_DetectionOutputParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmDropoutOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_DROPOUT, NOT_SET, NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmEltwiseOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    EltwiseParam* p = (dynamic_cast<Eltwise*>(op))->GetParam();
    TM_EltwiseParam tm_param;
    tm_param.type = p->type;
    tm_param.caffe_flavor = p->caffe_flavor;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_ELTWISE, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_EltwiseParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmFCOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    FCParam* p = (dynamic_cast<FullyConnected*>(op))->GetParam();
    TM_FCParam tm_param;
    tm_param.num_output = p->num_output;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_FULLYCONNECTED, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_FCParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmFlattenOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    FlattenParam* p = (dynamic_cast<Flatten*>(op))->GetParam();
    TM_FlattenParam tm_param;
    tm_param.axis = p->axis;
    tm_param.end_axis = p->end_axis;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_FLATTEN, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_FlattenParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmInputOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_INPUTOP, NOT_SET, NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmLRNOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    LRNParam* p = (dynamic_cast<LRN*>(op))->GetParam();
    TM_LRNParam tm_param;
    tm_param.local_size = p->local_size;
    tm_param.alpha = p->alpha;
    tm_param.beta = p->beta;
    tm_param.norm_region = p->norm_region;
    tm_param.k = p->k;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_LRN, NOT_SET, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_LRNParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmNormalizeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    NormalizeParam* p = (dynamic_cast<Normalize*>(op))->GetParam();
    TM_NormalizeParam tm_param;
    tm_param.across_spatial = p->across_spatial;
    tm_param.channel_shared = p->channel_shared;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_NORMALIZE, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_NormalizeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmPermuteOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    PermuteParam* p = (dynamic_cast<Permute*>(op))->GetParam();
    TM_PermuteParam tm_param;
    tm_param.flag = p->flag;
    tm_param.order0 = p->order0;
    tm_param.order1 = p->order1;
    tm_param.order2 = p->order2;
    tm_param.order3 = p->order3;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_PERMUTE, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_PermuteParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmPoolOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    PoolParam* p = (dynamic_cast<Pooling*>(op))->GetParam();
    TM_PoolParam tm_param;
    tm_param.alg = p->alg;
    tm_param.kernel_h = p->kernel_h;
    tm_param.kernel_w = p->kernel_w;
    tm_param.pad_h = p->pad_h0;
    tm_param.pad_w = p->pad_w0;
    tm_param.stride_h = p->stride_h;
    tm_param.stride_w = p->stride_w;
    tm_param.global = p->global;
    tm_param.caffe_flavor = p->caffe_flavor;
    tm_param.kernel_shape[0] = p->kernel_h;
    tm_param.kernel_shape[1] = p->kernel_w;
    tm_param.strides[0] = p->stride_h;
    tm_param.strides[1] = p->stride_w;
    tm_param.pads[0] = p->pad_h0;
    tm_param.pads[1] = p->pad_w0;
    tm_param.pads[2] = p->pad_h1;
    tm_param.pads[3] = p->pad_w1;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_POOLING, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_PoolParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmPreluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_PRELU, NOT_SET, NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmPriorBoxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    PriorBoxParam* p = (dynamic_cast<PriorBox*>(op))->GetParam();
    TM_PriorBoxParam tm_param;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->min_size.size();
    TM_Vector_floats* v_minsizes = ( TM_Vector_floats* )malloc(vector_size);
    v_minsizes->v_num = p->min_size.size();
    for(unsigned int i = 0; i < p->min_size.size(); i++)
    {
        v_minsizes->data[i] = p->min_size[i];
    }
    tm_param.offset_vf_min_size = WriteTmObject(start_ptr, cur_pos, v_minsizes, vector_size);
    free(v_minsizes);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->max_size.size();
    TM_Vector_floats* v_maxsizes = ( TM_Vector_floats* )malloc(vector_size);
    v_maxsizes->v_num = p->max_size.size();
    for(unsigned int i = 0; i < p->max_size.size(); i++)
    {
        v_maxsizes->data[i] = p->max_size[i];
    }
    tm_param.offset_vf_max_size = WriteTmObject(start_ptr, cur_pos, v_maxsizes, vector_size);
    free(v_maxsizes);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->variance.size();
    TM_Vector_floats* v_variance = ( TM_Vector_floats* )malloc(vector_size);
    v_variance->v_num = p->variance.size();
    for(unsigned int i = 0; i < p->variance.size(); i++)
    {
        v_variance->data[i] = p->variance[i];
    }
    tm_param.offset_vf_variance = WriteTmObject(start_ptr, cur_pos, v_variance, vector_size);
    free(v_variance);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->aspect_ratio.size();
    TM_Vector_floats* v_ratios = ( TM_Vector_floats* )malloc(vector_size);
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

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_PRIORBOX, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_PriorBoxParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmRegionOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    RegionParam* p = (dynamic_cast<Region*>(op))->GetParam();
    TM_RegionParam tm_param;
    tm_param.num_classes = p->num_classes;
    tm_param.side = p->side;
    tm_param.num_box = p->num_box;
    tm_param.coords = p->coords;
    tm_param.confidence_threshold = p->confidence_threshold;
    tm_param.nms_threshold = p->nms_threshold;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->biases.size();
    TM_Vector_floats* v_biases = ( TM_Vector_floats* )malloc(vector_size);
    v_biases->v_num = p->biases.size();
    for(unsigned int i = 0; i < p->biases.size(); i++)
    {
        v_biases->data[i] = p->biases[i];
    }
    tm_param.offset_vf_biases = WriteTmObject(start_ptr, cur_pos, v_biases, vector_size);
    free(v_biases);

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_REGION, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_RegionParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmReLuOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ReLuParam* p = (dynamic_cast<ReLu*>(op))->GetParam();
    TM_ReLuParam tm_param;
    tm_param.negative_slope = p->negative_slope;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_RELU, NOT_SET, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_ReLuParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmRelu6Op(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_RELU6, NOT_SET, NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmReorgOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ReorgParam* p = (dynamic_cast<Reorg*>(op))->GetParam();
    TM_ReorgParam tm_param;
    tm_param.stride = p->stride;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_REORG, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_ReorgParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmReshapeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ReshapeParam* p = (dynamic_cast<Reshape*>(op))->GetParam();
    TM_ReshapeParam tm_param;
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
        TM_Vector_dims* v_re_shape = ( TM_Vector_dims* )malloc(vector_size);
        v_re_shape->v_num = (p->re_shape).size();
        for(unsigned int i = 0; i < (p->re_shape).size(); i++)
        {   
            v_re_shape->dims[i] = p->re_shape[i];
        }   
        tm_param.offset_re_shape = WriteTmObject(start_ptr, cur_pos, v_re_shape, vector_size);
        free(v_re_shape);
    }   
    else
        tm_param.offset_re_shape = NOT_SET;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_RESHAPE, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_ReshapeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmResizeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ResizeParam* p = (dynamic_cast<Resize*>(op))->GetParam();
    TM_ResizeParam tm_param;
    tm_param.scale_x = p->scale_w;
    tm_param.scale_y = p->scale_h;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_BILINEARRESIZE, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_ResizeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmROIPoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ROIPoolingParam* p = (dynamic_cast<ROIPooling*>(op))->GetParam();
    TM_ROIPoolingParam tm_param;
    tm_param.pooled_h = p->pooled_h;
    tm_param.pooled_w = p->pooled_w;
    tm_param.spatial_scale = p->spatial_scale;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_ROIPOOLING, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_ROIPoolingParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmRPNOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    RPNParam* p = (dynamic_cast<RPN*>(op))->GetParam();
    TM_RPNParam tm_param;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->ratios.size();
    TM_Vector_floats* v_ratios = ( TM_Vector_floats* )malloc(vector_size);
    v_ratios->v_num = p->ratios.size();
    for(unsigned int i = 0; i < p->ratios.size(); i++)
    {
        v_ratios->data[i] = p->ratios[i];
    }
    tm_param.offset_vf_ratios = WriteTmObject(start_ptr, cur_pos, v_ratios, vector_size);
    free(v_ratios);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->anchor_scales.size();
    TM_Vector_floats* v_scales = ( TM_Vector_floats* )malloc(vector_size);
    v_scales->v_num = p->anchor_scales.size();
    for(unsigned int i = 0; i < p->anchor_scales.size(); i++)
    {
        v_scales->data[i] = p->anchor_scales[i];
    }
    tm_param.offset_vf_anchor_scales = WriteTmObject(start_ptr, cur_pos, v_scales, vector_size);
    free(v_scales);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->anchors_.size() * 4;
    TM_Vector_anchors* v_anchors = ( TM_Vector_anchors* )malloc(vector_size);
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

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_RPN, NOT_SET, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_RPNParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmScaleOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    ScaleParam* p = (dynamic_cast<Scale*>(op))->GetParam();
    TM_ScaleParam tm_param;
    tm_param.axis = p->axis;
    tm_param.num_axes = p->num_axes;
    tm_param.bias_term = p->bias_term;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_SCALE, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_ScaleParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmSliceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SliceParam* p = (dynamic_cast<Slice*>(op))->GetParam();
    TM_SliceParam tm_param;
    tm_param.axis = p->axis;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_SLICE, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_SliceParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmSoftmaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    SoftmaxParam* p = (dynamic_cast<Softmax*>(op))->GetParam();
    TM_SoftmaxParam tm_param;
    tm_param.axis = p->axis;

    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_SOFTMAX, NOT_SET,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM_SoftmaxParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

static tm_uoffset_t SaveTmSplitOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    TM_Operator tm_op;
    SetTmOperator(&tm_op, TM_OPTYPE_SPLIT, NOT_SET, NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM_Operator));
}

tm_uoffset_t SaveTmOperator(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op)
{
    std::string op_str = op->GetName();

    if(op_str == OP_STR_ACCURACY)
        return SaveTmAccuracyOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_BATCHNORMALIZATION)
        return SaveTmBatchNormOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_BILINEARRESIZE)
        return SaveTmResizeOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_CONCAT)
        return SaveTmConcatOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_CONST)
        return SaveTmConstOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_CONVOLUTION)
        return SaveTmConvOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_DECONVOLUTION)
        return SaveTmDeconvOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_DETECTIONOUTPUT)
        return SaveTmDetectionOutputOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_DROPOUT)
        return SaveTmDropoutOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_ELTWISE)
        return SaveTmEltwiseOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_FLATTEN)
        return SaveTmFlattenOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_FULLYCONNECTED)
        return SaveTmFCOp(start_ptr, cur_pos, op);
    if(op_str == "Input")
        return SaveTmInputOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_LRN)
        return SaveTmLRNOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_NORMALIZE)
        return SaveTmNormalizeOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_PERMUTE)
        return SaveTmPermuteOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_POOLING)
        return SaveTmPoolOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_PRELU)
        return SaveTmPreluOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_PRIORBOX)
        return SaveTmPriorBoxOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_REGION)
        return SaveTmRegionOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_RELU)
        return SaveTmReLuOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_RELU6)
        return SaveTmRelu6Op(start_ptr, cur_pos, op);
    if(op_str == OP_STR_REORG)
        return SaveTmReorgOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_RESHAPE)
        return SaveTmReshapeOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_ROIPOOLING)
        return SaveTmROIPoolingOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_RPN)
        return SaveTmRPNOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_SCALE)
        return SaveTmScaleOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_SLICE)
        return SaveTmSliceOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_SOFTMAX)
        return SaveTmSoftmaxOp(start_ptr, cur_pos, op);
    if(op_str == OP_STR_SPLIT)
        return SaveTmSplitOp(start_ptr, cur_pos, op);

    LOG_ERROR() << "Operator " << op->GetName() << " not supported in tengine model yet\n";
    return 0;
}

}    // namespace TMSerializer1

}    // namespace TEngine
