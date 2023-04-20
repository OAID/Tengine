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

#include "tm2_op_save.hpp"
// #include "utility/log.h"
// #include "tengine_ir.h"

inline void SetTmOperator(TM2_Operator* tm_op, const uint32_t op_type, const tm_uoffset_t offset)
{
    tm_op->op_ver = TM2_OP_VER;
    tm_op->operator_type = op_type;
    tm_op->offset_t_param = offset;
}

tm_uoffset_t SaveTmBatchNormOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct batchnorm_param* p = (struct batchnorm_param*)node->op.param_mem;
    TM2_BatchNormParam tm_param;
    tm_param.rescale_factor = p->rescale_factor;
    tm_param.eps = p->eps;
    tm_param.caffe_flavor = p->caffe_flavor;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_BATCHNORMALIZATION,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_BatchNormParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmConcatOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct concat_param* p = (struct concat_param*)node->op.param_mem;
    TM2_ConcatParam tm_param;
    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CONCAT, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ConcatParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmConstOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CONST, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmConvOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct conv_param* p = (struct conv_param*)node->op.param_mem;
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

tm_uoffset_t SaveTmDeconvOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct deconv_param* p = (struct deconv_param*)node->op.param_mem;
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
    tm_param.output_pad_h0 = p->output_pad_h0;
    tm_param.output_pad_w0 = p->output_pad_w0;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_DECONVOLUTION,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_DeconvParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmDetectionOutputOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct detection_output_param* p = (struct detection_output_param*)node->op.param_mem;
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

tm_uoffset_t SaveTmDropoutOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_DROPOUT, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmMishOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MISH, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmEltwiseOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct eltwise_param* p = (struct eltwise_param*)node->op.param_mem;
    TM2_EltwiseParam tm_param;
    tm_param.type = p->type;
    tm_param.caffe_flavor = p->caffe_flavor;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ELTWISE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_EltwiseParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmFCOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct fc_param* p = (struct fc_param*)node->op.param_mem;
    TM2_FCParam tm_param;
    tm_param.num_output = p->num_output;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_FULLYCONNECTED, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_FCParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmFlattenOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct flatten_param* p = (struct flatten_param*)node->op.param_mem;
    TM2_FlattenParam tm_param;
    tm_param.axis = p->axis;
    tm_param.end_axis = p->end_axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_FLATTEN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_FlattenParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmInputOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_INPUTOP, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmLRNOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct lrn_param* p = (struct lrn_param*)node->op.param_mem;
    TM2_LRNParam tm_param;
    tm_param.local_size = p->local_size;
    tm_param.alpha = p->alpha;
    tm_param.beta = p->beta;
    tm_param.norm_region = p->norm_region;
    tm_param.k = p->k;
    // tm_param.is_onnx = p->is_onnx;
    tm_param.is_onnx = 0;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_LRN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_LRNParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmNormalizeOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct normalize_param* p = (struct normalize_param*)node->op.param_mem;
    TM2_NormalizeParam tm_param;
    tm_param.across_spatial = p->across_spatial;
    tm_param.channel_shared = p->channel_shared;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_NORMALIZE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_NormalizeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmPermuteOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct permute_param* p = (struct permute_param*)node->op.param_mem;
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

tm_uoffset_t SaveTmPoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct pool_param* p = (struct pool_param*)node->op.param_mem;
    TM2_PoolParam tm_param;
    // tm_param.alg = p->alg;
    tm_param.alg = p->pool_method;
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

tm_uoffset_t SaveTmPreluOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_PRELU, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmPriorBoxOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct priorbox_param* p = (struct priorbox_param*)node->op.param_mem;
    TM2_PriorBoxParam tm_param;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->min_size_num;
    TM2_Vector_floats* v_minsizes = (TM2_Vector_floats*)malloc(vector_size);
    v_minsizes->v_num = p->min_size_num;
    for (unsigned int i = 0; i < p->min_size_num; i++)
    {
        v_minsizes->data[i] = p->min_size[i];
    }
    tm_param.offset_vf_min_size = WriteTmObject(start_ptr, cur_pos, v_minsizes, vector_size);
    free(v_minsizes);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->max_size_num;
    TM2_Vector_floats* v_maxsizes = (TM2_Vector_floats*)malloc(vector_size);
    v_maxsizes->v_num = p->max_size_num;
    for (unsigned int i = 0; i < p->max_size_num; i++)
    {
        v_maxsizes->data[i] = p->max_size[i];
    }
    tm_param.offset_vf_max_size = WriteTmObject(start_ptr, cur_pos, v_maxsizes, vector_size);
    free(v_maxsizes);

    int variance_num = 4; // tengine lite does not set the variable.
    vector_size = sizeof(tm_size_t) + sizeof(float) * variance_num;
    TM2_Vector_floats* v_variance = (TM2_Vector_floats*)malloc(vector_size);
    v_variance->v_num = variance_num;
    for (unsigned int i = 0; i < variance_num; i++)
    {
        v_variance->data[i] = p->variance[i];
    }
    tm_param.offset_vf_variance = WriteTmObject(start_ptr, cur_pos, v_variance, vector_size);
    free(v_variance);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->aspect_ratio_size;
    TM2_Vector_floats* v_ratios = (TM2_Vector_floats*)malloc(vector_size);
    v_ratios->v_num = p->aspect_ratio_size;
    for (unsigned int i = 0; i < p->aspect_ratio_size; i++)
    {
        v_ratios->data[i] = p->aspect_ratio[i];
    }
    tm_param.offset_vf_aspect_ratio = WriteTmObject(start_ptr, cur_pos, v_ratios, vector_size);
    free(v_ratios);

    tm_param.flip = p->flip;
    tm_param.clip = p->clip;
    tm_param.img_size = p->image_size;
    tm_param.img_h = p->image_h;
    tm_param.img_w = p->image_w;
    tm_param.step_w = p->step_w;
    tm_param.step_h = p->step_h;
    tm_param.offset = p->offset;
    tm_param.num_priors = p->num_priors;
    tm_param.out_dim = p->out_dim;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_PRIORBOX, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PriorBoxParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmRegionOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct region_param* p = (struct region_param*)node->op.param_mem;
    TM2_RegionParam tm_param;
    tm_param.num_classes = p->num_classes;
    tm_param.side = p->side;
    tm_param.num_box = p->num_box;
    tm_param.coords = p->coords;
    tm_param.confidence_threshold = p->confidence_threshold;
    tm_param.nms_threshold = p->nms_threshold;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->biases_num;
    TM2_Vector_floats* v_biases = (TM2_Vector_floats*)malloc(vector_size);
    v_biases->v_num = p->biases_num;
    for (unsigned int i = 0; i < p->biases_num; i++)
    {
        v_biases->data[i] = p->biases[i];
    }
    tm_param.offset_vf_biases = WriteTmObject(start_ptr, cur_pos, v_biases, vector_size);
    free(v_biases);

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_REGION, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_RegionParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReLuOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct relu_param* p = (struct relu_param*)node->op.param_mem;
    TM2_ReLuParam tm_param;
    tm_param.negative_slope = p->negative_slope;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RELU, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReLuParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmRelu6Op(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RELU6, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReorgOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct reorg_param* p = (struct reorg_param*)node->op.param_mem;
    TM2_ReorgParam tm_param;
    tm_param.stride = p->stride;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_REORG, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReorgParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReshapeOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct reshape_param* p = (struct reshape_param*)node->op.param_mem;
    TM2_ReshapeParam tm_param;
    if (p->reverse)
        tm_param.reverse = 1;
    else
        tm_param.reverse = 0;
    if (p->is_mxnet)
        tm_param.is_mxnet = 1;
    else
        tm_param.is_mxnet = 0;
    if (p->is_onnx)
        tm_param.is_onnx = 1;
    else
        tm_param.is_onnx = 0;

    if (p->dim_size)
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * p->dim_size;
        TM2_Vector_dims* v_re_shape = (TM2_Vector_dims*)malloc(vector_size);
        v_re_shape->v_num = p->dim_size;
        for (unsigned int i = 0; i < p->dim_size; i++)
        {
            v_re_shape->dims[i] = p->re_shape[i];
        }
        tm_param.offset_re_shape = WriteTmObject(start_ptr, cur_pos, v_re_shape, vector_size);
        free(v_re_shape);
    }
    else
    {
        tm_param.offset_re_shape = TM2_NOT_SET;
    }

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RESHAPE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReshapeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmResizeOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct resize_param* p = (struct resize_param*)node->op.param_mem;
    TM2_ResizeParam tm_param;
    tm_param.scale_x = p->scale_w;
    tm_param.scale_y = p->scale_h;
    tm_param.type = p->type;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RESIZE,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ResizeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmROIPoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct roipooling_param* p = (struct roipooling_param*)node->op.param_mem;
    TM2_ROIPoolingParam tm_param;
    tm_param.pooled_h = p->pooled_h;
    tm_param.pooled_w = p->pooled_w;
    tm_param.spatial_scale = p->spatial_scale;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ROIPOOLING,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ROIPoolingParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmRPNOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct rpn_param* p = (struct rpn_param*)node->op.param_mem;
    TM2_RPNParam tm_param;

    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * p->ratios->elem_num;
    TM2_Vector_floats* v_ratios = (TM2_Vector_floats*)malloc(vector_size);
    v_ratios->v_num = p->ratios->elem_num;
    for (unsigned int i = 0; i < p->ratios->elem_num; i++)
    {
        v_ratios->data[i] = *(float*)get_vector_data(p->ratios, i);
    }
    tm_param.offset_vf_ratios = WriteTmObject(start_ptr, cur_pos, v_ratios, vector_size);
    free(v_ratios);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->anchor_scales->elem_num;
    TM2_Vector_floats* v_scales = (TM2_Vector_floats*)malloc(vector_size);
    v_scales->v_num = p->anchor_scales->elem_num;
    for (unsigned int i = 0; i < p->anchor_scales->elem_num; i++)
    {
        v_scales->data[i] = *(float*)get_vector_data(p->anchor_scales, i);
    }
    tm_param.offset_vf_anchor_scales = WriteTmObject(start_ptr, cur_pos, v_scales, vector_size);
    free(v_scales);

    vector_size = sizeof(tm_size_t) + sizeof(float) * p->anchors_->elem_num * 4;
    TM2_Vector_anchors* v_anchors = (TM2_Vector_anchors*)malloc(vector_size);
    v_anchors->v_num = p->anchors_->elem_num;
    for (unsigned int i = 0; i < p->anchors_->elem_num; i++)
    {
        v_anchors->data[i][0] = ((Anchor_t*)get_vector_data(p->anchors_, i))->x0;
        v_anchors->data[i][1] = ((Anchor_t*)get_vector_data(p->anchors_, i))->y0;
        v_anchors->data[i][2] = ((Anchor_t*)get_vector_data(p->anchors_, i))->x1;
        v_anchors->data[i][3] = ((Anchor_t*)get_vector_data(p->anchors_, i))->y1;
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

tm_uoffset_t SaveTmScaleOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct scale_param* p = (struct scale_param*)node->op.param_mem;
    TM2_ScaleParam tm_param;
    tm_param.axis = p->axis;
    tm_param.num_axes = p->num_axes;
    tm_param.bias_term = p->bias_term;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SCALE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ScaleParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSliceOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct slice_param* p = (struct slice_param*)node->op.param_mem;
    TM2_SliceParam tm_param;

    tm_param.axis = p->axis;
    tm_param.begin = p->begin;
    tm_param.end = p->end;
    tm_param.step = p->step;
    tm_param.iscaffe = p->iscaffe;
    tm_param.isonnx = p->isonnx;
    tm_param.ismxnet = p->ismxnet;

    if (p->slice_point_ && p->slice_point_->elem_num)
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * p->slice_point_->elem_num;
        TM2_Vector_dims* v_slice_points = (TM2_Vector_dims*)malloc(vector_size);
        v_slice_points->v_num = p->slice_point_->elem_num;
        for (unsigned int i = 0; i < p->slice_point_->elem_num; i++)
        {
            v_slice_points->dims[i] = *(int32_t*)get_vector_data(p->slice_point_, i);
        }
        tm_param.offset_vi_slice_points = WriteTmObject(start_ptr, cur_pos, v_slice_points, vector_size);
        free(v_slice_points);
    }
    else
        tm_param.offset_vi_slice_points = TM2_NOT_SET;

    if (p->begin_ && p->begin_->elem_num)
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * p->begin_->elem_num;
        TM2_Vector_dims* v_begins = (TM2_Vector_dims*)malloc(vector_size);
        v_begins->v_num = p->begin_->elem_num;
        for (unsigned int i = 0; i < p->begin_->elem_num; i++)
        {
            v_begins->dims[i] = *(int32_t*)get_vector_data(p->begin_, i);
        }
        tm_param.offset_vi_begins = WriteTmObject(start_ptr, cur_pos, v_begins, vector_size);
        free(v_begins);
    }
    else
        tm_param.offset_vi_begins = TM2_NOT_SET;

    if (p->size_ && p->size_->elem_num)
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * p->size_->elem_num;
        TM2_Vector_dims* v_sizes = (TM2_Vector_dims*)malloc(vector_size);
        v_sizes->v_num = p->size_->elem_num;
        for (unsigned int i = 0; i < p->size_->elem_num; i++)
        {
            v_sizes->dims[i] = *(int32_t*)get_vector_data(p->size_, i);
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

tm_uoffset_t SaveTmSoftmaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct softmax_param* p = (struct softmax_param*)node->op.param_mem;
    TM2_SoftmaxParam tm_param;
    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SOFTMAX, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SoftmaxParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSplitOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct split_param* p = (struct split_param*)node->op.param_mem;
    TM2_SplitParam tm_param;
    if (p->is_caffe)
        tm_param.is_caffe = 1;
    else
        tm_param.is_caffe = 0;

    if (p->is_onnx)
    {
        tm_param.is_onnx = 1;
    }
    else
    {
        tm_param.is_onnx = 0;
    }
    if (!p->is_caffe)
    {
        if (p->is_onnx)
            tm_param.axis = p->axis;
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * p->split_sizes_->elem_num;
        TM2_Vector_dims* v_split_sizes = (TM2_Vector_dims*)malloc(vector_size);
        v_split_sizes->v_num = p->split_sizes_->elem_num;
        for (unsigned int i = 0; i < p->split_sizes_->elem_num; i++)
        {
            v_split_sizes->dims[i] = *(int32_t*)get_vector_data(p->split_sizes_, i);
        }
        tm_param.offset_split_sizes = WriteTmObject(start_ptr, cur_pos, v_split_sizes, vector_size);
        free(v_split_sizes);
        tm_param.split_dim = p->split_dim;
    }

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SPLIT, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SplitParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmDetectionPostProcessOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct detection_postprocess_param* p = (struct detection_postprocess_param*)node->op.param_mem;
    TM2_DetectionPostProcessParam tm_param;

    tm_param.max_detections = p->max_detections;
    tm_param.max_classes_per_detection = p->max_classes_per_detection;
    tm_param.nms_score_threshold = p->nms_score_threshold;
    tm_param.nms_iou_threshold = p->nms_iou_threshold;
    tm_param.num_classes = p->num_classes;

    int param_scales_num = 4;
    size_t vector_size = sizeof(tm_size_t) + sizeof(float) * param_scales_num;
    TM2_Vector_floats* v_scales = (TM2_Vector_floats*)malloc(vector_size);
    v_scales->v_num = param_scales_num;
    for (unsigned int i = 0; i < param_scales_num; i++)
    {
        v_scales->data[i] = p->scales[i];
    }
    tm_param.offset_vf_scales = WriteTmObject(start_ptr, cur_pos, v_scales, vector_size);

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_DETECTIONPOSTPROCESS,
                  WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_DetectionPostProcessParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmGemmOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct gemm_param* p = (struct gemm_param*)node->op.param_mem;
    TM2_GemmParam tm_param;

    tm_param.alpha = p->alpha;
    tm_param.beta = p->beta;
    tm_param.transA = p->transA;
    tm_param.transB = p->transB;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_GEMM, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_GemmParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmLogisticOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_LOGISTIC, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmLstmOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct lstm_param* p = (struct lstm_param*)node->op.param_mem;
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
    tm_param.mxnet_flag = p->mxnet_flag;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_LSTM, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_LstmParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmRnnOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct rnn_param* p = (struct rnn_param*)node->op.param_mem;
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

tm_uoffset_t SaveTmTanhOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_TANH, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSigmoidOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SIGMOID, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmMaximumOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MAX, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmMinimumOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MIN, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSqueezeOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct squeeze_param* p = (struct squeeze_param*)node->op.param_mem;
    TM2_SqueezeParam tm_param;

    tm_param.dim_0 = p->dim_0;
    tm_param.dim_1 = p->dim_1;
    tm_param.dim_2 = p->dim_2;
    tm_param.dim_3 = p->dim_3;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SQUEEZE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SqueezeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmArgMaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct argmax_param* p = (struct argmax_param*)node->op.param_mem;
    TM2_ArgMaxParam tm_param;

    tm_param.axis = p->axis;
    tm_param.keepdims = p->keepdims;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ARGMAX, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ArgMaxParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmArgMinOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct argmin_param* p = (struct argmin_param*)node->op.param_mem;
    TM2_ArgMinParam tm_param;

    tm_param.axis = p->axis;
    tm_param.keepdims = p->keepdims;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ARGMIN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ArgMinParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmTopKV2Op(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct topkv2_param* p = (struct topkv2_param*)node->op.param_mem;
    TM2_TopKV2Param tm_param;

    tm_param.k = p->k;
    if (p->sorted)
        tm_param.sorted = 1;
    else
        tm_param.sorted = 0;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_TOPKV2, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_TopKV2Param)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmStridedSliceOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct strided_slice_param* p = (struct strided_slice_param*)node->op.param_mem;
    TM2_StridedSliceParam tm_param;

    tm_param.begin_n = p->begin[0];
    tm_param.begin_c = p->begin[1];
    tm_param.begin_h = p->begin[2];
    tm_param.begin_w = p->begin[3];
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

tm_uoffset_t SaveTmPadOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct pad_param* p = (struct pad_param*)node->op.param_mem;
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

tm_uoffset_t SaveTmReductionOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct reduction_param* p = (struct reduction_param*)node->op.param_mem;
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

tm_uoffset_t SaveTmSwapAxisOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct swap_axis_param* p = (struct swap_axis_param*)node->op.param_mem;
    TM2_SwapAxisParam tm_param;

    tm_param.dim_0 = p->dim_0;
    tm_param.dim_1 = p->dim_1;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SWAPAXIS, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SwapAxisParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmGruOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct gru_param* p = (struct gru_param*)node->op.param_mem;
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

tm_uoffset_t SaveTmUpsampleOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct upsample_param* p = (struct upsample_param*)node->op.param_mem;
    TM2_UpsampleParam tm_param;

    tm_param.scale = p->scale;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_UPSAMPLE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_UpsampleParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmShuffleChannelOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct shuffle_channel_param* p = (struct shuffle_channel_param*)node->op.param_mem;
    TM2_ShuffleChannelParam tm_param;

    tm_param.group = p->group;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SHUFFLECHANNEL, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ShuffleChannelParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSpaceToBatchNDOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct spacetobatchnd_param* p = (struct spacetobatchnd_param*)node->op.param_mem;
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

tm_uoffset_t SaveTmBatchToSpaceNDOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct batchtospacend_param* p = (struct batchtospacend_param*)node->op.param_mem;
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
tm_uoffset_t SaveTmCropOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct crop_param* p = (struct crop_param*)node->op.param_mem;
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

tm_uoffset_t SaveTmUnaryOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct unary_param* p = (struct unary_param*)node->op.param_mem;
    TM2_UnaryParam tm_param;

    tm_param.type = p->type;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_UNARY, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_UnaryParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmPsroipoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct psroipooling_param* p = (struct psroipooling_param*)node->op.param_mem;
    TM2_PsroipoolingParam tm_param;

    tm_param.spatial_scale = p->spatial_scale;
    tm_param.pooled_w = p->pooled_w;
    tm_param.pooled_h = p->pooled_h;
    tm_param.output_dim = p->output_dim;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_PSROIPOOLING, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_PsroipoolingParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmExpanddimsOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct expanddims_param* p = (struct expanddims_param*)node->op.param_mem;
    TM2_ExpanddimsParam tm_param;

    tm_param.axis = p->axis;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_EXPANDDIMS, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ExpanddimsParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmRoialignOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct roialign_param* p = (struct roialign_param*)node->op.param_mem;
    TM2_RoialignParam tm_param;

    tm_param.spatial_scale = p->spatial_scale;
    tm_param.pooled_width = p->pooled_width;
    tm_param.pooled_height = p->pooled_height;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ROIALIGN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_RoialignParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmThresholdOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct threshold_param* p = (struct threshold_param*)node->op.param_mem;
    TM2_ThresholdParam tm_param;

    tm_param.threshold = p->threshold;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_THRESHOLD, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ThresholdParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmNoopOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_NOOP, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmEmbedOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct embedding_param* p = (struct embedding_param*)node->op.param_mem;
    TM2_EmbedParam tm_param;

    //tm_param.bias_term = p->bias_term;
    tm_param.input_dim = p->input_dim;
    tm_param.num_output = p->num_output;
    tm_param.weight_data_size = p->weight_data_size;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_EMBED, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_EmbedParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmHardsigmoidOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct hard_sigmoid_param* p = (struct hard_sigmoid_param*)node->op.param_mem;
    TM2_HardsigmoidParam tm_param;

    tm_param.alpha = p->alpha;
    tm_param.beta = p->beta;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_HARDSIGMOID, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_HardsigmoidParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmInstanceNormOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct instancenorm_Param* p = (struct instancenorm_Param*)node->op.param_mem;
    TM2_InstanceNormParam tm_param;
    tm_param.eps = p->eps;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_INSTANCENORM, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_InstanceNormParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmMVNOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct mvn_param* p = (struct mvn_param*)node->op.param_mem;
    TM2_MVNParam tm_param;

    tm_param.across_channels = p->across_channels;
    tm_param.eps = p->eps;
    tm_param.normalize_variance = p->normalize_variance;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MVN, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_MVNParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmCastOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct cast_param* p = (struct cast_param*)node->op.param_mem;
    TM2_CastParam tm_param;
    tm_param.type_from = p->type_from;
    tm_param.type_to = p->type_to;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CAST, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_CastParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmHardSwishOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct hardswish_param* p = (struct hardswish_param*)node->op.param_mem;
    TM2_HardSwishParam tm_param;
    tm_param.alpha = p->alpha;
    tm_param.beta = p->beta;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_HARDSWISH, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_HardSwishParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmInterpOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct interp_param* p = (struct interp_param*)node->op.param_mem;
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
tm_uoffset_t SaveTmSeluOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct selu_param* p = (struct selu_param*)node->op.param_mem;
    TM2_SeluParam tm_param;
    tm_param.alpha = p->alpha;
    tm_param.lambda = p->lambda;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SELU, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SeluParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmEluOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct elu_param* p = (struct elu_param*)node->op.param_mem;
    TM2_EluParam tm_param;
    tm_param.alpha = p->alpha;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ELU, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_EluParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmBroadMulOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_BROADMUL, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmLogicalOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct logical_param* p = (struct logical_param*)node->op.param_mem;
    TM2_LogicalParam tm_param;

    tm_param.type = p->type;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_LOGICAL, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_LogicalParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmGatherOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct gather_param* p = (struct gather_param*)node->op.param_mem;
    TM2_GatherParam tm_param;

    tm_param.axis = p->axis;
    tm_param.indices_num = p->indices_num;
    tm_param.is_onnx = p->is_onnx;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_GATHER, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_GatherParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmTransposeOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct transpose_param* p = (struct transpose_param*)node->op.param_mem;
    TM2_TransposeParam tm_param;
    if (p->tr_shape_size)
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * p->tr_shape_size;
        TM2_Vector_dims* v_re_shape = (TM2_Vector_dims*)malloc(vector_size);
        v_re_shape->v_num = p->tr_shape_size;
        for (unsigned int i = 0; i < p->tr_shape_size; i++)
        {
            v_re_shape->dims[i] = p->tr_shape[i];
        }
        tm_param.offset_tr_shape = WriteTmObject(start_ptr, cur_pos, v_re_shape, vector_size);
        free(v_re_shape);
    }
    else
    {
        tm_param.offset_tr_shape = TM2_NOT_SET;
    }
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_TRANSPOSE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_TransposeParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmComparisonOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct comparison_param* p = (struct comparison_param*)node->op.param_mem;
    TM2_ComparisonParam tm_param;

    tm_param.type = p->type;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_COMPARISON, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ComparisonParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmReverseOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_REVERSE, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmSpaceToDepthOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct spacetodepth_param* p = (struct spacetodepth_param*)node->op.param_mem;
    TM2_SpaceToDepthParam tm_param;

    tm_param.block_size = p->block_size;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SPACETODEPTH, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SpaceToDepthParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmDepthToSpaceOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct depthtospace_param* p = (struct depthtospace_param*)node->op.param_mem;
    TM2_DepthToSpaceParam tm_param;

    tm_param.block_size = p->block_size;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_DEPTHTOSPACE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_DepthToSpaceParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmSquaredDifferenceOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SQUAREDDIFFERENCE, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmSparseToDenseOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct sparsetodense_param* p = (struct sparsetodense_param*)node->op.param_mem;
    TM2_SparseToDenseParam tm_param;

    tm_param.output_shape_size0 = p->output_shape_size0;
    tm_param.output_shape_size1 = p->output_shape_size1;
    tm_param.default_value = p->default_value;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SPARSETODENSE, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SparseToDenseParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmCeilOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CEIL, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmRoundOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ROUND, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmZerosLikeOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_ZEROSLIKE, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmClipOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct clip_param* p = (struct clip_param*)node->op.param_mem;
    TM2_ClipParam tm_param;

    tm_param.max = p->max;
    tm_param.min = p->min;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_CLIP, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ClipParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}
tm_uoffset_t SaveTmUnsqueezeOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct unsqueeze_param* p = (struct unsqueeze_param*)node->op.param_mem;
    TM2_UnsqueezeParam tm_param;

    if (p->axises_size)
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * p->axises_size;
        TM2_Vector_dims* v_axises = (TM2_Vector_dims*)malloc(vector_size);
        v_axises->v_num = p->axises_size;
        for (unsigned int i = 0; i < p->axises_size; i++)
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

tm_uoffset_t SaveTmReduceL2Op(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct reducel2_param* p = (struct reducel2_param*)node->op.param_mem;
    TM2_ReduceL2Param tm_param;

    tm_param.axis = p->axis;
    tm_param.keepdim = p->keepdim;

    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_REDUCEL2, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ReduceL2Param)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmMeanOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MEAN, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmMatMulOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_MATMUL, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmExpandOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct expand_param* p = (struct expand_param*)node->op.param_mem;
    TM2_ExpandParam tm_param;
    memset(&tm_param, 0, sizeof(TM2_ExpandParam));

    if (p->dim_num)
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * p->dim_num;
        TM2_Vector_dims* v_axises = (TM2_Vector_dims*)malloc(vector_size);
        v_axises->v_num = p->dim_num;
        for (unsigned int i = 0; i < p->dim_num; i++)
        {
            v_axises->dims[i] = p->ex_shape[i];
        }
        tm_param.offset_ex_shape = WriteTmObject(start_ptr, cur_pos, v_axises, vector_size);
        free(v_axises);
    }
    else
        tm_param.offset_ex_shape = TM2_NOT_SET;

    tm_param.dim_num = p->dim_num;
    TM2_Operator tm_op;
    memset(&tm_op, 0, sizeof(TM2_Operator));
    SetTmOperator(&tm_op, TM2_OPTYPE_EXPAND, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_ExpandParam)));

    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSpatialTransformerOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct spatialtransformer_param* p = (struct spatialtransformer_param*)node->op.param_mem;
    TM2_SpatialTransformerParam tm_param;
    memset(&tm_param, 0, sizeof(TM2_SpatialTransformerParam));
    tm_param.sampler_type = p->sampler_type;
    tm_param.transformer_type = p->transformer_type;
    tm_param.shape_size = sizeof(p->target_shape) / sizeof(p->target_shape[0]);
    if (tm_param.shape_size)
    {
        size_t vector_size = sizeof(tm_size_t) + sizeof(int32_t) * tm_param.shape_size;
        TM2_Vector_dims* v_ta_shape = (TM2_Vector_dims*)malloc(vector_size);
        v_ta_shape->v_num = tm_param.shape_size;
        for (unsigned int i = 0; i < tm_param.shape_size; i++)
        {
            v_ta_shape->dims[i] = p->target_shape[i];
        }
        tm_param.offset_ta_shape = WriteTmObject(start_ptr, cur_pos, v_ta_shape, vector_size);
        free(v_ta_shape);
    }
    else
    {
        tm_param.offset_ta_shape = TM2_NOT_SET;
    }

    TM2_Operator tm_op;
    memset(&tm_op, 0, sizeof(TM2_Operator));
    SetTmOperator(&tm_op, TM2_OPTYPE_SPATIALTRANSFORMER, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_SpatialTransformerParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmSoftplusOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_SOFTPLUS, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmReciprocalOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_RECIPROCAL, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmGeluOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_GELU, TM2_NOT_SET);
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

tm_uoffset_t SaveTmLayerNormOp(void* const start_ptr, tm_uoffset_t* cur_pos, ir_node_t* node)
{
    struct layernorm_Param* p = (struct layernorm_Param*)node->op.param_mem;
    TM2_LayerNormParam tm_param;
    tm_param.eps = p->eps;
    TM2_Operator tm_op;
    SetTmOperator(&tm_op, TM2_OPTYPE_LAYERNORM, WriteTmObject(start_ptr, cur_pos, &tm_param, sizeof(TM2_LayerNormParam)));
    return WriteTmObject(start_ptr, cur_pos, &tm_op, sizeof(TM2_Operator));
}

op_save_t SaveTmOpFunc(uint32_t op_type)
{
    switch (op_type)
    {
    case OP_BATCHNORM:
        return SaveTmBatchNormOp;
    case OP_CONCAT:
        return SaveTmConcatOp;
    case OP_CONST:
        return SaveTmConstOp;
    case OP_CONV:
        return SaveTmConvOp;
    case OP_DECONV:
        return SaveTmDeconvOp;
    case OP_DETECTION_OUTPUT:
        return SaveTmDetectionOutputOp;
    case OP_DROPOUT:
        return SaveTmDropoutOp;
    case OP_ELTWISE:
        return SaveTmEltwiseOp;
    case OP_FLATTEN:
        return SaveTmFlattenOp;
    case OP_FC:
        return SaveTmFCOp;
    case OP_INPUT:
        return SaveTmInputOp;
    case OP_LRN:
        return SaveTmLRNOp;
    case OP_NORMALIZE:
        return SaveTmNormalizeOp;
    case OP_PERMUTE:
        return SaveTmPermuteOp;
    case OP_POOL:
        return SaveTmPoolingOp;
    case OP_PRELU:
        return SaveTmPreluOp;
    case OP_PRIORBOX:
        return SaveTmPriorBoxOp;
    case OP_REGION:
        return SaveTmRegionOp;
    case OP_RELU:
        return SaveTmReLuOp;
    case OP_RELU6:
        return SaveTmRelu6Op;
    case OP_REORG:
        return SaveTmReorgOp;
    case OP_RESHAPE:
        return SaveTmReshapeOp;
    case OP_ROIPOOLING:
        return SaveTmROIPoolingOp;
    case OP_RPN:
        return SaveTmRPNOp;
    case OP_SCALE:
        return SaveTmScaleOp;
    case OP_SLICE:
        return SaveTmSliceOp;
    case OP_SOFTMAX:
        return SaveTmSoftmaxOp;
    case OP_SPLIT:
        return SaveTmSplitOp;
    case OP_DETECTION_POSTPROCESS:
        return SaveTmDetectionPostProcessOp;
    case OP_GEMM:
        return SaveTmGemmOp;
    case OP_LOGISTIC:
        return SaveTmLogisticOp;
    case OP_LSTM:
        return SaveTmLstmOp;
    case OP_RNN:
        return SaveTmRnnOp;
    case OP_TANH:
        return SaveTmTanhOp;
    case OP_SIGMOID:
        return SaveTmSigmoidOp;
    case OP_SQUEEZE:
        return SaveTmSqueezeOp;
    case OP_SWAP_AXIS:
        return SaveTmSwapAxisOp;
    case OP_GRU:
        return SaveTmGruOp;
    case OP_ARGMAX:
        return SaveTmArgMaxOp;
    case OP_ARGMIN:
        return SaveTmArgMinOp;
    case OP_TOPKV2:
        return SaveTmTopKV2Op;
    case OP_PAD:
        return SaveTmPadOp;
    case OP_STRIDED_SLICE:
        return SaveTmStridedSliceOp;
    case OP_REDUCTION:
        return SaveTmReductionOp;
    case OP_UPSAMPLE:
        return SaveTmUpsampleOp;
    case OP_SHUFFLECHANNEL:
        return SaveTmShuffleChannelOp;
    case OP_SPACETOBATCHND:
        return SaveTmSpaceToBatchNDOp;
    case OP_BATCHTOSPACEND:
        return SaveTmBatchToSpaceNDOp;
    case OP_RESIZE:
        return SaveTmResizeOp;
    case OP_CROP:
        return SaveTmCropOp;
    case OP_ROIALIGN:
        return SaveTmRoialignOp;
    case OP_PSROIPOOLING:
        return SaveTmPsroipoolingOp;
    case OP_EXPANDDIMS:
        return SaveTmExpanddimsOp;
    case OP_UNARY:
        return SaveTmUnaryOp;
    case OP_NOOP:
        return SaveTmNoopOp;
    case OP_THRESHOLD:
        return SaveTmThresholdOp;
    case OP_HARDSIGMOID:
        return SaveTmHardsigmoidOp;
    case OP_EMBEDDING:
        return SaveTmEmbedOp;
    case OP_INSTANCENORM:
        return SaveTmInstanceNormOp;
    case OP_MVN:
        return SaveTmMVNOp;
    case OP_CAST:
        return SaveTmCastOp;
    case OP_HARDSWISH:
        return SaveTmHardSwishOp;
    case OP_INTERP:
        return SaveTmInterpOp;
    case OP_SELU:
        return SaveTmSeluOp;
    case OP_ELU:
        return SaveTmEluOp;
    case OP_BROADMUL:
        return SaveTmBroadMulOp;
    case OP_LOGICAL:
        return SaveTmLogicalOp;
    case OP_GATHER:
        return SaveTmGatherOp;
    case OP_TRANSPOSE:
        return SaveTmTransposeOp;
    case OP_COMPARISON:
        return SaveTmComparisonOp;
    case OP_REVERSE:
        return SaveTmReverseOp;
    case OP_SPACETODEPTH:
        return SaveTmSpaceToDepthOp;
    case OP_DEPTHTOSPACE:
        return SaveTmDepthToSpaceOp;
    case OP_SQUAREDDIFFERENCE:
        return SaveTmSquaredDifferenceOp;
    case OP_SPARSETODENSE:
        return SaveTmSparseToDenseOp;
    case OP_CEIL:
        return SaveTmCeilOp;
    case OP_ROUND:
        return SaveTmRoundOp;
    case OP_ZEROSLIKE:
        return SaveTmZerosLikeOp;
    case OP_CLIP:
        return SaveTmClipOp;
    case OP_REDUCEL2:
        return SaveTmReduceL2Op;
    case OP_UNSQUEEZE:
        return SaveTmUnsqueezeOp;
    case OP_MEAN:
        return SaveTmMeanOp;
    case OP_MATMUL:
        return SaveTmMatMulOp;
    case OP_MISH:
        return SaveTmMishOp;
    case OP_SPATIALTRANSFORMER:
        return SaveTmSpatialTransformerOp;
    case OP_EXPAND:
        return SaveTmExpandOp;
    case OP_SOFTPLUS:
        return SaveTmSoftplusOp;
    case OP_RECIPROCAL:
        return SaveTmReciprocalOp;
    case OP_MAXIMUM:
        return SaveTmMaximumOp;
    case OP_MINIMUM:
        return SaveTmMinimumOp;
    case OP_GELU:
        return SaveTmGeluOp;
    case OP_LAYERNORM:
        return SaveTmLayerNormOp;
    default:
        // fprintf(stderr, "Operator #%d not supported in tengine model yet\n", op_type);
        return nullptr;
    }
}
