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

bool LoadTmAccuracyOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_ACCURACY);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmBatchNormOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_BATCHNORMALIZATION;

    BatchNormParam param = any_cast<BatchNormParam>(OpManager::GetOpDefParam(op_str));
    const TM2_BatchNormParam* tm_param = GetTmPtr<TM2_BatchNormParam>(start_ptr, tm_op->offset_t_param);

    param.rescale_factor = tm_param->rescale_factor;
    param.eps = tm_param->eps;
    param.caffe_flavor = tm_param->caffe_flavor;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmResizeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_BILINEARRESIZE;

    ResizeParam param = any_cast<ResizeParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ResizeParam* tm_param = GetTmPtr<TM2_ResizeParam>(start_ptr, tm_op->offset_t_param);

    param.scale_w = tm_param->scale_x;
    param.scale_h = tm_param->scale_y;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmConcatOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_CONCAT;

    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ConcatParam* tm_param = GetTmPtr<TM2_ConcatParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmConstOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_CONST);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmConvOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_CONVOLUTION;

    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ConvParam* tm_param = GetTmPtr<TM2_ConvParam>(start_ptr, tm_op->offset_t_param);

    param.kernel_h = tm_param->kernel_h;
    param.kernel_w = tm_param->kernel_w;
    param.stride_h = tm_param->stride_h;
    param.stride_w = tm_param->stride_w;
    param.dilation_h = tm_param->dilation_h;
    param.dilation_w = tm_param->dilation_w;
    param.input_channel = tm_param->input_channel;
    param.output_channel = tm_param->output_channel;
    param.group = tm_param->group;
    param.activation = tm_param->activation;
    param.pad_h0 = tm_param->pad_h0;
    param.pad_h1 = tm_param->pad_h1;
    param.pad_w0 = tm_param->pad_w0;
    param.pad_w1 = tm_param->pad_w1;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmDeconvOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_DECONVOLUTION;

    DeconvParam param = any_cast<DeconvParam>(OpManager::GetOpDefParam(op_str));
    const TM2_DeconvParam* tm_param = GetTmPtr<TM2_DeconvParam>(start_ptr, tm_op->offset_t_param);

    param.kernel_h = tm_param->kernel_h;
    param.kernel_w = tm_param->kernel_w;
    param.stride_h = tm_param->stride_h;
    param.stride_w = tm_param->stride_w;
    param.pad_w0 = tm_param->pad_w0;
    param.pad_w1 = tm_param->pad_w1;
    param.pad_h0 = tm_param->pad_h0;
    param.pad_h1 = tm_param->pad_h1;
    param.num_output = tm_param->num_output;
    param.dilation_h = tm_param->dilation_h;
    param.dilation_w = tm_param->dilation_w;
    param.group = tm_param->group;
    param.activation = tm_param->activation;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmDetectionOutputOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_DETECTIONOUTPUT;

    DetectionOutputParam param = any_cast<DetectionOutputParam>(OpManager::GetOpDefParam(op_str));
    const TM2_DetectionOutputParam* tm_param = GetTmPtr<TM2_DetectionOutputParam>(start_ptr, tm_op->offset_t_param);

    param.num_classes = tm_param->num_classes;
    param.keep_top_k = tm_param->keep_top_k;
    param.nms_top_k = tm_param->nms_top_k;
    param.confidence_threshold = tm_param->confidence_threshold;
    param.nms_threshold = tm_param->nms_threshold;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmDropoutOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_DROPOUT);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmEltwiseOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_ELTWISE;

    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam(op_str));
    const TM2_EltwiseParam* tm_param = GetTmPtr<TM2_EltwiseParam>(start_ptr, tm_op->offset_t_param);

    param.type = static_cast<EltType>(tm_param->type);
    param.caffe_flavor = tm_param->caffe_flavor;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmFlattenOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_FLATTEN;

    FlattenParam param = any_cast<FlattenParam>(OpManager::GetOpDefParam(op_str));
    const TM2_FlattenParam* tm_param = GetTmPtr<TM2_FlattenParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;
    param.end_axis = tm_param->end_axis;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmFCOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_FULLYCONNECTED;

    FCParam param = any_cast<FCParam>(OpManager::GetOpDefParam(op_str));
    const TM2_FCParam* tm_param = GetTmPtr<TM2_FCParam>(start_ptr, tm_op->offset_t_param);

    param.num_output = tm_param->num_output;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmInputOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_INPUTOP);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmLRNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_LRN;

    LRNParam param = any_cast<LRNParam>(OpManager::GetOpDefParam(op_str));
    const TM2_LRNParam* tm_param = GetTmPtr<TM2_LRNParam>(start_ptr, tm_op->offset_t_param);

    param.local_size = tm_param->local_size;
    param.alpha = tm_param->alpha;
    param.beta = tm_param->beta;
    param.norm_region = tm_param->norm_region;
    param.k = tm_param->k;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmNormalizeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_NORMALIZE;

    NormalizeParam param = any_cast<NormalizeParam>(OpManager::GetOpDefParam(op_str));
    const TM2_NormalizeParam* tm_param = GetTmPtr<TM2_NormalizeParam>(start_ptr, tm_op->offset_t_param);

    param.across_spatial = tm_param->across_spatial;
    param.channel_shared = tm_param->channel_shared;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmPermuteOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_PERMUTE;

    PermuteParam param = any_cast<PermuteParam>(OpManager::GetOpDefParam(op_str));
    const TM2_PermuteParam* tm_param = GetTmPtr<TM2_PermuteParam>(start_ptr, tm_op->offset_t_param);

    param.flag = tm_param->flag;
    param.order0 = tm_param->order0;
    param.order1 = tm_param->order1;
    param.order2 = tm_param->order2;
    param.order3 = tm_param->order3;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmPoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_POOLING;

    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam(op_str));
    const TM2_PoolParam* tm_param = GetTmPtr<TM2_PoolParam>(start_ptr, tm_op->offset_t_param);

    param.alg = static_cast<PoolArg>(tm_param->alg);
    param.kernel_h = tm_param->kernel_h;
    param.kernel_w = tm_param->kernel_w;
    param.stride_h = tm_param->stride_h;
    param.stride_w = tm_param->stride_w;
    param.global = tm_param->global;
    param.caffe_flavor = tm_param->caffe_flavor;
    param.pad_h0 = tm_param->pad_h0;
    param.pad_w0 = tm_param->pad_w0;
    param.pad_h1 = tm_param->pad_h1;
    param.pad_w1 = tm_param->pad_w1;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmPreluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_PRELU);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmPriorBoxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_PRIORBOX;

    PriorBoxParam param = any_cast<PriorBoxParam>(OpManager::GetOpDefParam(op_str));
    const TM2_PriorBoxParam* tm_param = GetTmPtr<TM2_PriorBoxParam>(start_ptr, tm_op->offset_t_param);
    const TM2_Vector_floats* v_minsizes = GetTmPtr<TM2_Vector_floats>(start_ptr, tm_param->offset_vf_min_size);
    const TM2_Vector_floats* v_maxsizes = GetTmPtr<TM2_Vector_floats>(start_ptr, tm_param->offset_vf_max_size);
    const TM2_Vector_floats* v_variances = GetTmPtr<TM2_Vector_floats>(start_ptr, tm_param->offset_vf_variance);
    const TM2_Vector_floats* v_ratios = GetTmPtr<TM2_Vector_floats>(start_ptr, tm_param->offset_vf_aspect_ratio);

    for(unsigned int i = 0; i < v_minsizes->v_num; i++)
        param.min_size.push_back(v_minsizes->data[i]);
    for(unsigned int i = 0; i < v_maxsizes->v_num; i++)
        param.max_size.push_back(v_maxsizes->data[i]);
    for(unsigned int i = 0; i < v_variances->v_num; i++)
        param.variance.push_back(v_variances->data[i]);
    for(unsigned int i = 0; i < v_ratios->v_num; i++)
        param.aspect_ratio.push_back(v_ratios->data[i]);
    param.flip = tm_param->flip;
    param.clip = tm_param->clip;
    param.img_size = tm_param->img_size;
    param.img_h = tm_param->img_h;
    param.img_w = tm_param->img_w;
    param.step_w = tm_param->step_w;
    param.step_h = tm_param->step_h;
    param.offset = tm_param->offset;
    param.num_priors_ = tm_param->num_priors;
    param.out_dim_ = tm_param->out_dim;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmRegionOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_REGION;

    RegionParam param = any_cast<RegionParam>(OpManager::GetOpDefParam(op_str));
    const TM2_RegionParam* tm_param = GetTmPtr<TM2_RegionParam>(start_ptr, tm_op->offset_t_param);
    const TM2_Vector_floats* v_biases = GetTmPtr<TM2_Vector_floats>(start_ptr, tm_param->offset_vf_biases);

    for(unsigned int i = 0; i < v_biases->v_num; i++)
        param.biases.push_back(v_biases->data[i]);
    param.num_classes = tm_param->num_classes;
    param.side = tm_param->side;
    param.num_box = tm_param->num_box;
    param.coords = tm_param->coords;
    param.confidence_threshold = tm_param->confidence_threshold;
    param.nms_threshold = tm_param->nms_threshold;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmReLuOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_RELU;

    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ReLuParam* tm_param = GetTmPtr<TM2_ReLuParam>(start_ptr, tm_op->offset_t_param);

    param.negative_slope = tm_param->negative_slope;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmRelu6Op(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_RELU6);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmReorgOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_REORG;

    ReorgParam param = any_cast<ReorgParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ReorgParam* tm_param = GetTmPtr<TM2_ReorgParam>(start_ptr, tm_op->offset_t_param);

    param.stride = tm_param->stride;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmReshapeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_RESHAPE;

    ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ReshapeParam* tm_param = GetTmPtr<TM2_ReshapeParam>(start_ptr, tm_op->offset_t_param);

    param.dim_0 = tm_param->dim_0;
    param.dim_1 = tm_param->dim_1;
    param.dim_2 = tm_param->dim_2;
    param.dim_3 = tm_param->dim_3;
    param.dim_size = tm_param->dim_size;
    param.axis = tm_param->axis;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmROIPoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_ROIPOOLING;

    ROIPoolingParam param = any_cast<ROIPoolingParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ROIPoolingParam* tm_param = GetTmPtr<TM2_ROIPoolingParam>(start_ptr, tm_op->offset_t_param);

    param.pooled_h = tm_param->pooled_h;
    param.pooled_w = tm_param->pooled_w;
    param.spatial_scale = tm_param->spatial_scale;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmRPNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_RPN;

    RPNParam param = any_cast<RPNParam>(OpManager::GetOpDefParam(op_str));
    const TM2_RPNParam* tm_param = GetTmPtr<TM2_RPNParam>(start_ptr, tm_op->offset_t_param);
    const TM2_Vector_floats* v_ratios = GetTmPtr<TM2_Vector_floats>(start_ptr, tm_param->offset_vf_ratios);
    const TM2_Vector_floats* v_scales = GetTmPtr<TM2_Vector_floats>(start_ptr, tm_param->offset_vf_anchor_scales);

    for(unsigned int i = 0; i < v_ratios->v_num; i++)
        param.ratios.push_back(v_ratios->data[i]);
    for(unsigned int i = 0; i < v_scales->v_num; i++)
        param.anchor_scales.push_back(v_scales->data[i]);
    param.feat_stride = tm_param->feat_stride;
    param.basesize = tm_param->basesize;
    param.min_size = tm_param->min_size;
    param.per_nms_topn = tm_param->per_nms_topn;
    param.post_nms_topn = tm_param->post_nms_topn;
    param.nms_thresh = tm_param->nms_thresh;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmScaleOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SCALE;

    ScaleParam param = any_cast<ScaleParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ScaleParam* tm_param = GetTmPtr<TM2_ScaleParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;
    param.num_axes = tm_param->num_axes;
    param.bias_term = tm_param->bias_term;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSliceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SLICE;

    SliceParam param = any_cast<SliceParam>(OpManager::GetOpDefParam(op_str));
    const TM2_SliceParam* tm_param = GetTmPtr<TM2_SliceParam>(start_ptr, tm_op->offset_t_param);

    if(tm_param->offset_vi_slice_points != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_slice_points = GetTmPtr<TM2_Vector_dims>(start_ptr, tm_param->offset_vi_slice_points);
        for(unsigned int i = 0; i < v_slice_points->v_num; i++)
            param.slice_point_.push_back(v_slice_points->dims[i]);
    }
    if(tm_param->offset_vi_begins != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_begins = GetTmPtr<TM2_Vector_dims>(start_ptr, tm_param->offset_vi_begins);
        for(unsigned int i = 0; i < v_begins->v_num; i++)
            param.begin_.push_back(v_begins->dims[i]);
    }
    if(tm_param->offset_vi_sizes != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_sizes = GetTmPtr<TM2_Vector_dims>(start_ptr, tm_param->offset_vi_sizes);
        for(unsigned int i = 0; i < v_sizes->v_num; i++)
            param.size_.push_back(v_sizes->dims[i]);
    }

    param.axis = tm_param->axis;
    param.iscaffe = tm_param->iscaffe;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSoftmaxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SOFTMAX;

    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam(op_str));
    const TM2_SoftmaxParam* tm_param = GetTmPtr<TM2_SoftmaxParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSplitOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_SPLIT);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmDetectionPostProcessOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_DETECTIONPOSTPROCESS;

    DetectionPostProcessParam param = any_cast<DetectionPostProcessParam>(OpManager::GetOpDefParam(op_str));
    const TM2_DetectionPostProcessParam* tm_param = GetTmPtr<TM2_DetectionPostProcessParam>(start_ptr, tm_op->offset_t_param);

    param.max_detections = tm_param->max_detections;
    param.max_classes_per_detection = tm_param->max_classes_per_detection;
    param.nms_score_threshold = tm_param->nms_score_threshold;
    param.nms_iou_threshold = tm_param->nms_iou_threshold;
    param.num_classes = tm_param->num_classes;

    const TM2_Vector_floats* v_scales = GetTmPtr<TM2_Vector_floats>(start_ptr, tm_param->offset_vf_scales);

    for(unsigned int i = 0; i < v_scales->v_num; i++)
        param.scales.push_back(v_scales->data[i]);

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmGemmOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_GEMM;

    GemmParam param = any_cast<GemmParam>(OpManager::GetOpDefParam(op_str));
    const TM2_GemmParam* tm_param = GetTmPtr<TM2_GemmParam>(start_ptr, tm_op->offset_t_param);

    param.alpha = tm_param->alpha;
    param.beta = tm_param->beta;
    param.transA = tm_param->transA;
    param.transB = tm_param->transB;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmGenericOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_GENERIC;

    GenericParam param = any_cast<GenericParam>(OpManager::GetOpDefParam(op_str));
    const TM2_GenericParam* tm_param = GetTmPtr<TM2_GenericParam>(start_ptr, tm_op->offset_t_param);

    param.max_input_num = tm_param->max_input_num;
    param.max_output_num = tm_param->max_output_num;

    const TM2_String* tm_string = GetTmPtr<TM2_String>(start_ptr, tm_param->offset_s_opname);
    char *op_name = (char *)malloc(tm_string->size);
    memcpy(op_name, GetTmPtr<char>(start_ptr, tm_string->offset_data), tm_string->size);
    param.op_name = op_name;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);

    return true;
}

bool LoadTmLogisticOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_LOGISTIC);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmLstmOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_LSTM;

    LSTMParam param = any_cast<LSTMParam>(OpManager::GetOpDefParam(op_str));
    const TM2_LstmParam* tm_param = GetTmPtr<TM2_LstmParam>(start_ptr, tm_op->offset_t_param);

    param.forget_bias = tm_param->forget_bias;
    param.clip = tm_param->clip;
    param.output_len = tm_param->output_len;
    param.sequence_len = tm_param->sequence_len;
    param.input_size = tm_param->input_size;
    param.hidden_size = tm_param->hidden_size;
    param.cell_size = tm_param->cell_size;
    param.has_peephole = tm_param->has_peephole;
    param.has_projection = tm_param->has_projection;
    param.has_clip = tm_param->has_clip;
    param.has_bias = tm_param->has_bias;
    param.has_init_state = tm_param->has_init_state;
    param.forget_act = tm_param->forget_act;
    param.input_act = tm_param->input_act;
    param.output_act = tm_param->output_act;
    param.cellin_act = tm_param->cellin_act;
    param.cellout_act = tm_param->cellout_act;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmRnnOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_RNN;

    RNNParam param = any_cast<RNNParam>(OpManager::GetOpDefParam(op_str));
    const TM2_RnnParam* tm_param = GetTmPtr<TM2_RnnParam>(start_ptr, tm_op->offset_t_param);

    param.clip = tm_param->clip;
    param.output_len = tm_param->output_len;
    param.sequence_len = tm_param->sequence_len;
    param.input_size = tm_param->input_size;
    param.hidden_size = tm_param->hidden_size;
    param.has_clip = tm_param->has_clip;
    param.has_bias = tm_param->has_bias;
    param.has_init_state = tm_param->has_init_state;
    param.activation = tm_param->activation;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmTanhOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_TANH);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSigmoidOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_SIGMOID);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSqueezeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SQUEEZE;

    SqueezeParam param = any_cast<SqueezeParam>(OpManager::GetOpDefParam(op_str));
    const TM2_SqueezeParam* tm_param = GetTmPtr<TM2_SqueezeParam>(start_ptr, tm_op->offset_t_param);

    param.dim_0 = tm_param->dim_0;
    param.dim_1 = tm_param->dim_1;
    param.dim_2 = tm_param->dim_2;
    param.dim_3 = tm_param->dim_3;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmFusedbnscalereluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_FUSEDBNSCALERELU);
    SetNodeOp(node, op);
    return true;
}

op_load_t LoadTmOpFunc(uint32_t op_type)
{
    switch(op_type)
    {
        case TM2_OPTYPE_ACCURACY:
            return LoadTmAccuracyOp;
        case TM2_OPTYPE_BATCHNORMALIZATION:
            return LoadTmBatchNormOp;
        case TM2_OPTYPE_BILINEARRESIZE:
            return LoadTmResizeOp;
        case TM2_OPTYPE_CONCAT:
            return LoadTmConcatOp;
        case TM2_OPTYPE_CONST:
            return LoadTmConstOp;
        case TM2_OPTYPE_CONVOLUTION:
            return LoadTmConvOp;
        case TM2_OPTYPE_DECONVOLUTION:
            return LoadTmDeconvOp;
        case TM2_OPTYPE_DETECTIONOUTPUT:
            return LoadTmDetectionOutputOp;
        case TM2_OPTYPE_DROPOUT:
            return LoadTmDropoutOp;
        case TM2_OPTYPE_ELTWISE:
            return LoadTmEltwiseOp;
        case TM2_OPTYPE_FLATTEN:
            return LoadTmFlattenOp;
        case TM2_OPTYPE_FULLYCONNECTED:
            return LoadTmFCOp;
        case TM2_OPTYPE_INPUTOP:
            return LoadTmInputOp;
        case TM2_OPTYPE_LRN:
            return LoadTmLRNOp;
        case TM2_OPTYPE_NORMALIZE:
            return LoadTmNormalizeOp;
        case TM2_OPTYPE_PERMUTE:
            return LoadTmPermuteOp;
        case TM2_OPTYPE_POOLING:
            return LoadTmPoolingOp;
        case TM2_OPTYPE_PRELU:
            return LoadTmPreluOp;
        case TM2_OPTYPE_PRIORBOX:
            return LoadTmPriorBoxOp;
        case TM2_OPTYPE_REGION:
            return LoadTmRegionOp;
        case TM2_OPTYPE_RELU:
            return LoadTmReLuOp;
        case TM2_OPTYPE_RELU6:
            return LoadTmRelu6Op;
        case TM2_OPTYPE_REORG:
            return LoadTmReorgOp;
        case TM2_OPTYPE_RESHAPE:
            return LoadTmReshapeOp;
        case TM2_OPTYPE_ROIPOOLING:
            return LoadTmROIPoolingOp;
        case TM2_OPTYPE_RPN:
            return LoadTmRPNOp;
        case TM2_OPTYPE_SCALE:
            return LoadTmScaleOp;
        case TM2_OPTYPE_SLICE:
            return LoadTmSliceOp;
        case TM2_OPTYPE_SOFTMAX:
            return LoadTmSoftmaxOp;
        case TM2_OPTYPE_SPLIT:
            return LoadTmSplitOp;
        case TM2_OPTYPE_DETECTIONPOSTPROCESS:
            return LoadTmDetectionPostProcessOp;
        case TM2_OPTYPE_GEMM:
            return LoadTmGemmOp;
        case TM2_OPTYPE_GENERIC:
            return LoadTmGenericOp;
        case TM2_OPTYPE_LOGISTIC:
            return LoadTmLogisticOp;
        case TM2_OPTYPE_LSTM:
            return LoadTmLstmOp;
        case TM2_OPTYPE_RNN:
            return LoadTmRnnOp;
        case TM2_OPTYPE_TANH:
            return LoadTmTanhOp;
        case TM2_OPTYPE_SIGMOID:
            return LoadTmSigmoidOp;
        case TM2_OPTYPE_SQUEEZE:
            return LoadTmSqueezeOp;
        case TM2_OPTYPE_FUSEDBNSCALERELU:
            return LoadTmFusedbnscalereluOp;
        default:
            LOG_ERROR() << "Operator #" << op_type << " not supported in tengine model yet\n";
            return nullptr;
    }
}

std::string GetOpStr(uint32_t op_type)
{
    switch(op_type)
    {
        case TM2_OPTYPE_ACCURACY:
            return std::string(TM2_OPSTR_ACCURACY);
        case TM2_OPTYPE_BATCHNORMALIZATION:
            return std::string(TM2_OPSTR_BATCHNORMALIZATION);
        case TM2_OPTYPE_BILINEARRESIZE:
            return std::string(TM2_OPSTR_BILINEARRESIZE);
        case TM2_OPTYPE_CONCAT:
            return std::string(TM2_OPSTR_CONCAT);
        case TM2_OPTYPE_CONST:
            return std::string(TM2_OPSTR_CONST);
        case TM2_OPTYPE_CONVOLUTION:
            return std::string(TM2_OPSTR_CONVOLUTION);
        case TM2_OPTYPE_DECONVOLUTION:
            return std::string(TM2_OPSTR_DECONVOLUTION);
        case TM2_OPTYPE_DETECTIONOUTPUT:
            return std::string(TM2_OPSTR_DETECTIONOUTPUT);
        case TM2_OPTYPE_DROPOUT:
            return std::string(TM2_OPSTR_DROPOUT);
        case TM2_OPTYPE_ELTWISE:
            return std::string(TM2_OPSTR_ELTWISE);
        case TM2_OPTYPE_FLATTEN:
            return std::string(TM2_OPSTR_FLATTEN);
        case TM2_OPTYPE_FULLYCONNECTED:
            return std::string(TM2_OPSTR_FULLYCONNECTED);
        case TM2_OPTYPE_INPUTOP:
            return std::string(TM2_OPSTR_INPUTOP);
        case TM2_OPTYPE_LRN:
            return std::string(TM2_OPSTR_LRN);
        case TM2_OPTYPE_NORMALIZE:
            return std::string(TM2_OPSTR_NORMALIZE);
        case TM2_OPTYPE_PERMUTE:
            return std::string(TM2_OPSTR_PERMUTE);
        case TM2_OPTYPE_POOLING:
            return std::string(TM2_OPSTR_POOLING);
        case TM2_OPTYPE_PRELU:
            return std::string(TM2_OPSTR_PRELU);
        case TM2_OPTYPE_PRIORBOX:
            return std::string(TM2_OPSTR_PRIORBOX);
        case TM2_OPTYPE_REGION:
            return std::string(TM2_OPSTR_REGION);
        case TM2_OPTYPE_RELU:
            return std::string(TM2_OPSTR_RELU);
        case TM2_OPTYPE_RELU6:
            return std::string(TM2_OPSTR_RELU6);
        case TM2_OPTYPE_REORG:
            return std::string(TM2_OPSTR_REORG);
        case TM2_OPTYPE_RESHAPE:
            return std::string(TM2_OPSTR_RESHAPE);
        case TM2_OPTYPE_ROIPOOLING:
            return std::string(TM2_OPSTR_ROIPOOLING);
        case TM2_OPTYPE_RPN:
            return std::string(TM2_OPSTR_RPN);
        case TM2_OPTYPE_SCALE:
            return std::string(TM2_OPSTR_SCALE);
        case TM2_OPTYPE_SLICE:
            return std::string(TM2_OPSTR_SLICE);
        case TM2_OPTYPE_SOFTMAX:
            return std::string(TM2_OPSTR_SOFTMAX);
        case TM2_OPTYPE_SPLIT:
            return std::string(TM2_OPSTR_SPLIT);
        case TM2_OPTYPE_DETECTIONPOSTPROCESS:
            return std::string(TM2_OPSTR_DETECTIONPOSTPROCESS);
        case TM2_OPTYPE_GEMM:
            return std::string(TM2_OPSTR_GEMM);
        case TM2_OPTYPE_GENERIC:
            return std::string(TM2_OPSTR_GENERIC);
        case TM2_OPTYPE_LOGISTIC:
            return std::string(TM2_OPSTR_LOGISTIC);
        case TM2_OPTYPE_LSTM:
            return std::string(TM2_OPSTR_LSTM);
        case TM2_OPTYPE_RNN:
            return std::string(TM2_OPSTR_RNN);
        case TM2_OPTYPE_TANH:
            return std::string(TM2_OPSTR_TANH);
        case TM2_OPTYPE_SIGMOID:
            return std::string(TM2_OPSTR_SIGMOID);
        case TM2_OPTYPE_SQUEEZE:
            return std::string(TM2_OPSTR_SQUEEZE);
        case TM2_OPTYPE_FUSEDBNSCALERELU:
            return std::string(TM2_OPSTR_FUSEDBNSCALERELU);
        default:
            LOG_ERROR() << "Get operator string failed\n";
            return std::string("");
    }
}

}    // namespace TMSerializer2

}    // namespace TEngine
