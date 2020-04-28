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
    param.type = tm_param->type;

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
   
    // set the reverse
    int load_op_ver=tm_op->op_ver;

    if (load_op_ver==1)
    { 
        const TM2_ReshapeParam_V1* tm_param = GetTmPtr<TM2_ReshapeParam_V1>(start_ptr, tm_op->offset_t_param);
        if(tm_param->dim_0!=-2 && tm_param->dim_0!=0)
            param.re_shape.push_back(tm_param->dim_0);
        if(tm_param->dim_1!=-2 && tm_param->dim_1!=0)
            param.re_shape.push_back(tm_param->dim_1);
        if(tm_param->dim_2!=-2 && tm_param->dim_2!=0)
            param.re_shape.push_back(tm_param->dim_2);
        if(tm_param->dim_3!=-2 && tm_param->dim_3!=0)
            param.re_shape.push_back(tm_param->dim_3);
    }
    else
    {
        const TM2_ReshapeParam* tm_param = GetTmPtr<TM2_ReshapeParam>(start_ptr, tm_op->offset_t_param);
        if(tm_param->reverse)
            param.reverse = true;
        else
            param.reverse = false;
        // set the is_mxnet
        if(tm_param->is_mxnet)
            param.is_mxnet = true;
        else
            param.is_mxnet = false;

        if(tm_param->offset_re_shape != TM2_NOT_SET)
        {
            const TM2_Vector_dims* v_re_shape = GetTmPtr<TM2_Vector_dims>(start_ptr, tm_param->offset_re_shape);
            for(unsigned int i = 0; i < v_re_shape->v_num; i++){
                param.re_shape.push_back(v_re_shape->dims[i]);
            }
        }
    }
    // #endif

    
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
    if(tm_param->iscaffe == 1)
    {
        param.iscaffe = true;
    }
    else
    {
        param.iscaffe = false;
    }
    if(tm_param->ismxnet == 1)
    {
        param.ismxnet = true;
    }
    else
    {
        param.ismxnet = false;
    }
    if(tm_param->isonnx == 1){
        param.isonnx = true;
    }
    else
    {
        param.isonnx = false;
    }
    param.begin = tm_param->begin;
    param.end = tm_param->end;
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
    const std::string& op_str = TM2_OPSTR_SPLIT;
    SplitParam param = any_cast<SplitParam>(OpManager::GetOpDefParam(op_str));
    const TM2_SplitParam* tm_param = GetTmPtr<TM2_SplitParam>(start_ptr, tm_op->offset_t_param);
    if(tm_param->is_caffe)
        param.is_caffe = true;
    else
        param.is_caffe = false;

    if(tm_param->is_onnx){
        param.is_onnx = true;
    } else {
        param.is_onnx = false;
    }

    if(!param.is_caffe)
    {
        if(tm_param->is_onnx)
            param.axis = tm_param->axis;
        param.split_dim = tm_param->split_dim;
        if(tm_param->offset_split_sizes != TM2_NOT_SET)
        {
            const TM2_Vector_dims* v_split_sizes = GetTmPtr<TM2_Vector_dims>(start_ptr, tm_param->offset_split_sizes);
            for(unsigned int i = 0; i < v_split_sizes->v_num; i++)
                param.split_sizes_.push_back(v_split_sizes->dims[i]);
        }
    }

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmDetectionPostProcessOp(StaticGraph* graph, StaticNode* node, void* const start_ptr,
                                  const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_DETECTIONPOSTPROCESS;

    DetectionPostProcessParam param = any_cast<DetectionPostProcessParam>(OpManager::GetOpDefParam(op_str));
    const TM2_DetectionPostProcessParam* tm_param =
        GetTmPtr<TM2_DetectionPostProcessParam>(start_ptr, tm_op->offset_t_param);

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
    char* op_name = ( char* )malloc(tm_string->size);
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
    param.mxnet_flag = tm_param->mxnet_flag;

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

bool LoadTmAddnOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_ADDN;

    AddnParam param = any_cast<AddnParam>(OpManager::GetOpDefParam(op_str));
    const TM2_AddnParam* tm_param = GetTmPtr<TM2_AddnParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmSwapAxisOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SWAPAXIS;

    SwapAxisParam param = any_cast<SwapAxisParam>(OpManager::GetOpDefParam(op_str));
    const TM2_SwapAxisParam* tm_param = GetTmPtr<TM2_SwapAxisParam>(start_ptr, tm_op->offset_t_param);

    param.dim_0 = tm_param->dim_0;
    param.dim_1 = tm_param->dim_1;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmGruOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_GRU;

    GRUParam param = any_cast<GRUParam>(OpManager::GetOpDefParam(op_str));
    const TM2_GRUParam* tm_param = GetTmPtr<TM2_GRUParam>(start_ptr, tm_op->offset_t_param);

    param.clip = tm_param->clip;
    param.output_len = tm_param->output_len;
    param.sequence_len = tm_param->sequence_len;
    param.input_size = tm_param->input_size;
    param.hidden_size = tm_param->hidden_size;
    param.has_clip = tm_param->has_clip;
    param.has_gate_bias = tm_param->has_gate_bias;
    param.has_candidate_bias = tm_param->has_candidate_bias;
    param.has_init_state = tm_param->has_init_state;
    param.mxnet_flag = tm_param->mxnet_flag;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmMaxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_MAX);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmMinOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_MIN);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmArgMaxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_ARGMAX;

    ArgMaxParam param = any_cast<ArgMaxParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ArgMaxParam* tm_param = GetTmPtr<TM2_ArgMaxParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmTopKV2Op(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_TOPKV2;

    TopKV2Param param = any_cast<TopKV2Param>(OpManager::GetOpDefParam(op_str));
    const TM2_TopKV2Param* tm_param = GetTmPtr<TM2_TopKV2Param>(start_ptr, tm_op->offset_t_param);

    param.k = tm_param->k;
    if(tm_param->sorted)
        param.sorted = true;
    else
        param.sorted = false;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmArgMinOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_ARGMIN;

    ArgMinParam param = any_cast<ArgMinParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ArgMinParam* tm_param = GetTmPtr<TM2_ArgMinParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmStridedSliceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_STRIDEDSLICE;

    StridedSliceParam param = any_cast<StridedSliceParam>(OpManager::GetOpDefParam(op_str));
    const TM2_StridedSliceParam* tm_param = GetTmPtr<TM2_StridedSliceParam>(start_ptr, tm_op->offset_t_param);

    param.begin[0] = tm_param->begine_n;
    param.begin[1] = tm_param->begine_c;
    param.begin[2] = tm_param->begine_h;
    param.begin[3] = tm_param->begine_w;
    param.end[0] = tm_param->end_n;
    param.end[1] = tm_param->end_c;
    param.end[2] = tm_param->end_h;
    param.end[3] = tm_param->end_w;
    param.stride[0] = tm_param->stride_n;
    param.stride[1] = tm_param->stride_c;
    param.stride[2] = tm_param->stride_h;
    param.stride[3] = tm_param->stride_w;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmPadOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_PAD;

    PadParam param = any_cast<PadParam>(OpManager::GetOpDefParam(op_str));
    const TM2_PadParam* tm_param = GetTmPtr<TM2_PadParam>(start_ptr, tm_op->offset_t_param);

    param.mode = tm_param->mode;
    param.value = tm_param->value;
    param.pad_0_h = tm_param->pad_n_0;
    param.pad_0_w = tm_param->pad_n_1;
    param.pad_1_h = tm_param->pad_c_0;
    param.pad_1_w = tm_param->pad_c_1;
    param.pad_2_h = tm_param->pad_h_0;
    param.pad_2_w = tm_param->pad_h_1;
    param.pad_3_h = tm_param->pad_w_0;
    param.pad_3_w = tm_param->pad_w_1;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmReductionOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_REDUCTION;

    ReductionParam param = any_cast<ReductionParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ReductionParam* tm_param = GetTmPtr<TM2_ReductionParam>(start_ptr, tm_op->offset_t_param);

    param.dim_0 = tm_param->dim_0;
    param.dim_1 = tm_param->dim_1;
    param.dim_2 = tm_param->dim_2;
    param.dim_3 = tm_param->dim_3;
    param.type = tm_param->type;
    param.keepdim = tm_param->keepdim;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmUpsampleOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_UPSAMPLE;

    UpsampleParam param = any_cast<UpsampleParam>(OpManager::GetOpDefParam(op_str));
    const TM2_UpsampleParam* tm_param = GetTmPtr<TM2_UpsampleParam>(start_ptr, tm_op->offset_t_param);

    param.scale = tm_param->scale;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmShuffleChannelOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SHUFFLECHANNEL;

    ShuffleChannelParam param = any_cast<ShuffleChannelParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ShuffleChannelParam* tm_param = GetTmPtr<TM2_ShuffleChannelParam>(start_ptr, tm_op->offset_t_param);

    param.group = tm_param->group;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmBatchToSpaceNDOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_BATCHTOSPACEND;

    BatchToSpaceNDParam param = any_cast<BatchToSpaceNDParam>(OpManager::GetOpDefParam(op_str));
    const TM2_BatchToSpaceNDParam* tm_param = GetTmPtr<TM2_BatchToSpaceNDParam>(start_ptr, tm_op->offset_t_param);

    param.dilation_x = tm_param->dilation_x;
    param.dilation_y = tm_param->dilation_y;
    param.crop_top = tm_param->crop_top;
    param.crop_bottom = tm_param->crop_bottom;
    param.crop_left = tm_param->crop_left;
    param.crop_right = tm_param->crop_right;
					        
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmSpaceToBatchNDOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SPACETOBATCHND;

    SpaceToBatchNDParam param = any_cast<SpaceToBatchNDParam>(OpManager::GetOpDefParam(op_str));
    const TM2_SpaceToBatchNDParam* tm_param = GetTmPtr<TM2_SpaceToBatchNDParam>(start_ptr, tm_op->offset_t_param);

    param.dilation_x = tm_param->dilation_x;
    param.dilation_y = tm_param->dilation_y;
    param.pad_top = tm_param->pad_top;
    param.pad_bottom = tm_param->pad_bottom;
    param.pad_left = tm_param->pad_left;
    param.pad_right = tm_param->pad_right;
			        
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmCropOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_CROP;

    CropParam param = any_cast<CropParam>(OpManager::GetOpDefParam(op_str));
    const TM2_CropParam* tm_param = GetTmPtr<TM2_CropParam>(start_ptr, tm_op->offset_t_param);

    param.flag = tm_param->flag;
    param.crop_h = tm_param->crop_h;
    param.crop_w = tm_param->crop_w;
    param.offset_c = tm_param->offset_c;
    param.offset_h = tm_param->offset_h;
    param.offset_w = tm_param->offset_w;
    param.num_args = tm_param->num_args;
    param.center_crop = tm_param->center_crop;
    param.axis = tm_param->axis;
			        
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}


bool LoadTmPsroipoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){
    const std::string& op_str = TM2_OPSTR_PSROIPOOLING;

    PsroipoolingParam param = any_cast<PsroipoolingParam>(OpManager::GetOpDefParam(op_str));
    const TM2_PsroipoolingParam* tm_param = GetTmPtr<TM2_PsroipoolingParam>(start_ptr, tm_op->offset_t_param);

    param.output_dim = tm_param->output_dim;
    param.pooled_h = tm_param->pooled_h;
    param.pooled_w = tm_param->pooled_w;
    param.spatial_scale = tm_param->spatial_scale;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmRoialignOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){
    const std::string& op_str = TM2_OPSTR_ROIALIGN;

    RoialignParam param = any_cast<RoialignParam>(OpManager::GetOpDefParam(op_str));
    const TM2_RoialignParam* tm_param = GetTmPtr<TM2_RoialignParam>(start_ptr, tm_op->offset_t_param);

    param.pooled_height = tm_param->pooled_height;
    param.pooled_width = tm_param->pooled_width;
    param.spatial_scale = tm_param->spatial_scale;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmUnaryOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){

    const std::string& op_str = TM2_OPSTR_UNARY;

    UnaryParam param = any_cast<UnaryParam>(OpManager::GetOpDefParam(op_str));
    const TM2_UnaryParam* tm_param = GetTmPtr<TM2_UnaryParam>(start_ptr, tm_op->offset_t_param);
    param.type = tm_param->type;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmExpanddimsOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){

    const std::string& op_str = TM2_OPSTR_EXPANDDIMS;

    ExpandDimsParam param = any_cast<ExpandDimsParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ExpanddimsParam* tm_param = GetTmPtr<TM2_ExpanddimsParam>(start_ptr, tm_op->offset_t_param);
    param.axis = tm_param->axis;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmNoopOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_NOOP);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmMVNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_MVN;

    MVNParam param = any_cast<MVNParam>(OpManager::GetOpDefParam(op_str));
    const TM2_MVNParam* tm_param = GetTmPtr<TM2_MVNParam>(start_ptr, tm_op->offset_t_param);

    param.across_channels = tm_param->across_channels;
    param.eps = tm_param->eps;
    param.normalize_variance = tm_param->normalize_variance;
	 		        
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmBiasOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_BIAS;

    BiasParam param = any_cast<BiasParam>(OpManager::GetOpDefParam(op_str));
    const TM2_BiasParam* tm_param = GetTmPtr<TM2_BiasParam>(start_ptr, tm_op->offset_t_param);

    param.bias_size = tm_param->bias_size;
    
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmInstanceNormOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_INSTANCENORM;

    InstanceNormParam param = any_cast<InstanceNormParam>(OpManager::GetOpDefParam(op_str));
    const TM2_InstanceNormParam* tm_param = GetTmPtr<TM2_InstanceNormParam>(start_ptr, tm_op->offset_t_param);

    param.eps = tm_param->eps;
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmThresholdOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_THRESHOLD;

    ThresholdParam param = any_cast<ThresholdParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ThresholdParam* tm_param = GetTmPtr<TM2_ThresholdParam>(start_ptr, tm_op->offset_t_param);

    param.threshold = tm_param->threshold;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmHardsigmoidOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_HARDSIGMOID;

    HardsigmoidParam param = any_cast<HardsigmoidParam>(OpManager::GetOpDefParam(op_str));
    const TM2_HardsigmoidParam* tm_param = GetTmPtr<TM2_HardsigmoidParam>(start_ptr, tm_op->offset_t_param);

    param.alpha = tm_param->alpha;
    param.beta = tm_param->beta;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmEmbedOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_EMBED;

    EmbedParam param = any_cast<EmbedParam>(OpManager::GetOpDefParam(op_str));
    const TM2_EmbedParam* tm_param = GetTmPtr<TM2_EmbedParam>(start_ptr, tm_op->offset_t_param);

    //param.bias_term = tm_param->bias_term;
    param.input_dim = tm_param->input_dim;
    param.num_output = tm_param->num_output;
    param.weight_data_size = tm_param->weight_data_size;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmAbsvalOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_ABSVAL);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmCastOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){
    const std::string& op_str = TM2_OPSTR_CAST;

    CastParam param = any_cast<CastParam>(OpManager::GetOpDefParam(op_str));
    const TM2_CastParam* tm_param = GetTmPtr<TM2_CastParam>(start_ptr, tm_op->offset_t_param);

    param.type_from = tm_param->type_from;
    param.type_to = tm_param->type_to;
			        
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmHardSwishOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){
    const std::string& op_str = TM2_OPSTR_HARDSWISH;

    HardswishParam param = any_cast<HardswishParam>(OpManager::GetOpDefParam(op_str));
    const TM2_HardSwishParam* tm_param = GetTmPtr<TM2_HardSwishParam>(start_ptr, tm_op->offset_t_param);

    param.alpha = tm_param->alpha;
    param.beta = tm_param->beta;
			        
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmInterpOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){
    const std::string& op_str = TM2_OPSTR_INTERP;

    InterpParam param = any_cast<InterpParam>(OpManager::GetOpDefParam(op_str));
    const TM2_InterpParam* tm_param = GetTmPtr<TM2_InterpParam>(start_ptr, tm_op->offset_t_param);

    param.height_scale = tm_param->height_scale;
    param.output_height = tm_param->output_height;
    param.output_width = tm_param->output_width;
    param.resize_type = tm_param->resize_type;
    param.width_scale = tm_param->width_scale;
			        
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSeluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){
    const std::string& op_str = TM2_OPSTR_SELU;

    SeluParam param = any_cast<SeluParam>(OpManager::GetOpDefParam(op_str));
    const TM2_SeluParam* tm_param = GetTmPtr<TM2_SeluParam>(start_ptr, tm_op->offset_t_param);

    param.alpha = tm_param->alpha;
    param.lambda = tm_param->lambda;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;    
}
bool LoadTmEluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){
    const std::string& op_str = TM2_OPSTR_ELU;

    EluParam param = any_cast<EluParam>(OpManager::GetOpDefParam(op_str));
    const TM2_EluParam* tm_param = GetTmPtr<TM2_EluParam>(start_ptr, tm_op->offset_t_param);

    param.alpha = tm_param->alpha;
    
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;   
}
bool LoadTmBroadMulOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, TM2_OPSTR_BROADMUL);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmLogicalOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_LOGICAL;

    LogicalParam param = any_cast<LogicalParam>(OpManager::GetOpDefParam(op_str));
    const TM2_LogicalParam* tm_param = GetTmPtr<TM2_LogicalParam>(start_ptr, tm_op->offset_t_param);

    param.type = tm_param->type;
			        
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmGatherOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_GATHER;

    GatherParam param = any_cast<GatherParam>(OpManager::GetOpDefParam(op_str));
    const TM2_GatherParam* tm_param = GetTmPtr<TM2_GatherParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;
    param.indices_num = tm_param->indices_num;
       
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmTransposeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op){
    const std::string& op_str = TM2_OPSTR_TRANSPOSE;

    TransposeParam param = any_cast<TransposeParam>(OpManager::GetOpDefParam(op_str));
    const TM2_TransposeParam* tm_param = GetTmPtr<TM2_TransposeParam>(start_ptr, tm_op->offset_t_param);
    
    if(tm_param->offset_tr_shape != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_re_shape = GetTmPtr<TM2_Vector_dims>(start_ptr, tm_param->offset_tr_shape);
        for(unsigned int i = 0; i < v_re_shape->v_num; i++){
            param.tr_shape.push_back(v_re_shape->dims[i]);
        }
    } 
    
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;   
}
bool LoadTmReverseOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_REVERSE;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetNodeOp(node, op);
    return true;
}       
bool LoadTmComparisonOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_MVN;

    ComparisonParam param = any_cast<ComparisonParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ComparisonParam* tm_param = GetTmPtr<TM2_ComparisonParam>(start_ptr, tm_op->offset_t_param);

    param.type = tm_param->type;
	 		        
    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSpaceToDepthOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SPACETODEPTH;

    SpaceToDepthParam param = any_cast<SpaceToDepthParam>(OpManager::GetOpDefParam(op_str));
    const TM2_SpaceToDepthParam* tm_param = GetTmPtr<TM2_SpaceToDepthParam>(start_ptr, tm_op->offset_t_param);

    param.block_size = tm_param->block_size;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmDepthToSpaceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_DEPTHTOSPACE;

    DepthToSpaceParam param = any_cast<DepthToSpaceParam>(OpManager::GetOpDefParam(op_str));
    const TM2_DepthToSpaceParam* tm_param = GetTmPtr<TM2_DepthToSpaceParam>(start_ptr, tm_op->offset_t_param);

    param.block_size = tm_param->block_size;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}
bool LoadTmSquaredDifferenceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SQUAREDDIFFERENCE;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetNodeOp(node, op);
    return true;
} 

bool LoadTmSparseToDenseOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_SPARSETODENSE;

    SparseToDenseParam param = any_cast<SparseToDenseParam>(OpManager::GetOpDefParam(op_str));
    const TM2_SparseToDenseParam* tm_param = GetTmPtr<TM2_SparseToDenseParam>(start_ptr, tm_op->offset_t_param);

    param.output_shape_size0 = tm_param->output_shape_size0;
    param.output_shape_size1 = tm_param->output_shape_size1;
    param.default_value = tm_param->default_value;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmCeilOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_CEIL;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmRoundOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_ROUND;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmZerosLikeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_ZEROSLIKE;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmClipOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_CLIP;

    ClipParam param = any_cast<ClipParam>(OpManager::GetOpDefParam(op_str));
    const TM2_ClipParam* tm_param = GetTmPtr<TM2_ClipParam>(start_ptr, tm_op->offset_t_param);

    param.max = tm_param->max;
    param.min = tm_param->min;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}


bool LoadTmMatMulOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_MATMUL;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmReduceL2Op(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_REDUCEL2;

    ReduceL2Param param = any_cast<ReduceL2Param>(OpManager::GetOpDefParam(op_str));
    const TM2_ReduceL2Param* tm_param = GetTmPtr<TM2_ReduceL2Param>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;
    param.keepdim = tm_param->keepdim;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmUnsqueezeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op)
{
    const std::string& op_str = TM2_OPSTR_UNSQUEEZE;

    UnsqueezeParam param = any_cast<UnsqueezeParam>(OpManager::GetOpDefParam(op_str));
    const TM2_UnsqueezeParam* tm_param = GetTmPtr<TM2_UnsqueezeParam>(start_ptr, tm_op->offset_t_param);

    if(tm_param->offset_vi_axises != TM2_NOT_SET)
    {
        const TM2_Vector_dims* v_axises = GetTmPtr<TM2_Vector_dims>(start_ptr, tm_param->offset_vi_axises);
        for(unsigned int i = 0; i < v_axises->v_num; i++)
            param.axises.push_back(v_axises->dims[i]);
    }

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
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
        case TM2_OPTYPE_SWAPAXIS:
            return LoadTmSwapAxisOp;
        case TM2_OPTYPE_GRU:
            return LoadTmGruOp;
        case TM2_OPTYPE_ADDN:
            return LoadTmAddnOp;
        case TM2_OPTYPE_MAX:
            return LoadTmMaxOp;
        case TM2_OPTYPE_MIN:
            return LoadTmMinOp;
        case TM2_OPTYPE_ARGMAX:
            return LoadTmArgMaxOp;
        case TM2_OPTYPE_ARGMIN:
            return LoadTmArgMinOp;
        case TM2_OPTYPE_TOPKV2:
            return LoadTmTopKV2Op;
        case TM2_OPTYPE_PAD:
            return LoadTmPadOp;
        case TM2_OPTYPE_STRIDEDSLICE:
            return LoadTmStridedSliceOp;
        case TM2_OPTYPE_REDUCTION:
            return LoadTmReductionOp;
        case TM2_OPTYPE_UPSAMPLE:
            return LoadTmUpsampleOp;
        case TM2_OPTYPE_SHUFFLECHANNEL:
            return LoadTmShuffleChannelOp;
        case TM2_OPTYPE_SPACETOBATCHND:
            return LoadTmSpaceToBatchNDOp;   
        case TM2_OPTYPE_BATCHTOSPACEND:
            return LoadTmBatchToSpaceNDOp;
        case TM2_OPTYPE_RESIZE:
	        return LoadTmResizeOp;
        case TM2_OPTYPE_CROP:
            return LoadTmCropOp;
        case TM2_OPTYPE_PSROIPOOLING:
            return LoadTmPsroipoolingOp;
        case TM2_OPTYPE_ROIALIGN:
            return LoadTmRoialignOp;
        case TM2_OPTYPE_UNARY:
            return LoadTmUnaryOp;
        case TM2_OPTYPE_EXPANDDIMS:
            return LoadTmExpanddimsOp; 
        case TM2_OPTYPE_NOOP:
            return LoadTmNoopOp;
        case TM2_OPTYPE_BIAS:
            return LoadTmBiasOp;
        case TM2_OPTYPE_THRESHOLD:
            return LoadTmThresholdOp;
        case TM2_OPTYPE_EMBED:
            return LoadTmEmbedOp;
        case TM2_OPTYPE_HARDSIGMOID:
            return LoadTmHardsigmoidOp;
	    case TM2_OPTYPE_INSTANCENORM:
            return LoadTmInstanceNormOp;
        case TM2_OPTYPE_MVN:
            return LoadTmMVNOp;  
        case TM2_OPTYPE_ABSVAL:
            return LoadTmAbsvalOp;    
        case TM2_OPTYPE_CAST:
            return LoadTmCastOp;
        case TM2_OPTYPE_HARDSWISH:
            return LoadTmHardSwishOp;
        case TM2_OPTYPE_INTERP:
            return LoadTmInterpOp;
        case TM2_OPTYPE_SELU:
            return LoadTmSeluOp;
        case TM2_OPTYPE_ELU:
            return LoadTmEluOp;                          
        case TM2_OPTYPE_BROADMUL:
            return LoadTmBroadMulOp; 
        case TM2_OPTYPE_LOGICAL:
            return LoadTmLogicalOp;  
        case TM2_OPTYPE_GATHER:
            return LoadTmGatherOp;                         
        case TM2_OPTYPE_TRANSPOSE:
            return LoadTmTransposeOp; 
        case TM2_OPTYPE_COMPARISON:                                   
            return LoadTmComparisonOp;
        case TM2_OPTYPE_SPACETODEPTH:
            return LoadTmSpaceToDepthOp;
        case TM2_OPTYPE_DEPTHTOSPACE: 
            return LoadTmDepthToSpaceOp; 
        case TM2_OPTYPE_REVERSE:
            return LoadTmReverseOp;
        case TM2_OPTYPE_SQUAREDDIFFERENCE:
            return LoadTmSquaredDifferenceOp;
        case TM2_OPTYPE_SPARSETODENSE:
            return LoadTmSparseToDenseOp;
        case TM2_OPTYPE_CEIL:
            return LoadTmCeilOp;
        case TM2_OPTYPE_ROUND:
            return LoadTmRoundOp;
        case TM2_OPTYPE_ZEROSLIKE:
            return LoadTmZerosLikeOp;
        case TM2_OPTYPE_CLIP:
            return LoadTmClipOp;                                                     
	case TM2_OPTYPE_MATMUL:
	    return LoadTmMatMulOp;
	case TM2_OPTYPE_REDUCEL2:
	    return LoadTmReduceL2Op;
	case TM2_OPTYPE_UNSQUEEZE:
	    return LoadTmUnsqueezeOp;
	default:
            LOG_ERROR() << "Operator #" << op_type << " not supported in tengine model yet\n";
            return nullptr;
    }
}

using op_tm_name_map_t = std::unordered_map<unsigned int, std::string>;

static op_tm_name_map_t gTmOpName;
void AddOpStr(uint32_t op_type, const std::string& name)
{
    gTmOpName[op_type] = name;
}

std::string GetOpStr(uint32_t op_type)
{
    op_tm_name_map_t::const_iterator it = gTmOpName.find(op_type);
    if(it != gTmOpName.end())
    {
        return it->second;
    }

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
        case TM2_OPTYPE_SWAPAXIS:
            return std::string(TM2_OPSTR_SWAPAXIS);
        case TM2_OPTYPE_ADDN:
            return std::string(TM2_OPSTR_ADDN);
        case TM2_OPTYPE_GRU:
            return std::string(TM2_OPSTR_GRU);
        case TM2_OPTYPE_MAX:
            return std::string(TM2_OPSTR_MAX);
        case TM2_OPTYPE_MIN:
            return std::string(TM2_OPSTR_MIN);
        case TM2_OPTYPE_ARGMAX:
            return std::string(TM2_OPSTR_ARGMAX);
        case TM2_OPTYPE_ARGMIN:
            return std::string(TM2_OPSTR_ARGMIN);
        case TM2_OPTYPE_TOPKV2:
            return std::string(TM2_OPSTR_TOPKV2);
        case TM2_OPTYPE_PAD:
            return std::string(TM2_OPSTR_PAD);
        case TM2_OPTYPE_STRIDEDSLICE:
            return std::string(TM2_OPSTR_STRIDEDSLICE);
        case TM2_OPTYPE_REDUCTION:
            return std::string(TM2_OPSTR_REDUCTION);
        case TM2_OPTYPE_UPSAMPLE:
            return std::string(TM2_OPSTR_UPSAMPLE);
        case TM2_OPTYPE_SHUFFLECHANNEL:
            return std::string(TM2_OPSTR_SHUFFLECHANNEL);
        case TM2_OPTYPE_SPACETOBATCHND:
            return std::string(TM2_OPSTR_SPACETOBATCHND);    
        case TM2_OPTYPE_BATCHTOSPACEND:
            return std::string(TM2_OPSTR_BATCHTOSPACEND); 
        case TM2_OPTYPE_RESIZE:
	        return std::string(TM2_OPSTR_RESIZE);
        case TM2_OPTYPE_CROP:
            return std::string(TM2_OPSTR_CROP);
        case TM2_OPTYPE_PSROIPOOLING:
            return std::string(TM2_OPSTR_PSROIPOOLING);
        case TM2_OPTYPE_ROIALIGN:
            return std::string(TM2_OPSTR_ROIALIGN); 
        case TM2_OPTYPE_EXPANDDIMS:
            return std::string(TM2_OPSTR_EXPANDDIMS);
        case TM2_OPTYPE_UNARY:
            return std::string(TM2_OPSTR_UNARY);    
        case TM2_OPTYPE_BIAS:
            return std::string(TM2_OPSTR_BIAS);
        case TM2_OPTYPE_NOOP:
            return std::string(TM2_OPSTR_NOOP);
        case TM2_OPTYPE_THRESHOLD:
            return std::string(TM2_OPSTR_THRESHOLD);
        case TM2_OPTYPE_HARDSIGMOID:
            return std::string(TM2_OPSTR_HARDSIGMOID);
        case TM2_OPTYPE_EMBED:
            return std::string(TM2_OPSTR_EMBED);
        case TM2_OPTYPE_INSTANCENORM:
            return std::string(TM2_OPSTR_INSTANCENORM);
        case TM2_OPTYPE_MVN:
            return std::string(TM2_OPSTR_MVN); 
        case TM2_OPTYPE_ABSVAL:
            return std::string(TM2_OPSTR_ABSVAL);  
        case TM2_OPTYPE_CAST:
            return std::string(TM2_OPSTR_CAST);  
        case TM2_OPTYPE_HARDSWISH:
            return std::string(TM2_OPSTR_HARDSWISH); 
        case TM2_OPTYPE_INTERP:
            return std::string(TM2_OPSTR_INTERP); 
        case TM2_OPTYPE_SELU:
            return std::string(TM2_OPSTR_SELU); 
        case TM2_OPTYPE_ELU:
            return std::string(TM2_OPSTR_ELU);                        
        case TM2_OPTYPE_BROADMUL:
            return std::string(TM2_OPSTR_BROADMUL); 
        case TM2_OPTYPE_LOGICAL:
            return std::string(TM2_OPSTR_LOGICAL); 
        case TM2_OPTYPE_GATHER:
            return std::string(TM2_OPSTR_GATHER);                        
        case TM2_OPTYPE_TRANSPOSE:
            return std::string(TM2_OPSTR_TRANSPOSE);
        case TM2_OPTYPE_COMPARISON:
            return std::string(TM2_OPSTR_COMPARISON); 
        case TM2_OPTYPE_SPACETODEPTH:
            return std::string(TM2_OPSTR_SPACETODEPTH);
        case TM2_OPTYPE_DEPTHTOSPACE:                  
            return std::string(TM2_OPSTR_DEPTHTOSPACE);   
        case TM2_OPTYPE_REVERSE:
            return std::string(TM2_OPSTR_REVERSE); 
        case TM2_OPTYPE_SQUAREDDIFFERENCE:
            return std::string(TM2_OPSTR_SQUAREDDIFFERENCE);
        case TM2_OPTYPE_SPARSETODENSE:
            return std::string(TM2_OPSTR_SPARSETODENSE);
        case TM2_OPTYPE_CEIL:
            return std::string(TM2_OPSTR_CEIL);
        case TM2_OPTYPE_ROUND:
            return std::string(TM2_OPSTR_ROUND);
        case TM2_OPTYPE_ZEROSLIKE:
            return std::string(TM2_OPSTR_ZEROSLIKE);
        case TM2_OPTYPE_CLIP:
            return std::string(TM2_OPSTR_CLIP);
        case TM2_OPTYPE_MATMUL:
	    return std::string(TM2_OPSTR_MATMUL);	    
        case TM2_OPTYPE_REDUCEL2:
	    return std::string(TM2_OPSTR_REDUCEL2);
	case TM2_OPTYPE_UNSQUEEZE:
            return std::string(TM2_OPSTR_UNSQUEEZE);
	default:
            LOG_ERROR() << "Get operator string failed\n";
            return std::string("");
    }
}

}    // namespace TMSerializer2

}    // namespace TEngine
