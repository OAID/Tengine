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

bool LoadTmAccuracyOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, OP_STR_ACCURACY);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmBatchNormOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_BATCHNORMALIZATION;

    BatchNormParam param = any_cast<BatchNormParam>(OpManager::GetOpDefParam(op_str));
    const TM_BatchNormParam* tm_param = GetTmPtr<TM_BatchNormParam>(start_ptr, tm_op->offset_t_param);

    param.rescale_factor = tm_param->rescale_factor;
    param.eps = tm_param->eps;
    param.caffe_flavor = tm_param->caffe_flavor;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmResizeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_BILINEARRESIZE;

    ResizeParam param = any_cast<ResizeParam>(OpManager::GetOpDefParam(op_str));
    const TM_ResizeParam* tm_param = GetTmPtr<TM_ResizeParam>(start_ptr, tm_op->offset_t_param);

    param.scale_w = tm_param->scale_x;
    param.scale_h = tm_param->scale_y;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmConcatOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_CONCAT;

    ConcatParam param = any_cast<ConcatParam>(OpManager::GetOpDefParam(op_str));
    const TM_ConcatParam* tm_param = GetTmPtr<TM_ConcatParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmConstOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, OP_STR_CONST);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmConvOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_CONVOLUTION;

    ConvParam param = any_cast<ConvParam>(OpManager::GetOpDefParam(op_str));
    const TM_ConvParam* tm_param = GetTmPtr<TM_ConvParam>(start_ptr, tm_op->offset_t_param);

    param.kernel_h = tm_param->kernel_h;
    param.kernel_w = tm_param->kernel_w;
    param.stride_h = tm_param->stride_h;
    param.stride_w = tm_param->stride_w;
    param.dilation_h = tm_param->dilation_h;
    param.dilation_w = tm_param->dilation_w;
    param.output_channel = tm_param->output_channel;
    param.activation = tm_param->activation;
    param.group = tm_param->group;
    param.pad_h0 = tm_param->pad_h;
    param.pad_h1 = tm_param->pad_h;
    param.pad_w0 = tm_param->pad_w;
    param.pad_w1 = tm_param->pad_w;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmDeconvOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_DECONVOLUTION;

    DeconvParam param = any_cast<DeconvParam>(OpManager::GetOpDefParam(op_str));
    const TM_DeconvParam* tm_param = GetTmPtr<TM_DeconvParam>(start_ptr, tm_op->offset_t_param);

    param.kernel_h = tm_param->kernel_size;
    param.kernel_w = tm_param->kernel_size;
    param.stride_h = tm_param->stride;
    param.stride_w = tm_param->stride;
    param.pad_w0 = tm_param->pad;
    param.pad_w1 = tm_param->pad;
    param.pad_h0 = tm_param->pad;
    param.pad_h1 = tm_param->pad;
    param.num_output = tm_param->num_output;
    param.dilation_h = tm_param->dilation;
    param.dilation_w = tm_param->dilation;
    param.group = 1;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmDetectionOutputOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_DETECTIONOUTPUT;

    DetectionOutputParam param = any_cast<DetectionOutputParam>(OpManager::GetOpDefParam(op_str));
    const TM_DetectionOutputParam* tm_param = GetTmPtr<TM_DetectionOutputParam>(start_ptr, tm_op->offset_t_param);

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

bool LoadTmDropoutOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, OP_STR_DROPOUT);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmEltwiseOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_ELTWISE;

    EltwiseParam param = any_cast<EltwiseParam>(OpManager::GetOpDefParam(op_str));
    const TM_EltwiseParam* tm_param = GetTmPtr<TM_EltwiseParam>(start_ptr, tm_op->offset_t_param);

    param.type = static_cast<EltType>(tm_param->type);
    param.caffe_flavor = tm_param->caffe_flavor;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmFlattenOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_FLATTEN;

    FlattenParam param = any_cast<FlattenParam>(OpManager::GetOpDefParam(op_str));
    const TM_FlattenParam* tm_param = GetTmPtr<TM_FlattenParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;
    param.end_axis = tm_param->end_axis;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmFCOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_FULLYCONNECTED;

    FCParam param = any_cast<FCParam>(OpManager::GetOpDefParam(op_str));
    const TM_FCParam* tm_param = GetTmPtr<TM_FCParam>(start_ptr, tm_op->offset_t_param);

    param.num_output = tm_param->num_output;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmInputOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, OP_STR_INPUTOP);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmLRNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_LRN;

    LRNParam param = any_cast<LRNParam>(OpManager::GetOpDefParam(op_str));
    const TM_LRNParam* tm_param = GetTmPtr<TM_LRNParam>(start_ptr, tm_op->offset_t_param);

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

bool LoadTmNormalizeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_NORMALIZE;

    NormalizeParam param = any_cast<NormalizeParam>(OpManager::GetOpDefParam(op_str));
    const TM_NormalizeParam* tm_param = GetTmPtr<TM_NormalizeParam>(start_ptr, tm_op->offset_t_param);

    param.across_spatial = tm_param->across_spatial;
    param.channel_shared = tm_param->channel_shared;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmPermuteOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_PERMUTE;

    PermuteParam param = any_cast<PermuteParam>(OpManager::GetOpDefParam(op_str));
    const TM_PermuteParam* tm_param = GetTmPtr<TM_PermuteParam>(start_ptr, tm_op->offset_t_param);

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

bool LoadTmPoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_POOLING;

    PoolParam param = any_cast<PoolParam>(OpManager::GetOpDefParam(op_str));
    const TM_PoolParam* tm_param = GetTmPtr<TM_PoolParam>(start_ptr, tm_op->offset_t_param);

    param.alg = static_cast<PoolArg>(tm_param->alg);
    param.kernel_h = tm_param->kernel_h;
    param.kernel_w = tm_param->kernel_w;
    param.stride_h = tm_param->stride_h;
    param.stride_w = tm_param->stride_w;
    param.global = tm_param->global;
    param.caffe_flavor = tm_param->caffe_flavor;
    param.pad_h0 = tm_param->pads[0];
    param.pad_w0 = tm_param->pads[1];
    param.pad_h1 = tm_param->pads[2];
    param.pad_w1 = tm_param->pads[3];

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmPreluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, OP_STR_PRELU);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmPriorBoxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_PRIORBOX;

    PriorBoxParam param = any_cast<PriorBoxParam>(OpManager::GetOpDefParam(op_str));
    const TM_PriorBoxParam* tm_param = GetTmPtr<TM_PriorBoxParam>(start_ptr, tm_op->offset_t_param);
    const TM_Vector_floats* v_minsizes = GetTmPtr<TM_Vector_floats>(start_ptr, tm_param->offset_vf_min_size);
    const TM_Vector_floats* v_maxsizes = GetTmPtr<TM_Vector_floats>(start_ptr, tm_param->offset_vf_max_size);
    const TM_Vector_floats* v_variances = GetTmPtr<TM_Vector_floats>(start_ptr, tm_param->offset_vf_variance);
    const TM_Vector_floats* v_ratios = GetTmPtr<TM_Vector_floats>(start_ptr, tm_param->offset_vf_aspect_ratio);

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

bool LoadTmRegionOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_REGION;

    RegionParam param = any_cast<RegionParam>(OpManager::GetOpDefParam(op_str));
    const TM_RegionParam* tm_param = GetTmPtr<TM_RegionParam>(start_ptr, tm_op->offset_t_param);
    const TM_Vector_floats* v_biases = GetTmPtr<TM_Vector_floats>(start_ptr, tm_param->offset_vf_biases);

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

bool LoadTmReLuOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_RELU;

    ReLuParam param = any_cast<ReLuParam>(OpManager::GetOpDefParam(op_str));
    const TM_ReLuParam* tm_param = GetTmPtr<TM_ReLuParam>(start_ptr, tm_op->offset_t_param);

    param.negative_slope = tm_param->negative_slope;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmRelu6Op(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, OP_STR_RELU6);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmReorgOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_REORG;

    ReorgParam param = any_cast<ReorgParam>(OpManager::GetOpDefParam(op_str));
    const TM_ReorgParam* tm_param = GetTmPtr<TM_ReorgParam>(start_ptr, tm_op->offset_t_param);

    param.stride = tm_param->stride;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmReshapeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_RESHAPE;

    ReshapeParam param = any_cast<ReshapeParam>(OpManager::GetOpDefParam(op_str));
    const TM_ReshapeParam* tm_param = GetTmPtr<TM_ReshapeParam>(start_ptr, tm_op->offset_t_param);
    
        // set the reverse
    if(tm_param->reverse)
        param.reverse = true;
    else 
        param.reverse =false;
    // set the is_mxnet
    if(tm_param->is_mxnet)
        param.is_mxnet = true;
    else 
        param.is_mxnet = false;

    if(tm_param->offset_re_shape != NOT_SET)
    {    
        const TM_Vector_dims* v_re_shape = GetTmPtr<TM_Vector_dims>(start_ptr, tm_param->offset_re_shape);
        for(unsigned int i = 0; i < v_re_shape->v_num; i++) 
            param.re_shape.push_back(v_re_shape->dims[i]);
    } 

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;

}

bool LoadTmROIPoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_ROIPOOLING;

    ROIPoolingParam param = any_cast<ROIPoolingParam>(OpManager::GetOpDefParam(op_str));
    const TM_ROIPoolingParam* tm_param = GetTmPtr<TM_ROIPoolingParam>(start_ptr, tm_op->offset_t_param);

    param.pooled_h = tm_param->pooled_h;
    param.pooled_w = tm_param->pooled_w;
    param.spatial_scale = tm_param->spatial_scale;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmRPNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_RPN;

    RPNParam param = any_cast<RPNParam>(OpManager::GetOpDefParam(op_str));
    const TM_RPNParam* tm_param = GetTmPtr<TM_RPNParam>(start_ptr, tm_op->offset_t_param);
    const TM_Vector_floats* v_ratios = GetTmPtr<TM_Vector_floats>(start_ptr, tm_param->offset_vf_ratios);
    const TM_Vector_floats* v_scales = GetTmPtr<TM_Vector_floats>(start_ptr, tm_param->offset_vf_anchor_scales);

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

bool LoadTmScaleOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_SCALE;

    ScaleParam param = any_cast<ScaleParam>(OpManager::GetOpDefParam(op_str));
    const TM_ScaleParam* tm_param = GetTmPtr<TM_ScaleParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;
    param.num_axes = tm_param->num_axes;
    param.bias_term = tm_param->bias_term;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSliceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_SLICE;

    SliceParam param = any_cast<SliceParam>(OpManager::GetOpDefParam(op_str));
    const TM_SliceParam* tm_param = GetTmPtr<TM_SliceParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;
    param.iscaffe = true;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSoftmaxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    const std::string& op_str = OP_STR_SOFTMAX;

    SoftmaxParam param = any_cast<SoftmaxParam>(OpManager::GetOpDefParam(op_str));
    const TM_SoftmaxParam* tm_param = GetTmPtr<TM_SoftmaxParam>(start_ptr, tm_op->offset_t_param);

    param.axis = tm_param->axis;

    StaticOp* op = CreateStaticOp(graph, op_str);
    SetOperatorParam(op, param);
    SetNodeOp(node, op);
    return true;
}

bool LoadTmSplitOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op)
{
    StaticOp* op = CreateStaticOp(graph, OP_STR_SPLIT);
    SetNodeOp(node, op);
    return true;
}

op_load_t LoadTmOpFunc(uint32_t op_type)
{
    switch(op_type)
    {
        case TM_OPTYPE_ACCURACY:
            return LoadTmAccuracyOp;
        case TM_OPTYPE_BATCHNORMALIZATION:
            return LoadTmBatchNormOp;
        case TM_OPTYPE_BILINEARRESIZE:
            return LoadTmResizeOp;
        case TM_OPTYPE_CONCAT:
            return LoadTmConcatOp;
        case TM_OPTYPE_CONST:
            return LoadTmConstOp;
        case TM_OPTYPE_CONVOLUTION:
            return LoadTmConvOp;
        case TM_OPTYPE_DECONVOLUTION:
            return LoadTmDeconvOp;
        case TM_OPTYPE_DETECTIONOUTPUT:
            return LoadTmDetectionOutputOp;
        case TM_OPTYPE_DROPOUT:
            return LoadTmDropoutOp;
        case TM_OPTYPE_ELTWISE:
            return LoadTmEltwiseOp;
        case TM_OPTYPE_FLATTEN:
            return LoadTmFlattenOp;
        case TM_OPTYPE_FULLYCONNECTED:
            return LoadTmFCOp;
        case TM_OPTYPE_INPUTOP:
            return LoadTmInputOp;
        case TM_OPTYPE_LRN:
            return LoadTmLRNOp;
        case TM_OPTYPE_NORMALIZE:
            return LoadTmNormalizeOp;
        case TM_OPTYPE_PERMUTE:
            return LoadTmPermuteOp;
        case TM_OPTYPE_POOLING:
            return LoadTmPoolingOp;
        case TM_OPTYPE_PRELU:
            return LoadTmPreluOp;
        case TM_OPTYPE_PRIORBOX:
            return LoadTmPriorBoxOp;
        case TM_OPTYPE_REGION:
            return LoadTmRegionOp;
        case TM_OPTYPE_RELU:
            return LoadTmReLuOp;
        case TM_OPTYPE_RELU6:
            return LoadTmRelu6Op;
        case TM_OPTYPE_REORG:
            return LoadTmReorgOp;
        case TM_OPTYPE_RESHAPE:
            return LoadTmReshapeOp;
        case TM_OPTYPE_ROIPOOLING:
            return LoadTmROIPoolingOp;
        case TM_OPTYPE_RPN:
            return LoadTmRPNOp;
        case TM_OPTYPE_SCALE:
            return LoadTmScaleOp;
        case TM_OPTYPE_SLICE:
            return LoadTmSliceOp;
        case TM_OPTYPE_SOFTMAX:
            return LoadTmSoftmaxOp;
        case TM_OPTYPE_SPLIT:
            return LoadTmSplitOp;
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
    switch(op_type)
    {
        case TM_OPTYPE_ACCURACY:
            return std::string(OP_STR_ACCURACY);
        case TM_OPTYPE_BATCHNORMALIZATION:
            return std::string(OP_STR_BATCHNORMALIZATION);
        case TM_OPTYPE_BILINEARRESIZE:
            return std::string(OP_STR_BILINEARRESIZE);
        case TM_OPTYPE_CONCAT:
            return std::string(OP_STR_CONCAT);
        case TM_OPTYPE_CONST:
            return std::string(OP_STR_CONST);
        case TM_OPTYPE_CONVOLUTION:
            return std::string(OP_STR_CONVOLUTION);
        case TM_OPTYPE_DECONVOLUTION:
            return std::string(OP_STR_DECONVOLUTION);
        case TM_OPTYPE_DETECTIONOUTPUT:
            return std::string(OP_STR_DETECTIONOUTPUT);
        case TM_OPTYPE_DROPOUT:
            return std::string(OP_STR_DROPOUT);
        case TM_OPTYPE_ELTWISE:
            return std::string(OP_STR_ELTWISE);
        case TM_OPTYPE_FLATTEN:
            return std::string(OP_STR_FLATTEN);
        case TM_OPTYPE_FULLYCONNECTED:
            return std::string(OP_STR_FULLYCONNECTED);
        case TM_OPTYPE_INPUTOP:
            return std::string(OP_STR_INPUTOP);
        case TM_OPTYPE_LRN:
            return std::string(OP_STR_LRN);
        case TM_OPTYPE_NORMALIZE:
            return std::string(OP_STR_NORMALIZE);
        case TM_OPTYPE_PERMUTE:
            return std::string(OP_STR_PERMUTE);
        case TM_OPTYPE_POOLING:
            return std::string(OP_STR_POOLING);
        case TM_OPTYPE_PRELU:
            return std::string(OP_STR_PRELU);
        case TM_OPTYPE_PRIORBOX:
            return std::string(OP_STR_PRIORBOX);
        case TM_OPTYPE_REGION:
            return std::string(OP_STR_REGION);
        case TM_OPTYPE_RELU:
            return std::string(OP_STR_RELU);
        case TM_OPTYPE_RELU6:
            return std::string(OP_STR_RELU6);
        case TM_OPTYPE_REORG:
            return std::string(OP_STR_REORG);
        case TM_OPTYPE_RESHAPE:
            return std::string(OP_STR_RESHAPE);
        case TM_OPTYPE_ROIPOOLING:
            return std::string(OP_STR_ROIPOOLING);
        case TM_OPTYPE_RPN:
            return std::string(OP_STR_RPN);
        case TM_OPTYPE_SCALE:
            return std::string(OP_STR_SCALE);
        case TM_OPTYPE_SLICE:
            return std::string(OP_STR_SLICE);
        case TM_OPTYPE_SOFTMAX:
            return std::string(OP_STR_SOFTMAX);
        case TM_OPTYPE_SPLIT:
            return std::string(OP_STR_SPLIT);
        default:
            LOG_ERROR() << "Get operator string failed\n";
            return std::string("");
    }
}

}    // namespace TMSerializer1

}    // namespace TEngine
