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
#ifndef __TM2_OP_SERIALIZER_HPP__
#define __TM2_OP_SERIALIZER_HPP__

#include "static_graph_interface.hpp"
#include "logger.hpp"

#include "operator/batch_norm.hpp"
#include "operator/concat.hpp"
#include "operator/convolution.hpp"
#include "operator/deconvolution.hpp"
#include "operator/detection_output.hpp"
#include "operator/eltwise.hpp"
#include "operator/fully_connected.hpp"
#include "operator/flatten.hpp"
#include "operator/lrn.hpp"
#include "operator/normalize.hpp"
#include "operator/permute.hpp"
#include "operator/pooling.hpp"
#include "operator/priorbox.hpp"
#include "operator/region.hpp"
#include "operator/relu.hpp"
#include "operator/reorg.hpp"
#include "operator/reshape.hpp"
#include "operator/resize.hpp"
#include "operator/roi_pooling.hpp"
#include "operator/rpn.hpp"
#include "operator/scale.hpp"
#include "operator/slice.hpp"
#include "operator/softmax.hpp"
#include "operator/detection_postprocess.hpp"
#include "operator/gemm.hpp"
#include "operator/generic.hpp"
#include "operator/logistic.hpp"
#include "operator/lstm.hpp"
#include "operator/rnn.hpp"
#include "operator/tanh.hpp"
#include "operator/sigmoid.hpp"
#include "operator/squeeze.hpp"
#include "operator/fused_operator.hpp"

#include "operator/batch_norm_param.hpp"
#include "operator/concat_param.hpp"
#include "operator/conv_param.hpp"
#include "operator/deconv_param.hpp"
#include "operator/detection_output_param.hpp"
#include "operator/eltwise_param.hpp"
#include "operator/fc_param.hpp"
#include "operator/flatten_param.hpp"
#include "operator/lrn_param.hpp"
#include "operator/normalize_param.hpp"
#include "operator/permute_param.hpp"
#include "operator/pool_param.hpp"
#include "operator/priorbox_param.hpp"
#include "operator/region_param.hpp"
#include "operator/relu_param.hpp"
#include "operator/reorg_param.hpp"
#include "operator/reshape_param.hpp"
#include "operator/resize_param.hpp"
#include "operator/roi_pooling_param.hpp"
#include "operator/rpn_param.hpp"
#include "operator/scale_param.hpp"
#include "operator/slice_param.hpp"
#include "operator/softmax_param.hpp"
#include "operator/detection_postprocess_param.hpp"
#include "operator/gemm_param.hpp"
#include "operator/generic_param.hpp"
#include "operator/lstm_param.hpp"
#include "operator/rnn_param.hpp"
#include "operator/squeeze_param.hpp"

#include "tm2_format.h"

namespace TEngine {

namespace TMSerializer2 {

using op_load_t = std::function<bool(StaticGraph*, StaticNode*, void* const, const TM2_Operator*)>;
using op_save_t = std::function<tm_uoffset_t(void* const, tm_uoffset_t*, Operator*)>;

std::string GetOpStr(uint32_t op_type);

op_load_t LoadTmOpFunc(uint32_t op_type);
bool LoadTmAccuracyOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmBatchNormOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmResizeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmConcatOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmConstOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmConvOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmDeconvOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmDetectionOutputOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmDropoutOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmEltwiseOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmFlattenOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmFCOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmInputOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmLRNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmNormalizeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmPermuteOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmPoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmPreluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmPriorBoxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmRegionOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmReLuOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmRelu6Op(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmReorgOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmReshapeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmROIPoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmRPNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmScaleOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSliceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSoftmaxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSplitOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmDetectionPostProcessOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmGemmOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmGenericOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmLogisticOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmLstmOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmRnnOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmTanhOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSigmoidOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSqueezeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmFusedbnscalereluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);

op_save_t SaveTmOpFunc(uint32_t op_type);
tm_uoffset_t SaveTmAccuracyOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmBatchNormOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmConcatOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmConstOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmConvOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmDeconvOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmDetectionOutputOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmDropoutOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmEltwiseOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmFCOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmFlattenOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmInputOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmLRNOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmNormalizeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmPermuteOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmPoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmPreluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmPriorBoxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmRegionOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmReLuOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmRelu6Op(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmReorgOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmReshapeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmResizeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmROIPoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmRPNOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmScaleOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSliceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSoftmaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSplitOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmDetectionPostProcessOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmGemmOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmGenericOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmLogisticOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmLstmOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmRnnOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmTanhOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSigmoidOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSqueezeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmFusedbnscalereluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);

template <typename T> const T* GetTmPtr(void* const start_ptr, tm_uoffset_t tm_offset)
{
    if(tm_offset != TM2_NOT_SET)
        return reinterpret_cast<const T*>(reinterpret_cast<char*>(start_ptr) + tm_offset);
    else
        return nullptr;
}

}    // namespace TMSerializer2

}    // namespace TEngine

#endif
