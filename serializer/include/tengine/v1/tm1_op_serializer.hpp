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
#ifndef __TM1_OP_SERIALIZER_HPP__
#define __TM1_OP_SERIALIZER_HPP__

#include <string>
#include "static_graph_interface.hpp"
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
#include "logger.hpp"

#include "tm1_format.h"

namespace TEngine {

namespace TMSerializer1 {

using op_load_t = std::function<bool(StaticGraph*, StaticNode*, void* const, const TM_Operator*)>;

tm_uoffset_t SaveTmOperator(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
op_load_t LoadTmOpFunc(uint32_t op_type);
std::string GetOpStr(uint32_t op_type);
void AddOpStr(uint32_t op_type, const std::string& name);

#define REG_TM_OPNAME(optype, opname) AddOpStr(optype, opname);

bool LoadTmAccuracyOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmBatchNormOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmResizeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmConcatOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmConstOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmConvOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmDeconvOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmDetectionOutputOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmDropoutOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmEltwiseOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmFlattenOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmFCOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmInputOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmLRNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmNormalizeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmPermuteOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmPoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmPreluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmPriorBoxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmRegionOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmReLuOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmRelu6Op(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmReorgOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmReshapeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmROIPoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmRPNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmScaleOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmSliceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmSoftmaxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);
bool LoadTmSplitOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM_Operator* tm_op);

template <typename T> const T* GetTmPtr(void* const start_ptr, tm_uoffset_t tm_offset)
{
    if(tm_offset != NOT_SET)
        return reinterpret_cast<const T*>(reinterpret_cast<char*>(start_ptr) + tm_offset);
    else
        return nullptr;
}

}    // namespace TMSerializer1

}    // namespace TEngine

#endif
