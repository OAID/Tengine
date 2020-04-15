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
#include "operator/argmax.hpp"
#include "operator/argmin.hpp"
#include "operator/maximum.hpp"
#include "operator/minimum.hpp"
#include "operator/topkv2.hpp"
#include "operator/reduction.hpp"
#include "operator/stridedslice.hpp"
#include "operator/pad.hpp"
#include "operator/split.hpp"
#include "operator/swap_axis.hpp"
#include "operator/gru.hpp"
#include "operator/add_n.hpp"
#include "operator/fused_operator.hpp"
#include "operator/upsample.hpp"
#include "operator/shuffle_channel.hpp"
#include "operator/spaceToBatchND.hpp"
#include "operator/batchToSpaceND.hpp"
#include "operator/crop.hpp"
#include "operator/psroipooling.hpp"
#include "operator/unary.hpp"
#include "operator/roialign.hpp"
#include "operator/expanddims.hpp"
#include "operator/bias.hpp"
#include "operator/noop.hpp"
#include "operator/threshold.hpp"
#include "operator/hardsigmoid.hpp"
#include "operator/embed.hpp"
#include "operator/instancenorm.hpp"
#include "operator/mvn.hpp"
#include "operator/absval.hpp"
#include "operator/cast.hpp"
#include "operator/hardswish.hpp"
#include "operator/interp.hpp"
#include "operator/selu.hpp"
#include "operator/elu.hpp"
#include "operator/broadmul.hpp"
#include "operator/logical.hpp"
#include "operator/gather.hpp"
#include "operator/transpose.hpp"
#include "operator/reverse.hpp"
#include "operator/squared_difference.hpp"
#include "operator/sparsetodense.hpp"
#include "operator/ceil.hpp"
#include "operator/round.hpp"
#include "operator/zeros_like.hpp"
#include "operator/comparison.hpp"
#include "operator/spacetodepth.hpp"
#include "operator/depthtospace.hpp"
#include "operator/clip.hpp"
#include "operator/matmul.hpp"
#include "operator/reducel2.hpp"
#include "operator/unsqueeze.hpp"

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
#include "operator/addn_param.hpp"
#include "operator/gru_param.hpp"
#include "operator/swap_axis_param.hpp"
#include "operator/squeeze_param.hpp"
#include "operator/argmax_param.hpp"
#include "operator/argmin_param.hpp"
#include "operator/reduction_param.hpp"
#include "operator/topkv2_param.hpp"
#include "operator/stridedslice_param.hpp"
#include "operator/pad_param.hpp"
#include "operator/split_param.hpp"
#include "operator/upsample_param.hpp"
#include "operator/shuffle_channel_param.hpp"
#include "operator/spaceToBatchND_param.hpp"
#include "operator/batchToSpaceND_param.hpp"
#include "operator/crop_param.hpp"
#include "operator/psroipooling_param.hpp"
#include "operator/unary_param.hpp"
#include "operator/roialign_param.hpp"
#include "operator/expanddims_param.hpp"
#include "operator/bias_param.hpp"
#include "operator/threshold_param.hpp"
#include "operator/hardsigmoid_param.hpp"
#include "operator/embed_param.hpp"
#include "operator/instancenorm_param.hpp"
#include "operator/mvn_param.hpp"
#include "operator/cast_param.hpp"
#include "operator/hardswish_param.hpp"
#include "operator/interp_param.hpp"
#include "operator/selu_param.hpp"
#include "operator/elu_param.hpp"
#include "operator/logical_param.hpp"
#include "operator/gather_param.hpp"
#include "operator/transpose_param.hpp"
#include "operator/comparison_param.hpp"
#include "operator/spacetodepth_param.hpp"
#include "operator/depthtospace_param.hpp"
#include "operator/sparsetodense_param.hpp"
#include "operator/clip_param.hpp"
#include "operator/reducel2_param.hpp"
#include "operator/unsqueeze_param.hpp"
#include "tm2_format.h"

namespace TEngine {

namespace TMSerializer2 {

using op_load_t = std::function<bool(StaticGraph*, StaticNode*, void* const, const TM2_Operator*)>;
using op_save_t = std::function<tm_uoffset_t(void* const, tm_uoffset_t*, Operator*)>;

std::string GetOpStr(uint32_t op_type);
void AddOpStr(uint32_t op_type, const std::string& name);

#define REG_TM_OPNAME(optype, opname) AddOpStr(optype, opname);

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
bool LoadTmDetectionPostProcessOp(StaticGraph* graph, StaticNode* node, void* const start_ptr,
                                  const TM2_Operator* tm_op);
bool LoadTmGemmOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmGenericOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmLogisticOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmLstmOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmRnnOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmTanhOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSigmoidOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSqueezeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmFusedbnscalereluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmPadOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmArgMaxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmArgMinOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmReductionOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmTopKV2Op(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmStridedSliceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmMaxOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmMinOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSwapAixsOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmGruOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmAddnOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmUpsampleOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmShuffleChannelOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSpaceToBatchNDOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmBatchToSpaceNDOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmCropOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmPsroipoolingOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmRoialignOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmUnaryOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmExpanddimsOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmBiasOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmThresholdOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmNoopOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmHardsigmoidOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmEmbedOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmInstanceNormOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmMVNOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmAbsvalOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmCastOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmHardSwishOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmInterpOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSeluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmEluOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmBroadMulOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmLogicalOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmGatherOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmTransposeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmReverseOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmComparisonOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSpaceToDepthOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmDepthToSpaceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSquaredDifferenceOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmSparseToDenseOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmCeilOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmRoundOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmZerosLikeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmClipOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmMatMulOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmReduceL2Op(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);
bool LoadTmUnsqueezeOp(StaticGraph* graph, StaticNode* node, void* const start_ptr, const TM2_Operator* tm_op);

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
tm_uoffset_t SaveTmPadOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmStridedSliceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmArgMaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmArgMinOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmTopKV2Op(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmReductionOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmMaxOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmMinOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmAddnOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmGruOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSwapAxisOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmUpsampleOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmShffleChannelOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSpaceToBatchNDOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmBatchToSpaceNDOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmCropOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmUnaryOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmPsroipoolingOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmExpanddimsOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmRoialignOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmBiasOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmThresholdOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmNoopOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmHardsigmoidOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmEmbedOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmInstanceNormOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmMVNOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmAbsvalOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmCastOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmHardSwishOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmInterpOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSeluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmEluOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmBroadMulOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmLogicalOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmGatherOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmTransposeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmReverseOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmComparisonOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSpaceToDepthOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmDepthToSpaceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSquaredDifferenceOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmSparseToDenseOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmCeilOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmRoundOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmZerosLikeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmClipOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmMatMulOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmReduceL2Op(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);
tm_uoffset_t SaveTmUnsqueezeOp(void* const start_ptr, tm_uoffset_t* cur_pos, Operator* op);

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
