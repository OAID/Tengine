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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <iostream>
#include <functional>
#include "logger.hpp"

#include "operator/convolution.hpp"
#include "operator/input_op.hpp"
#include "operator/pooling.hpp"
#include "operator/softmax.hpp"
#include "operator/fully_connected.hpp"
#include "operator/split.hpp"
#include "operator/concat.hpp"
#include "operator/const_op.hpp"
#include "operator/accuracy.hpp"
#include "operator/dropout.hpp"
#include "operator/relu.hpp"
#include "operator/relu6.hpp"
#include "operator/batch_norm.hpp"
#include "operator/scale.hpp"
#include "operator/lrn.hpp"
#include "operator/fused_operator.hpp"
#include "operator/prelu.hpp"
#include "operator/eltwise.hpp"
#include "operator/slice.hpp"
#include "operator/demo_op.hpp"
#include "operator/normalize.hpp"
#include "operator/permute.hpp"
#include "operator/flatten.hpp"
#include "operator/priorbox.hpp"
#include "operator/reshape.hpp"
#include "operator/detection_output.hpp"
#include "operator/rpn.hpp"
#include "operator/roi_pooling.hpp"
#include "operator/reorg.hpp"
#include "operator/region.hpp"
#include "operator/deconvolution.hpp"
#include "operator/resize.hpp"
#include "operator/gemm.hpp"
#include "operator/generic.hpp"
#include "operator/lstm.hpp"
#include "operator/logistic.hpp"
#include "operator/detection_postprocess.hpp"
#include "operator/rnn.hpp"
#include "operator/tanh.hpp"
#include "operator/sigmoid.hpp"
#include "operator/squeeze.hpp"
#include "operator/pad.hpp"
#include "operator/reduction.hpp"
#include "operator/swap_axis.hpp"
#include "operator/gru.hpp"
#include "operator/add_n.hpp"
#include "operator/stridedslice.hpp"
#include "operator/upsample.hpp"
#include "operator/crop.hpp"
#include "operator/copy.hpp"
#include "operator/power.hpp"
#include "operator/floor.hpp"
#include "operator/clip.hpp"
#include "operator/tile.hpp"
#include "operator/topkv2.hpp"
#include "operator/maximum.hpp"
#include "operator/matmul.hpp"
#include "operator/minimum.hpp"
#include "operator/argmax.hpp"
#include "operator/argmin.hpp"
#include "operator/reverse.hpp"
#include "operator/feature_match.hpp"
#include "operator/shuffle_channel.hpp"
#include "operator/batchToSpaceND.hpp"
#include "operator/spaceToBatchND.hpp"
#include "operator/absval.hpp"
#include "operator/hardswish.hpp"
#include "operator/interp.hpp"
#include "operator/selu.hpp"
#include "operator/l2normalization.hpp"
#include "operator/l2pool.hpp"
#include "operator/reducel2.hpp"
#include "operator/elu.hpp"
#include "operator/layernormlstm.hpp"
#include "operator/relu1.hpp"
#include "operator/log_softmax.hpp"
#include "operator/cast.hpp"
#include "operator/expanddims.hpp"
#include "operator/unary.hpp"
#include "operator/roialign.hpp"
#include "operator/psroipooling.hpp"
#include "operator/bias.hpp"
#include "operator/noop.hpp"
#include "operator/threshold.hpp"
#include "operator/hardsigmoid.hpp"
#include "operator/embed.hpp"
#include "operator/instancenorm.hpp"
#include "operator/mvn.hpp"
#include "operator/broadmul.hpp"
#include "operator/logical.hpp"
#include "operator/gather.hpp"
#include "operator/transpose.hpp"
#include "operator/comparison.hpp"
#include "operator/spacetodepth.hpp"
#include "operator/depthtospace.hpp"
#include "operator/squared_difference.hpp"
#include "operator/sparsetodense.hpp"
#include "operator/ceil.hpp"
#include "operator/round.hpp"
#include "operator/zeros_like.hpp"
#include "operator/unsqueeze.hpp"
#include "operator/yolov3detectionoutput.hpp"
using namespace TEngine;
 
int operator_plugin_init(void)
{
    RegisterOp<Convolution>("Convolution");
    RegisterOp<InputOp>("InputOp");
    RegisterOp<Pooling>("Pooling");
    RegisterOp<Softmax>("Softmax");
    RegisterOp<FullyConnected>("FullyConnected");
    RegisterOp<Accuracy>("Accuracy");
    RegisterOp<Concat>("Concat");
    RegisterOp<Dropout>("Dropout");
    RegisterOp<Split>("Split");
    RegisterOp<ConstOp>("Const");
    RegisterOp<ReLu>("ReLu");
    RegisterOp<ReLu6>("ReLu6");
    RegisterOp<BatchNorm>(BatchNormName);
    RegisterOp<Scale>("Scale");
    RegisterOp<LRN>("LRN");
    RegisterOp<FusedBNScaleReLu>(FusedBNScaleReLu::class_name);
    RegisterOp<PReLU>("PReLU");
    RegisterOp<Eltwise>("Eltwise");
    RegisterOp<Slice>("Slice");
    RegisterOp<DemoOp>("DemoOp");
    RegisterOp<Normalize>("Normalize");
    RegisterOp<Permute>("Permute");
    RegisterOp<Flatten>("Flatten");
    RegisterOp<PriorBox>("PriorBox");
    RegisterOp<Reshape>("Reshape");
    RegisterOp<DetectionOutput>("DetectionOutput");
    RegisterOp<RPN>("RPN");
    RegisterOp<ROIPooling>("ROIPooling");
    RegisterOp<Reorg>("Reorg");
    RegisterOp<Region>("Region");
    RegisterOp<Deconvolution>("Deconvolution");
    RegisterOp<Resize>("Resize");
    RegisterOp<Gemm>("Gemm");
    RegisterOp<Generic>("Generic");
    RegisterOp<LSTM>("LSTM");
    RegisterOp<Logistic>("Logistic");
    RegisterOp<DetectionPostProcess>("DetectionPostProcess");
    RegisterOp<RNN>("RNN");
    RegisterOp<Tanh>("Tanh");
    RegisterOp<Sigmoid>("Sigmoid");
    RegisterOp<Squeeze>("Squeeze");
    RegisterOp<Pad>("Pad");
    RegisterOp<Reduction>("Reduction");
    RegisterOp<SwapAxis>("SwapAxis");
    RegisterOp<GRU>("GRU");
    RegisterOp<Addn>("Addn");
    RegisterOp<StridedSlice>("StridedSlice");
    RegisterOp<Floor>("Floor");
    RegisterOp<Upsample>("Upsample");
    RegisterOp<Crop>("Crop");
    RegisterOp<Copy>("Copy");
    RegisterOp<Power>("Power");
    RegisterOp<Clip>("Clip");
    RegisterOp<Tile>("Tile");
    RegisterOp<Maximum>("Maximum");
    RegisterOp<Minimum>("Minimum");
    RegisterOp<ArgMax>("ArgMax");
    RegisterOp<ArgMin>("ArgMin");
    RegisterOp<TopKV2>("TopKV2");
    RegisterOp<Reverse>("Reverse");
    RegisterOp<FeatureMatch>("FeatureMatch");
    RegisterOp<ShuffleChannel>("ShuffleChannel");
    RegisterOp<BatchToSpaceND>("BatchToSpaceND");     
    RegisterOp<SpaceToBatchND>("SpaceToBatchND");
    RegisterOp<Absval>("Absval");
    RegisterOp<Hardswish>("Hardswish");
    RegisterOp<Interp>("Interp");
    RegisterOp<Selu>("Selu");
    RegisterOp<L2Normalization>("L2Normalization");
    RegisterOp<L2Pool>("L2Pool");
    RegisterOp<Elu>("Elu");
    RegisterOp<LayerNormLSTM>("LayerNormLSTM");
    RegisterOp<ReLU1>("ReLU1");
    RegisterOp<LogSoftmax>("LogSoftmax");
    RegisterOp<Cast>("Cast");
    RegisterOp<ExpandDims>("ExpandDims");
    RegisterOp<Unary>("Unary");
    RegisterOp<Roialign>("Roialign");   
    RegisterOp<Psroipooling>("Psroipooling");  
    RegisterOp<Bias>("Bias");
    RegisterOp<Noop>("Noop");
    RegisterOp<Threshold>("Threshold");
    RegisterOp<Hardsigmoid>("Hardsigmoid");
    RegisterOp<Embed>("Embedding");
    RegisterOp<InstanceNorm>("InstanceNorm");
    RegisterOp<MVN>("MVN"); 
    RegisterOp<BroadMul>("BroadMul"); 
    RegisterOp<Logical>("Logical"); 
    RegisterOp<Gather>("Gather"); 
    RegisterOp<Transpose>("Transpose");   
    RegisterOp<Comparison>("Comparison"); 
    RegisterOp<SpaceToDepth>("SpaceToDepth");
    RegisterOp<DepthToSpace>("DepthToSpace");
    RegisterOp<Ceil>("Ceil");
    RegisterOp<Round>("Round");
    RegisterOp<SquaredDifference>("SquaredDifference");
    RegisterOp<SparseToDense>("SparseToDense");
    RegisterOp<ZerosLike>("ZerosLike");   
    RegisterOp<MatMul>("MatMul");   
    RegisterOp<ReduceL2>("ReduceL2");
    RegisterOp<Unsqueeze>("Unsqueeze");
    RegisterOp<YOLOV3DetectionOutput>("YOLOV3DetectionOutput");
    // std::cout<<"OPERATOR PLUGIN INITED\n";
    return 0;
}
