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

    // std::cout<<"OPERATOR PLUGIN INITED\n";
    return 0;
}
