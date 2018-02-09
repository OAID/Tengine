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
#include "operator/batch_norm.hpp"
#include "operator/scale.hpp"
#include "operator/lrn.hpp"
#include "operator/fused_operator.hpp"
#include "operator/prelu.hpp"
#include "operator/eltwise.hpp"
#include "operator/slice.hpp"
#include "operator/demo_op.hpp"
#include "operator/normalize.hpp"

extern "C" {

   int tengine_plugin_init(void);
}

using namespace TEngine;

int tengine_plugin_init(void)
{
    RegisterOp<Convolution>("Convolution");
    RegisterOp<InputOp>("InputOp");
    RegisterOp<Pooling>("Pooling");
    RegisterOp<SoftMax>("SoftMax");
    RegisterOp<FullyConnected>("FullyConnected");
    RegisterOp<Accuracy>("Accuracy");
    RegisterOp<Concat>("Concat");
    RegisterOp<Dropout>("Dropout");
    RegisterOp<Split>("Split");
    RegisterOp<ConstOp>("Const");
    RegisterOp<ReLu>("ReLu");
    RegisterOp<BatchNorm>(BatchNormName);
    RegisterOp<Scale>("Scale");
    RegisterOp<LRN>("LRN");
    RegisterOp<FusedBNScaleReLu>(FusedBNScaleReLu::class_name);
    RegisterOp<PReLU>("PReLU");
    RegisterOp<Eltwise>("Eltwise");
    RegisterOp<Slice>("Slice");
    RegisterOp<DemoOp>("DemoOp");
    RegisterOp<Normalize>("Normalize");

    std::cout<<"OPERATOR PLUGIN INITED\n";
    return 0;
}

