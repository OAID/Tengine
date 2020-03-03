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
#include "operator.hpp"
#include "operator/convolution.hpp"
#include "operator/input_op.hpp"
#include "operator/pooling.hpp"
#include "operator/softmax.hpp"
#include "operator/prelu.hpp"

using namespace TEngine;

#define PRINT_PARAM_ENTRY(param, e) std::cout << #e << ":  " << param->e << std::endl;

extern "C" {

int tengine_plugin_init(void);
}

int main(void)
{
    tengine_plugin_init();

    Convolution op;

    op.Input({"weight:float32", "bias:float32", "input:float32"})
        .Output({"output:float32"})
        .SetLayout("NCHW")
        .SetDoc(R"DOC(This is a test)DOC")
        .SetAttr("kernel_h", 3);

    Convolution* conv_op = dynamic_cast<Convolution*>(CREATE_OPERATOR("Convolution"));

    std::cout << conv_op->GetDoc() << std::endl;

    std::cout << any_cast<int>((*conv_op)["kernel_h"]) << std::endl;

    (*conv_op)["kernel_h"] = 100;

    conv_op->ParseDefParam();

    ConvParam* p_param = conv_op->GetParam();

    PRINT_PARAM_ENTRY(p_param, kernel_h);
    PRINT_PARAM_ENTRY(p_param, stride_h);
    PRINT_PARAM_ENTRY(p_param, pad_w);
    PRINT_PARAM_ENTRY(p_param, output_channel);

    Pooling* pool_op = dynamic_cast<Pooling*>(CREATE_OPERATOR("Pooling"));

    std::cout << pool_op->GetDoc() << std::endl;

    PoolParam* pool_param = pool_op->GetParam();

    (*pool_op)["method"] = "avg";

    pool_op->ParseDefParam();

    PRINT_PARAM_ENTRY(pool_param, method);
    PRINT_PARAM_ENTRY(pool_param, alg);
    PRINT_PARAM_ENTRY(pool_param, kernel_h);

    Softmax* softmax_op = dynamic_cast<Softmax*>(CREATE_OPERATOR("Softmax"));

    std::cout << "INPUT NUMBER: " << softmax_op->GetInputNum() << std::endl;

    InputOp* input_op = dynamic_cast<InputOp*>(CREATE_OPERATOR("InputOp"));

    std::cout << "DataLayout: " << input_op->GetLayout() << std::endl;

    // prelu
    PReLU* prelu_op = dynamic_cast<PReLU*>(CREATE_OPERATOR("PReLU"));
    std::cout << prelu_op->GetDoc() << std::endl;
    std::cout << "PReLU INPUT NUMBER: " << prelu_op->GetInputNum() << std::endl;
    return 0;
}
