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
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/fully_connected.hpp"
#include <math.h>
#include <cblas.h>

namespace TEngine {

namespace FCImpl {

struct FcBlasOps : public NodeOps
{
    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        Tensor* weight_tensor = node->GetInputTensor(1);
        bool has_bias = node->GetInputNum() > 2 ? true : false;

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        float* weight = ( float* )get_tensor_mem(weight_tensor);

        Tensor* bias_tensor;
        float* bias = nullptr;

        if(has_bias)
        {
            bias_tensor = node->GetInputTensor(2);
            bias = ( float* )get_tensor_mem(bias_tensor);
        }

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> in_dims = shape.GetDim();
        const TShape& shape1 = output_tensor->GetShape();
        const std::vector<int> out_dims = shape1.GetDim();

        int batch_number = in_dims[0];
        int inc = in_dims[1];
        int inh =1; 
        int inw =1; 

        if(in_dims.size()>2)
            inh=in_dims[2];

        if(in_dims.size()>3)
            inw=in_dims[3];

        int in_chw = inc * inh * inw;

        /* specially handling on tensorflow models */
        float* converted = nullptr;

        if(exec_attr->model_format == MODEL_FORMAT_TENSORFLOW && (inh * inw > 1))
        {
            converted = ( float* )malloc(batch_number * inc * inh * inw * sizeof(float));

            for(int n = 0; n < batch_number; n++)
            {
                int img_size = inc * inh * inw;

                float* img = converted + n * img_size;
                float* src_img = input + n * img_size;

                for(int c = 0; c < inc; c++)
                    for(int h = 0; h < inh; h++)
                        for(int w = 0; w < inw; w++)
                        {
                            img[h * (inw * inc) + w * inc + c] = src_img[c * inh * inw + h * inw + w];
                        }
            }

            input = converted;
        }

        int outc = out_dims[1];

        int m = batch_number;
        int k = in_chw;
        int n = outc;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, input, k, weight, k, 0, output, n);

        if(has_bias)
        {
            for(int b = 0; b < batch_number; b++)
            {
                float* out_ptr = output + b * outc;
                for(int i = 0; i < outc; ++i)
                {
                    out_ptr[i] += bias[i];
                }
            }
        }

        if(converted)
            free(converted);

        return true;
    }
};

}    // namespace FCImpl

using namespace FCImpl;
void RegisterFcBlasNodeExec(void)
{
    FcBlasOps* ops = new FcBlasOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common", "FullyConnected", ops);
}

}    // namespace TEngine
