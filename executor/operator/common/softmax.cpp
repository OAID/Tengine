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
 *         chunyinglv@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <complex>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/softmax.hpp"
#include "data_type.hpp"

namespace TEngine {

namespace SoftmaxImpl {

struct SoftmaxOps : public NodeOps
{
    template <typename data_type> inline void GetMaxArray(void* input, void* array, int in_size, int on_size)
    {
        data_type* input_ptr = ( data_type* )input;
        data_type* array_ptr = ( data_type* )array;
        std::memset(array, 0, in_size * sizeof(data_type));

        for(int j = 0; j < on_size; j++)
            for(int l = 0; l < in_size; l++)
            {
                if(array_ptr[l] < input_ptr[j * in_size + l])
                    array_ptr[l] = input_ptr[j * in_size + l];
            }
    }

    template <typename data_type>
    inline void GetOutResult(void* input, void* output, void* array, void* sum_array, int in_size, int on_size)
    {
        data_type* input_ptr = ( data_type* )input;
        data_type* output_ptr = ( data_type* )output;
        data_type* array_ptr = ( data_type* )array;
        data_type* sum_array_ptr = ( data_type* )sum_array;

        std::memset(sum_array, 0x0, in_size * sizeof(data_type));

        /* get the exp and the summary */

        for(int j = 0; j < on_size; j++)
            for(int l = 0; l < in_size; l++)
            {
                int index = j * in_size + l;
                output_ptr[index] = std::exp(input_ptr[index] - array_ptr[l]);
                sum_array_ptr[l] += output_ptr[index];
            }

        /* the final result */
        for(int j = 0; j < on_size; j++)
            for(int l = 0; l < in_size; l++)
            {
                int index = j * in_size + l;
                output_ptr[index] /= sum_array_ptr[l];
            }
    }

    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        int element_size = DataType::GetTypeSize(input_tensor->GetDataType());

        const std::vector<int>& dims = input_tensor->GetShape().GetDim();

        //
        Softmax* softmax_op = dynamic_cast<Softmax*>(node->GetOp());
        SoftmaxParam* param_ = softmax_op->GetParam();
        int axis = param_->axis;
        int out_size, in_size, on_size;
        out_size = 1;
        for(int i = 0; i < axis; i++)
        {
            out_size *= dims[i];
        }
        in_size = 1;
        for(size_t i = axis + 1; i < dims.size(); i++)
        {
            in_size *= dims[i];
        }
        on_size = dims[axis];

        uint8_t* input = ( uint8_t* )get_tensor_mem(input_tensor);
        uint8_t* output = ( uint8_t* )get_tensor_mem(output_tensor);
        float* max_array = ( float* )std::malloc(in_size * sizeof(float));
        float* sum_array = ( float* )std::malloc(in_size * sizeof(float));

        int on_in_size = on_size * in_size;

        float* input_f = nullptr;
        float* output_f = nullptr;
        if(element_size == 1)
        {
            input_f = ( float* )std::malloc(on_in_size * 4);
            output_f = ( float* )std::malloc(on_in_size * 4);
        }
        for(int i = 0; i < out_size; i++)
        {
            /* get max */
            int img_base = i * on_in_size * element_size;
            switch(element_size)
            {
                case 4:
                    GetMaxArray<float>(input + img_base, max_array, in_size, on_size);
                    GetOutResult<float>(input + img_base, output + img_base, max_array, sum_array, in_size, on_size);
                    break;
#ifdef CONFIG_FLOAT16
                case 2:
                    GetMaxArray<__fp16>(input + img_base, max_array, in_size, on_size);
                    GetOutResult<__fp16>(input + img_base, output + img_base, max_array, sum_array, in_size, on_size);
                    break;
#endif
                case 1:
                    auto i_quant = input_tensor->GetQuantParam();
                    int i_zero = (*i_quant)[0].zero_point;
                    float i_scale = (*i_quant)[0].scale;
                    auto o_quant = output_tensor->GetQuantParam();
                    int o_zero = (*o_quant)[0].zero_point;
                    float o_scale = (*o_quant)[0].scale;
                    uint8_t* input_cur = input + img_base;
                    uint8_t* output_cur = output + img_base;
                    for(int i = 0; i < on_in_size; i++)
                        input_f[i] = (input_cur[i] - i_zero) * i_scale;

                    GetMaxArray<float>(input_f, max_array, in_size, on_size);
                    GetOutResult<float>(input_f, output_f, max_array, sum_array, in_size, on_size);

                    for(int i = 0; i < on_in_size; i++)
                    {
                        output_cur[i] = std::round(output_f[i] / o_scale) + o_zero;
                    }
                    break;
            }
        }
        if(element_size == 1)
        {
            std::free(input_f);
            std::free(output_f);
        }

        std::free(max_array);
        std::free(sum_array);
        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)
        return nullptr;

    SoftmaxOps* ops = new SoftmaxOps();

    return ops;
}

}    // namespace SoftmaxImpl

using namespace SoftmaxImpl;

void RegisterSoftmaxNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "Softmax", SoftmaxImpl::SelectFunc, 1000);
}

}    // namespace TEngine
