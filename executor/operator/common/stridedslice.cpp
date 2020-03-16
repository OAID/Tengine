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
#include "operator/stridedslice.hpp"
#include "data_type.hpp"
#include "compiler_fp16.h"

namespace TEngine {

namespace StridedSliceImpl {

struct StridedSliceOps : public NodeOps
{
    template <typename T>
    void kernel_run(T* input, T* output, int batch_num, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w,
                    StridedSliceParam* param)
    {
        int out_chw = out_c * out_h * out_w;
        int out_cw = out_c * out_w;
        int in_chw = in_c * in_h * in_w;
        int in_cw = in_c * in_w;
        for(int n = 0; n < batch_num; n++)
        {
            for(int h = 0; h < out_h; h++)
            {
                for(int w = 0; w < out_w; w++)
                {
                    for(int c = 0; c < out_c; c++)
                    {
                        output[n * out_chw + h * out_cw + w * out_c + c] =
                            input[(param->begin[0] + n * param->stride[0]) * in_chw +
                                  (param->begin[1] + h * param->stride[1]) * in_cw +
                                  (param->begin[2] + w * param->stride[2]) * in_c +
                                  (param->begin[3] + c * param->stride[3])];
                    }
                }
            }
        }
    }
    bool Run(Node* node) override
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);

        //    float * input=(float *)get_tensor_mem(input_tensor);
        //    float * output=(float *)get_tensor_mem(output_tensor);

        StridedSlice* slice_op = dynamic_cast<StridedSlice*>(node->GetOp());
        StridedSliceParam* param = slice_op->GetParam();

        const TShape& shape = input_tensor->GetShape();
        const std::vector<int>& in_dim = shape.GetDim();

        const TShape& shape1 = output_tensor->GetShape();
        const std::vector<int> &out_dim_tmp = shape1.GetDim(); 
        std::vector<int> out_dim;
        
        if(in_dim.size() == 2){
            out_dim.push_back(1);
            out_dim.push_back(1);
            out_dim.push_back(out_dim_tmp[0]);
            out_dim.push_back(out_dim_tmp[1]);
        }else if(in_dim.size() == 3){
            out_dim.push_back(out_dim_tmp[0]);
            out_dim.push_back(1);
            out_dim.push_back(out_dim_tmp[1]);
            out_dim.push_back(out_dim_tmp[2]); 
        }
        else {
            out_dim = out_dim_tmp;
        }


        int element_size = DataType::GetTypeSize(input_tensor->GetDataType());
        if(1 == element_size)
        {
            int8_t* input = ( int8_t* )get_tensor_mem(input_tensor);
            int8_t* output = ( int8_t* )get_tensor_mem(output_tensor);
            kernel_run<int8_t>(input, output, out_dim[0], in_dim[3], in_dim[1], in_dim[2], out_dim[3], out_dim[1],
                               out_dim[2], param);
            auto* in_quant = input_tensor->GetQuantParam();
            auto* out_quant = output_tensor->GetQuantParam();
            QuantParam q_param;
            q_param.scale = (*in_quant)[0].scale;
            q_param.zero_point = 0;
            out_quant->resize(0);
            out_quant->push_back(q_param);
        }
        if(2 == element_size)
        {
            __fp16* input = ( __fp16* )get_tensor_mem(input_tensor);
            __fp16* output = ( __fp16* )get_tensor_mem(output_tensor);
            kernel_run<__fp16>(input, output, out_dim[0], in_dim[3], in_dim[1], in_dim[2], out_dim[3], out_dim[1],
                               out_dim[2], param);
        }
        if(4 == element_size)
        {
            float* input = ( float* )get_tensor_mem(input_tensor);
            float* output = ( float* )get_tensor_mem(output_tensor);
            kernel_run<float>(input, output, out_dim[0], in_dim[3], in_dim[1], in_dim[2], out_dim[3], out_dim[1],
                              out_dim[2], param);
        }
#if 0
        int out_w   = out_dim[3];
        int out_hw  = out_dim[2] * out_w;
        int out_chw = out_dim[1] * out_hw;

        int in_w   = in_dim[3];
        int in_hw  = in_dim[2] * in_w;
        int in_chw = in_dim[1] * in_hw;

        for(int n=0;n<out_dim[0];n++)
        {
            for(int c=0;c<out_dim[1];c++)
            {
                for(int h=0;h<out_dim[2];h++)
                {
                    for(int w=0;w<out_dim[3];w++)
                    {
                        output[n*out_chw + c*out_hw + h*out_w + w ] =
                            input[(param->begin[0]+ n*param->stride[0])*in_chw +
                                (param->begin[1]+ c*param->stride[1])*in_hw +
                                (param->begin[2]+ h*param->stride[2])*in_w +
                                (param->begin[3]+ w*param->stride[3])];
                    }
                }
            }
        }
#endif

        return true;
    }
};

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if((data_type != TENGINE_DT_FP32 && data_type != TENGINE_DT_FP16 && data_type != TENGINE_DT_INT8) ||
       exec_attr->graph_layout != TENGINE_LAYOUT_NHWC)
        return nullptr;

    StridedSliceOps* ops = new StridedSliceOps();

    return ops;
}

}    // namespace StridedSliceImpl

using namespace StridedSliceImpl;

void RegisterStridedSliceNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("common", "StridedSlice", StridedSliceImpl::SelectFunc, 1000);
}

}    // namespace TEngine
