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
#include "arm_neon.h"

namespace TEngine {

namespace SoftmaxFP32Impl32 {

const int default_prio = 300;

/*
exp(x) = lim(1+x/n)^n       // n=10
*/
float exp10_f32(float x)
{
    x = 1.0 + x * 0.0009765625f;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    return x;
}
inline float32x4_t vexpq10_f32(float32x4_t x)
{
    x = vmlaq_n_f32(vdupq_n_f32(1.0f), x, 0.0009765625f);    // n = 10
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    return x;
}

struct SoftmaxOps : public MTNodeOps
{
    SoftmaxOps()
    {
        name_ = "arm_softmax_fp32";
    }
    int cpu_number;
    static void GetMaxArray(float* input, float* array, int in_size, int on_size)
    {
        float* input_ptr = ( float* )input;
        float* array_ptr = ( float* )array;
        memset(array, 0, in_size * sizeof(float));

        for(int j = 0; j < on_size; j++)
        {
            for(int i = 0; i < (in_size & -4); i += 4)
            {
                float32x4_t _p = vld1q_f32(array_ptr + i);
                float32x4_t _in = vld1q_f32(input_ptr + j* in_size +i);
                // _p = vpmaxq_f32(_p, _in);
                _p = vmaxq_f32(_p, vrev64q_f32(_in));
                _p = vmaxq_f32(_p, vextq_f32(_p, _in, 2));
                
                vst1q_f32(array_ptr+i, _p);
            }
            for(int i = in_size & ~3; i < in_size; i++)
            {
                if(array_ptr[i] < input_ptr[j * in_size + i])
                    array_ptr[i] = input_ptr[j * in_size + i];
            }
            /*
            for(int l = 0; l < in_size; l++)
            {
                if(array_ptr[l] < input_ptr[j * in_size + l])
                    array_ptr[l] = input_ptr[j * in_size + l];
            }
            */
        }   
    }

    static void GetOutResult(float* input, float* output, float* maxarray, float* sum_array, int in_size, int on_size)
    {
        float* input_ptr = ( float* )input;
        float* output_ptr = ( float* )output;
        float* maxarray_ptr = ( float* )maxarray;
        float* sum_array_ptr = ( float* )sum_array;

        memset(sum_array, 0x0, in_size * sizeof(float));

        /* get the exp and the summary */

        for(int j = 0; j < on_size; j++)
        {
            for(int i = 0; i < (in_size & -4); i += 4)
            {
                int index = j * in_size + i;
                float32x4_t out = vexpq10_f32(vsubq_f32(vld1q_f32(input_ptr + index), vld1q_f32(maxarray_ptr + i)));
                float32x4_t sum = vaddq_f32(vld1q_f32(sum_array_ptr+i), out);
                vst1q_f32(output_ptr+index, out);
                vst1q_f32(sum_array_ptr+i, sum);

            }
            for(int i = in_size & ~3; i < in_size; i++)
            {
                int index = j * in_size + i;
                output_ptr[index] = exp(input_ptr[index] - maxarray_ptr[i]);
                sum_array_ptr[i] += output_ptr[index];
            }
        }
        /*
            for(int l = 0; l < in_size; l++)
            {
                int index = j * in_size + l;
                output_ptr[index] = exp(input_ptr[index] - array_ptr[l]);
                sum_array_ptr[l] += output_ptr[index];
            }
        */
        /* the final result */
        for(int j = 0; j < on_size; j++)
            for(int l = 0; l < in_size; l++)
            {
                int index = j * in_size + l;
                output_ptr[index] /= sum_array_ptr[l];
            }
    }
    void softmax_kernel(const int i, const int tid, const void* step, 
                        int out_size, int on_in_size, int in_size,int on_size,
                        float* input, float* max_array_, float* output,float* sum_array_)
    {
        int my_step = (( int* )step)[0];
        float* max_array = max_array_ + in_size * tid;
        float* sum_array = sum_array_ + in_size * tid;
        for(int idx = tid; idx < out_size; idx += my_step)
        {
            /* get max */
            int img_base = idx * on_in_size;
            GetMaxArray(input + img_base, max_array, in_size, on_size);
            GetOutResult(input + img_base, output + img_base, max_array, sum_array, in_size, on_size);
        }
    }
    bool Run(Node* node)
    {
        Tensor* input_tensor = node->GetInputTensor(0);
        Tensor* output_tensor = node->GetOutputTensor(0);
        const std::vector<int>& dims = input_tensor->GetShape().GetDim();
        Softmax* softmax_op = dynamic_cast<Softmax*>(node->GetOp());
        SoftmaxParam* param_ = softmax_op->GetParam();


        int dim_size = dims.size();
        int axis = param_->axis;
        if(axis > dim_size)
            axis = dim_size - 1;
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

        float* input = ( float* )get_tensor_mem(input_tensor);
        float* output = ( float* )get_tensor_mem(output_tensor);
        float* max_array = ( float* )std::malloc(in_size * cpu_number*sizeof(float));
        float* sum_array = ( float* )std::malloc(in_size * cpu_number*sizeof(float));

        int on_in_size = on_size * in_size;
        if(cpu_number==1)
        {
            for(int i = 0; i < out_size; i++)
            {
                /* get max */
                int img_base = i * on_in_size;
                GetMaxArray(input + img_base, max_array, in_size, on_size);
                GetOutResult(input + img_base, output + img_base, max_array, sum_array, in_size, on_size);
            }
        }
        else
        {
            MULTI_THREAD_START(cpu_number, cpu_number, tid, param_step)
            softmax_kernel(0, tid, param_step, 
                            out_size, on_in_size,in_size,on_size,input,max_array,output,sum_array);
            MULTI_THREAD_END();
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
    ops->cpu_number = cpu_info->GetCPUNumber();

    return ops;
}

}    // namespace SoftmaxFP32Impl32

using namespace SoftmaxFP32Impl32;

void RegisterSoftmaxFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm32", "Softmax", SoftmaxFP32Impl32::SelectFunc, SoftmaxFP32Impl32::default_prio);
}

}    // namespace TEngine
