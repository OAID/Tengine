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
 * Author: xiaowei@openailab.com
 */
#include <iostream>
#include <functional>
#include <cstring>

#include "logger.hpp"
#include "operator/fully_connected.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"

#include "op_utils.hpp"

#include "feature_match_kernel/A17.inl"

namespace TEngine {

namespace FC_fast {

int default_prio = 200;

typedef void (*fc_kernel_t)(float* biases, float* input, float* kernel, int kernel_size, float* output);

// interleave most kernels in 8, last 7 interleave in 2, copy the lat kernels
void interleave_kernel(const float* kernel, const float* kernel_interleaved, int kernel_chan, int kernel_size)
{
    int i, j, k;
    float* cur_kernel[8];
    float* cur_kernel_interleaved;

    // interleave 8 kernel
    for(i = 0; i < (kernel_chan & -8); i += 8)
    {
        for(j = 0; j < 8; j++)
            cur_kernel[j] = ( float* )kernel + kernel_size * (i + j);
        cur_kernel_interleaved = ( float* )kernel_interleaved + kernel_size * i;
        for(k = 0; k < kernel_size; k++)
            for(j = 0; j < 8; j++)
                cur_kernel_interleaved[8 * k + j] = *(cur_kernel[j] + k);
    }

    // interleave 2 kernel
    for(; i < (kernel_chan & -2); i += 2)
    {
        for(j = 0; j < 2; j++)
            cur_kernel[j] = ( float* )kernel + kernel_size * (i + j);
        cur_kernel_interleaved = ( float* )kernel_interleaved + kernel_size * i;
        for(k = 0; k < kernel_size; k++)
            for(j = 0; j < 2; j++)
                cur_kernel_interleaved[2 * k + j] = *(cur_kernel[j] + k);
    }

    // copy last kernel
    if(kernel_chan & 0x1)
    {
        cur_kernel[0] = ( float* )kernel + kernel_size * i;
        cur_kernel_interleaved = ( float* )kernel_interleaved + kernel_size * i;
        for(k = 0; k < kernel_size; k++)
            cur_kernel_interleaved[k] = *(cur_kernel[0] + k);
    }

    return;
}

// start and end channel must be 8 aligned
void sgemv1x8(const float* input, float* weight_interleaved, bool have_biases, const float* biases, const float* output,
              int weight_stride, int start_channel, int end_channel, fc_kernel_t kernel_func)
{
    int ch;
    float *cur_kernel, *cur_biases, *cur_result;

    for(ch = (start_channel & -8); ch < (end_channel & -8); ch += 8)
    {
        cur_kernel = ( float* )(weight_interleaved + weight_stride * ch);
        cur_result = ( float* )(output + ch);
        cur_biases = have_biases ? ( float* )(biases + ch) : nullptr;
        kernel_func(cur_biases, ( float* )input, cur_kernel, weight_stride, cur_result);
    }
    return;
}

// start channel must be 2 aligned
void sgemv1x2(const float* input, float* weight_interleaved, bool have_biases, const float* biases, const float* output,
              int weight_stride, int start_channel, int end_channel, fc_kernel_t kernel_func)
{
    float sum;
    int ch;
    float *cur_kernel, *cur_biases, *cur_result;

    for(ch = (start_channel & -2); ch < (end_channel & -2); ch += 2)
    {
        cur_kernel = ( float* )(weight_interleaved + weight_stride * ch);
        cur_result = ( float* )(output + ch);
        cur_biases = have_biases ? ( float* )(biases + ch) : nullptr;
        kernel_func(( float* )cur_biases, ( float* )input, cur_kernel, weight_stride, cur_result);
    }

    if(end_channel & 0x1)
    {
        cur_kernel = ( float* )(weight_interleaved + weight_stride * ch);
        cur_result = ( float* )(output + ch);
        cur_biases = ( float* )(biases + ch);
        sum = have_biases ? *(biases + ch) : 0.0;
        for(int j = 0; j < weight_stride; j++)
            sum += input[j] * cur_kernel[j];
        *cur_result = sum;
    }

    return;
}

struct FCOps : public MTNodeOps
{
    FCOps()
    {
        name_ = "arm_fc_fp32";
    }
    fc_kernel_t kernel_run_1x8 = nullptr;
    fc_kernel_t kernel_run_1x2 = nullptr;

    int cpu_type;

    using sgemv_func_t =
        std::function<void(const float*, float*, bool, const float*, const float*, int, int, int, fc_kernel_t)>;

    struct SgemvParam
    {
        sgemv_func_t func;
        const float* input;
        float* weight_interleaved;
        bool have_biases;
        const float* biases;
        float* output;
        int weight_stride;
        int start_channel;
        int end_channel;
        fc_kernel_t kernel_func;
    };

    bool SgemvAider(int cpu, int aider, void* data)
    {
        SgemvParam* param = ( SgemvParam* )(data);

        param->func(param->input, param->weight_interleaved, param->have_biases, param->biases, param->output,
                    param->weight_stride, param->start_channel, param->end_channel, param->kernel_func);

        return true;
    }

    bool Prerun(Node* node)
    {
        Tensor* tensor;

        tensor = node->GetInputTensor(1);
        int M = tensor->GetShape().Shape(0);
        int K = tensor->GetShape().Shape(1);

        float* weight = ( float* )get_tensor_mem(tensor);

        float* weight_interleaved = ( float* )std::malloc(sizeof(float) * K * M);
        interleave_kernel(weight, weight_interleaved, M, K);

        (*node)["weight_interleaved"] = weight_interleaved;

        if(exec_attr->low_mem_mode)
        {
            tensor->FreeMem();
        }
        
        if(cpu_type == CPU_A17 )
        {   
            kernel_run_1x8 = sgemv_1x8_a17;
            kernel_run_1x2 = sgemv_1x2_a17;
        }

        return true;
    }

    bool Run(Node* node)
    {
        int cpu_number = cpu_info->GetCPUNumber();

        Tensor* tensor;

        /* input */
        tensor = node->GetInputTensor(0);
        float* input = ( float* )get_tensor_mem(tensor);

        const TShape& ishape = tensor->GetShape();

        int batch = ishape.Shape(0);
        int ic = ishape.Shape(1);
        int ih = ishape.Shape(2);
        int iw = ishape.Shape(3);

        float* converted = nullptr;

        if((ih * iw > 1) && exec_attr->model_format == MODEL_FORMAT_TENSORFLOW)
        {
            int img_size = ic * ih * iw;

            converted = ( float* )mem_alloc(batch * img_size * sizeof(float));

            for(int n = 0; n < batch; n++)
            {
                float* src_img = input + n * img_size;
                float* dst_img = converted + n * img_size;

                for(int c = 0; c < ic; c++)
                    for(int h = 0; h < ih; h++)
                        for(int w = 0; w < iw; w++)
                        {
                            dst_img[h * iw * ic + w * ic + c] = src_img[c * ih * iw + h * iw + w];
                        }
            }

            input = converted;
        }

        /* weight */
        tensor = node->GetInputTensor(1);
        int M = tensor->GetShape().Shape(0);
        int K = tensor->GetShape().Shape(1);
        float* weight_interleaved = any_cast<float*>(node->GetAttr("weight_interleaved"));

        /* output */
        tensor = node->GetOutputTensor(0);
        float* output = ( float* )get_tensor_mem(tensor);

        /* biases */
        bool have_biases = (node->GetInputNum() > 2);
        float* biases = have_biases ? ( float* )get_tensor_mem(node->GetInputTensor(2)) : nullptr;

        for(int n = 0; n < batch; n++)
        {
            if(cpu_number == 1 || !exec_attr->fc_mt)
            {
                if(M >= 8)
                    sgemv1x8(input, weight_interleaved, have_biases, biases, output, K, 0, M & -8, kernel_run_1x8);
                if(M & 0x7)
                    sgemv1x2(input, weight_interleaved, have_biases, biases, output, K, M & -8, M, kernel_run_1x2);
            }
            else
            {
                int m8_num = M / 8;

                int step = (m8_num + (cpu_number - 1)) / cpu_number;

                if((m8_num - (cpu_number - 1) * step) <= cpu_number / 2)
                {
                    step = m8_num / cpu_number;
                }

                step = step * 8;

                std::vector<sub_op_task> task_list;
                std::vector<SgemvParam> param_list;

                task_list.resize(cpu_number);
                param_list.resize(cpu_number);

                int start_channel = 0;

                for(int i = 0; i < cpu_number; i++)
                {
                    SgemvParam* param = &param_list[i];
                    sub_op_task* task = &task_list[i];

                    auto f = std::bind(&FCOps::SgemvAider, this, std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3);

                    task->exec_func = f;
                    task->seq = i;
                    task->data = param;

                    param->func = sgemv1x8;
                    param->input = input;
                    param->weight_interleaved = weight_interleaved;
                    param->have_biases = have_biases;
                    param->biases = biases;
                    param->output = output;
                    param->weight_stride = K;
                    param->start_channel = start_channel;
                    param->end_channel = param->start_channel + step;
                    param->kernel_func = kernel_run_1x8;

                    start_channel += step;
                }

                param_list[cpu_number - 1].end_channel = m8_num * 8;

                task_dispatch(task_list, -1);

                if(M & 0x7)
                    sgemv1x2(input, weight_interleaved, have_biases, biases, output, K, M & -8, M, kernel_run_1x2);

                wait_done();
            }

            input += K;
            output += M;
        }

        if(converted)
            mem_free(converted);

        return true;
    }

    bool Postrun(Node* node)
    {
        float* mem = any_cast<float*>(node->GetAttr("weight_interleaved"));
        std::free(mem);

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

    FCOps* ops = new FCOps();

    int master_cpu = cpu_info->GetMasterCPU();

    // if(cpu_info->GetCPUModel(master_cpu) == CPU_A17)
    //     ops->cpu_type = TYPE_A17;
    ops->cpu_type = cpu_info->GetCPUModel(master_cpu);
    return ops;
}

}    // namespace FC_fast

using namespace FC_fast;

void RegisterFullyConnectedFast(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm32", "FullyConnected", FC_fast::SelectFunc,
                                                      FC_fast::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio [" << FC_fast::default_prio << "]\n";
}

}    // namespace TEngine
