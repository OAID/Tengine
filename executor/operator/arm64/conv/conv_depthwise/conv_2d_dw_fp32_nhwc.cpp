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
 * Author: rzhuang@openailab.com
 */

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <cmath>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "data_type.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"

#include <stdio.h>

#include "dw_kernel_nhwc_fp32.h"

namespace TEngine {

namespace conv_dw_fp32_nhwc {
#define TYPE_A53 0
#define TYPE_A72 1

struct dw_param_3x3
{
    float* input_buf;
    float* output_buf;
    float* weight_buf;
    float* bias;
    int in_w;
    int in_h;
    int h_start;
    int h_end;
    int in_c;
    int out_h;
    int out_w;
    int hw_stride;
    int act;
    int pads[4];
};

const char* conv_name = "CONV_DW_FLOAT_NHWC";
const int default_prio = 5010;

void dw_3x3_run(float* input, float* output, float* kernel, float* bias, int input_w, int input_h, int h_start, int h_end, int input_c,
                int output_h, int output_w, int hw_stride, int pad_w0, int pad_w1, int act)
{
    float* input_ptr = input;
    float* output_ptr = output;
    float* kernel_ptr = kernel;
    float* bias_ptr = bias;

    if(1 == hw_stride)
    {
        if(1 == pad_w0)
        {
            for(int i=h_start; i<h_end; i++)
	    {
                if(i==0)
		    input_ptr = input;
                else
		    input_ptr = input + (i-1) * input_w * input_c;

		output_ptr = output + i * output_w * input_c;

		if(i==0)
                    k3s1p1_nhwc_fp32_hstc(input_ptr, kernel_ptr, output_ptr, bias_ptr, act, input_w, input_h, input_c, output_w, output_h);
		else if(i==output_h-1)
                    k3s1p1_nhwc_fp32_hetc(input_ptr, kernel_ptr, output_ptr, bias_ptr, act, input_w, input_h, input_c, output_w, output_h);
		else
                    k3s1p1_nhwc_fp32_hmtc(input_ptr, kernel_ptr, output_ptr, bias_ptr, act, input_w, input_h, input_c, output_w, output_h);
	    }
        }
        else
            printf("To do support!\n");
    }
    if(2 == hw_stride)
    {
        if(0 == pad_w0 && 1 == pad_w1)
        {
            for(int i=h_start; i<h_end; i++)
	    {
		input_ptr = input + 2 * i * input_w * input_c;

		output_ptr = output + i * output_w * input_c;

		if(i==output_h-1)
                    k3s2p0p1_nhwc_fp32_hetc(input_ptr, kernel_ptr, output_ptr, bias_ptr, act, input_w, input_h, input_c, output_w, output_h);
                else
                    k3s2p0p1_nhwc_fp32_hstc(input_ptr, kernel_ptr, output_ptr, bias_ptr, act, input_w, input_h, input_c, output_w, output_h);
	    }
        }
        else if(1 == pad_w0 && 1 == pad_w1)
        {
            for(int i=h_start; i<h_end; i++)
	    {
                if(i==0)
		    input_ptr = input;
                else
		    input_ptr = input + (2 * i - 1) * input_w * input_c;

		output_ptr = output + i * output_w * input_c;

		if(i==0)
                    k3s2p1_nhwc_fp32_hstc(input_ptr, kernel_ptr, output_ptr, bias_ptr, act, input_w, input_h, input_c, output_w, output_h);
		else if(i==output_h-1)
                    k3s2p1_nhwc_fp32_hetc(input_ptr, kernel_ptr, output_ptr, bias_ptr, act, input_w, input_h, input_c, output_w, output_h);
		else
                    k3s2p1_nhwc_fp32_hmtc(input_ptr, kernel_ptr, output_ptr, bias_ptr, act, input_w, input_h, input_c, output_w, output_h);
	    }
        }
        else
        {
            printf("To do support!\n");
        }
    }
}

struct ConvDw_nhwc_float : public MTNodeOps
{
    ConvDw_nhwc_float()
    {
        name_ = "arm_dw3x3_conv_fp32_nhwc";
    }

    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Reshape(Node* node) override;
    bool Postrun(Node* node) override;
    bool GetSharedMemorySize(Node*, unsigned int& mem_size) override;
    bool SetSharedMemoryAddr(Node*, void* mem_addr, int mem_size) override;
    bool use_dw_3x3;
    bool dynamic_shape;
    bool Aider3x3(int cpu, int seq, void* data);
    void dw_3x3_kernel(float* input, float* output, float* kernel, float* bias, int input_c, int input_h, int input_w,
                       int output_h, int output_w, int stride, int pad_w0, int pad_w1, int act, int cpu_num);
};

bool ConvDw_nhwc_float::Aider3x3(int cpu, int seq, void* data)
{
    dw_param_3x3* param = ( dw_param_3x3* )data;

    int cpu_type = -1;
    if(cpu_info->GetCPUModel(cpu) == CPU_A72)
        cpu_type = TYPE_A72;
    else
        cpu_type = TYPE_A53;

    if(TYPE_A72 == cpu_type || cpu_type == TYPE_A53)
    {
        dw_3x3_run(param->input_buf, param->output_buf, param->weight_buf, param->bias, param->in_w, param->in_h,
                   param->h_start, param->h_end, param->in_c, param->out_h, param->out_w, param->hw_stride,
                   param->pads[0], param->pads[2], param->act);
    }

    return true;
}

void ConvDw_nhwc_float::dw_3x3_kernel(float* input, float* output, float* kernel, float* bias, int input_c, int input_h,
                                      int input_w, int output_h, int output_w, int stride, int pad_w0, int pad_w1,
                                      int act, int cpu_num)
{
    if(1 == cpu_num)
    {
        dw_3x3_run(input, output, kernel, bias, input_w, input_h, 0, output_h, input_c, output_h, output_w, stride, pad_w0, pad_w1, act);
    }
    else
    {
        std::vector<sub_op_task> task_list;
        std::vector<dw_param_3x3> param_list;

        auto f = std::bind(&ConvDw_nhwc_float::Aider3x3, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);

        int task_num = cpu_num;
        task_list.resize(task_num);
        param_list.resize(task_num);
        
	int step = output_h / task_num;

        for(int i = 0; i < task_num; i++)
        {
            dw_param_3x3* param = &param_list[i];
            sub_op_task* task = &task_list[i];

            task->exec_func = f;
            task->seq = i;
            task->data = param;
            param->input_buf = input;
            param->output_buf = output;
            param->weight_buf = kernel;
            param->bias = bias;
            param->in_w = input_w;
            param->in_h = input_h;
            param->in_c = input_c;
            param->h_start = i * step;
            param->h_end = i * step + step;
            param->act = act;
            param->out_h = output_h;
            param->out_w = output_w;
            param->hw_stride = stride;
            param->pads[0] = pad_w0;
            param->pads[1] = pad_w0;
            param->pads[2] = pad_w1;
            param->pads[3] = pad_w1;
        }
        param_list[task_num - 1].h_end += (output_h - step * task_num);
        task_dispatch(task_list, -1);
        wait_done();
    }
}

bool ConvDw_nhwc_float::Reshape(Node* node)
{
    unsigned int new_col_size;

    GetSharedMemorySize(node, new_col_size);

    if(node->ExistAttr("col_buf_allocated"))
    {
        unsigned int col_size = any_cast<unsigned int>(node->GetAttr("col_buf_allocated"));

        if(new_col_size == col_size)
            return true;

        float* addr = any_cast<float*>(node->GetAttr("col_buf"));
        mem_free(addr);
    }

    float* col_buf = ( float* )mem_alloc(new_col_size);
    node->SetAttr("col_buf", col_buf);
    node->SetAttr("col_buf_allocated", new_col_size);
    return true;
}

bool ConvDw_nhwc_float::SetSharedMemoryAddr(Node* node, void* mem_addr, int mem_size)
{
    (*node)["shared_col_buf"] = mem_addr;
    return true;
}

bool ConvDw_nhwc_float::GetSharedMemorySize(Node* node, unsigned int& mem_size)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();

    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    int output_y = output_shape.GetH();
    int output_x = output_shape.GetW();

    Tensor* input_tensor = node->GetInputTensor(0);
    TShape& input_shape = input_tensor->GetShape();

    int input_chan = input_shape.GetC();
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int output_xy = output_x * output_y;

    mem_size = (kernel_size * output_xy);

    return true;
}

bool ConvDw_nhwc_float::Prerun(Node* node)
{
    if(!dynamic_shape)
    {
        if(node->ExistAttr("shared_col_buf"))
        {
            void* addr = any_cast<void*>(node->GetAttr("shared_col_buf"));

            (*node)["col_buf"] = addr;
        }
        else
        {
            unsigned int col_size;
            GetSharedMemorySize(node, col_size);
            void* col_buf = mem_alloc(col_size);
            (*node)["col_buf"] = col_buf;
            node->SetAttr("col_buf_allocated", col_size);
        }
    }
    // Get the input weight and output scale
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();

    use_dw_3x3 = true;
    if(param->kernel_h != 3 || param->kernel_w != 3 || (param->pad_w0 != 0 && param->pad_w0 != 1) ||
       (param->pad_h0 != 0 && param->pad_h0 != 1) || ((1 != param->stride_w) && (2 != param->stride_w)) ||
       (param->stride_w != param->stride_h) || (param->pad_w0 != param->pad_h0) || (param->pad_w1 != param->pad_h1))
    {
        use_dw_3x3 = false;
    }

    return true;
}

bool ConvDw_nhwc_float::Run(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int cpu_number = cpu_info->GetCPUNumber();
    //int cpu_number = 1;
    int pad_w0 = param->pad_w0;
    int pad_w1 = param->pad_w1;
    int act = param->activation;
    Tensor* input_tensor = node->GetInputTensor(0);
    float* input_org = ( float* )get_tensor_mem(input_tensor);
    TShape& input_shape = input_tensor->GetShape();
    int input_w = input_shape.GetW();
    int input_h = input_shape.GetH();
    int input_c = input_shape.GetC();
    int input_n = input_shape.GetN();
    int input_size = input_w * input_h * input_c;

    Tensor* kernel_tensor = node->GetInputTensor(1);
    float* kernel = ( float* )get_tensor_mem(kernel_tensor);

    float* bias_data = nullptr;
    if(node->GetInputNum() > 2)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        bias_data = ( float* )get_tensor_mem(bias_tensor);
    }

    Tensor* output_tensor = node->GetOutputTensor(0);
    float* output_org = ( float* )get_tensor_mem(output_tensor);

    TShape& output_shape = output_tensor->GetShape();
    int output_w = output_shape.GetW();
    int output_h = output_shape.GetH();
    int output_c = output_shape.GetC();
    int output_xy = output_h * output_w;
    if(use_dw_3x3)
    {
        for(int n = 0; n < input_n; n++)
        {
            float* input = input_org + n * input_size;
            float* output = output_org + n * output_xy * output_c;
            dw_3x3_kernel(input, output, kernel, bias_data, input_c, input_h, input_w, output_h, output_w,
                          param->stride_w, pad_w0, pad_w1, act, cpu_number);
        }
    }

    return true;
}

bool ConvDw_nhwc_float::Postrun(Node* node)
{
    if(node->ExistAttr("col_buf_allocated"))
    {
        void* addr = any_cast<void*>(node->GetAttr("col_buf"));
        mem_free(addr);
        node->RemoveAttr("col_buf_allocated");
    }
    use_dw_3x3 = true;
    return true;
}
static bool isDepthwiseSupported(const ConvParam* param, const TShape& input_shape)
{
    int input_c = input_shape.GetC();
    int group = param->group;

    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;

    if(group == 1 || input_c != group || dilation_h != 1 || dilation_w != 1
            || param->kernel_h != 3  || param->kernel_w != 3)
    {
        return false;
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();

    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();

    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NHWC)
        return nullptr;
    
    const TShape& input_shape = node->GetInputTensor(0)->GetShape();

    if(!isDepthwiseSupported(param, input_shape))
        return nullptr;

    ConvDw_nhwc_float* ops = new ConvDw_nhwc_float();

    if(node->IsDynamicShape())
        ops->dynamic_shape = true;
    else
        ops->dynamic_shape = false;

    return ops;
}

}    // namespace conv_dw_fp32_nhwc

void RegisterConv2d_DW_FP32_NHWC(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Convolution", conv_dw_fp32_nhwc::SelectFunc,
                                                      conv_dw_fp32_nhwc::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << conv_dw_fp32_nhwc::default_prio << "]\n";
}

}    // namespace TEngine
