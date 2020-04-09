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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <sys/time.h>

#include "tensor_mem.hpp"
#include "graph.hpp"
#include "node_ops.hpp"
#include "operator/convolution.hpp"

#include "wino_trans_ker.h"
#include "wino_trans_inp.h"
#include "wino_sgemm.h"

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

namespace TEngine {

namespace conv_2d_wino_nhwc {

// name priority
const char* conv_name = "CONV_WINO_NHWC";

int default_prio = 5200;

static inline unsigned long get_cur_time(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (tv.tv_sec * 1000000 + tv.tv_usec);
}

struct TimeLog
{
    uint64_t trans_inp;
    uint64_t gemm_out;
    int rep;

    TimeLog()
    {
        trans_inp = gemm_out = rep = 0;
    }
    void add_log(uint64_t inp_time, uint64_t gemm_time)
    {
        trans_inp += inp_time;
        gemm_out += gemm_time;
        rep++;
    }
    ~TimeLog()
    {
        const char* env = std::getenv("DEBUG");
        if(rep && env)
        {
            float inp_time = ( float )(trans_inp / 1000.) / rep;
            float gemm_time = ( float )(gemm_out / 1000.) / rep;
            float total = inp_time + gemm_time;
            printf("%.1f (%.2f%%)\t%.1f (%.2f%%)\t%.2f\t%d\n", inp_time, inp_time / total * 100, gemm_time,
                   gemm_time / total * 100, total, rep);
            // printf("------------------------------------------------------------------------------\n");
            // printf("[INP]:%.1f ms (%.2f%%)\t[GEMM_OUT]:%.1f ms (%.2f%%)\t[TOTOAL]:%.2f\t rep=%d\n",
            //         inp_time, inp_time/total, gemm_time,gemm_time/total,total,rep);
            // printf("------------------------------------------------------------------------------\n");
        }
    }
};

TimeLog time_log;

struct wino_sgemm_param
{
    float* ker;
    float* inp;
    float* output;
    float* bias;
    int bias_term;
    int input_c;
    int cpu_type;
    int cout_start;
    int cout_end;
    int block_start;
    int block_end;
    int block_h;
    int block_w;
    int out_c;
    int output_w;
    int resi_h;
    int resi_w;
    int activation;
};
struct wino_transinp_param
{
    float* inp;
    float* trans_inp;

    int inc;
    int nn_block0;
    int nn_block;

    int block_w;
    int in_hw;
    int inw;
};
struct Conv2dWinogradNHWC : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    bool sgemm_aider(int cpu, int seq, void* data);
    bool transinp_aider(int cpu, int seq, void* data);
    int activation;
};
bool Conv2dWinogradNHWC::sgemm_aider(int cpu, int seq, void* data)
{
    wino_sgemm_param* param = ( wino_sgemm_param* )data;

    wino_sgemm_4x16_nhwc(param->ker, param->inp, param->output, param->bias, param->bias_term, param->input_c,
                         param->cpu_type, param->cout_start, param->cout_end, param->block_start, param->block_end,
                         param->block_h, param->block_w, param->out_c, param->output_w, param->resi_h, param->resi_w,
                         param->activation);
    return true;
}
bool Conv2dWinogradNHWC::transinp_aider(int cpu, int seq, void* data)
{
    wino_transinp_param* param = ( wino_transinp_param* )data;

    tran_input_4block(param->inp, param->trans_inp, param->inc, param->nn_block0, param->nn_block,

                      param->block_w, param->in_hw, param->inw);
    return true;
}

// prerun
bool Conv2dWinogradNHWC::Prerun(Node* node)
{
    Tensor* kernel_tensor = node->GetInputTensor(1);
    Tensor* output_tensor = node->GetOutputTensor(0);
    Tensor* input_tensor = node->GetInputTensor(0);

    TShape& input_shape = input_tensor->GetShape();
    TShape& output_shape = output_tensor->GetShape();
    int output_c = output_shape.GetC();
    int input_c = input_shape.GetC();

    int trans_ker_size = output_c * input_c * 36 * sizeof(float);
    float* kernel_org = ( float* )get_tensor_mem(kernel_tensor);
    float* kernel_trans = ( float* )mem_alloc(trans_ker_size);
    float* kernel_interleaved = ( float* )mem_alloc(trans_ker_size + 128);

    // transform & interleave kernel
    ker_nhwc_to_nchw(kernel_org, kernel_interleaved, output_c, input_c);
    //
    transform_kernel_f43_tile(kernel_interleaved, kernel_trans, input_c, output_c);
    interleave_kernel(kernel_trans, kernel_interleaved, output_c, input_c);
    (*node)["kernel_interleaved"] = kernel_interleaved;
    mem_free(kernel_trans);

    if(exec_attr->low_mem_mode)
    {
        kernel_tensor->FreeMem();
    }
    return true;
}

// run
bool Conv2dWinogradNHWC::Run(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;

    int cpu_type;

    if(cpu_info->GetCPUModel(cpu_info->GetMasterCPU()) == CPU_A53)
        cpu_type = TYPE_A53;
    else
        cpu_type = TYPE_A72;

    /* input */
    Tensor* input_tensor = node->GetInputTensor(0);
    const TShape& input_shape = input_tensor->GetShape();
    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();
    int inp_chw = input_c * input_h * input_w;

    /* output */
    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    int output_h = output_shape.GetH();
    int output_w = output_shape.GetW();
    int output_c = output_shape.GetC();
    int out_hw = output_h * output_w;
    int out_chw = out_hw * output_c;
    int output_n = output_shape.GetN();

    int block_h = (output_h + TILE - 1) / TILE;
    int block_w = (output_w + TILE - 1) / TILE;
    int block_hw = block_h * block_w;
    int padded_inh = TILE * block_h + 2;
    int padded_inw = TILE * block_w + 2;
    int pad_inhw = padded_inh * padded_inw;

    int inp_padded_size = sizeof(float) * (input_c * pad_inhw + 2);
    float* input_padded = ( float* )mem_alloc(inp_padded_size);
    float* trans_inp = ( float* )mem_alloc(sizeof(float) * ELEM_SIZE * input_c * block_hw + 128);

    int nn_block = block_hw / BLOCK_HW_UNIT;
    int resi_block = nn_block * BLOCK_HW_UNIT;
    int resi_w = block_w * TILE - output_w;
    int resi_h = block_h * TILE - output_h;
    // printf("nn_block =%d l2_n=%d\n",nn_block,L2_n);

    float* kernel_interleaved = any_cast<float*>(node->GetAttr("kernel_interleaved"));
    float* input_org = ( float* )get_tensor_mem(input_tensor);
    float* output_org = ( float* )get_tensor_mem(output_tensor);
    float* bias = NULL;
    int bias_term = 0;
    if(node->GetInputNum() > 2)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        bias = ( float* )get_tensor_mem(bias_tensor);
        bias_term = 1;
    }

    int L2_CACHE_SIZE = (cpu_type == TYPE_A53) ? 512 * 1024 : 1024 * 1024;
    int L2_n = L2_CACHE_SIZE * 0.3 / (ELEM_SIZE * input_c * sizeof(float));
    L2_n = L2_n > 16 ? (L2_n & -16) : 16;

    int cpu_number = cpu_info->GetCPUNumber();

    for(int n = 0; n < output_n; n++)
    {
        float* input = input_org + n * inp_chw;
        float* output = output_org + n * out_chw;
        // pad_trans_interleave_inp
        long inp_start = get_cur_time();

        convert_pad_input(input, input_padded, input_c, input_h, input_w, padded_inh, padded_inw, pad_h0, pad_w0);
        if(cpu_number == 1)
        {
            tran_input_4block(input_padded, trans_inp, input_c, 0, nn_block, block_w, pad_inhw, padded_inw);
        }
        else
        {
            std::vector<sub_op_task> task_list;
            std::vector<wino_transinp_param> param_list;

            auto f = std::bind(&Conv2dWinogradNHWC::transinp_aider, this, std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3);

            int steps = nn_block / cpu_number;
            steps = (steps + cpu_number - 1) & (~(cpu_number - 1));
            task_list.resize(cpu_number);
            param_list.resize(cpu_number);

            int i = 0;

            while(1)
            {
                wino_transinp_param* param = &param_list[i];
                sub_op_task* task = &task_list[i];

                task->exec_func = f;
                task->seq = i;
                task->data = param;

                param->inp = input_padded;
                param->trans_inp = trans_inp;
                param->inc = input_c;
                param->nn_block0 = i * steps;
                param->nn_block = param->nn_block0 + steps;
                param->block_w = block_w;
                param->in_hw = pad_inhw;
                param->inw = padded_inw;

                if((param->nn_block < nn_block) && (i < cpu_number - 1))
                    i++;
                else
                    break;
            }
            param_list[i].nn_block = nn_block;
            task_list.resize(i + 1);
            task_dispatch(task_list, -1);
            wait_done();
        }
        if(resi_block != block_hw)
            tran_input_resi_block(input_padded, trans_inp, input_c, nn_block, resi_block, block_hw, block_w, pad_inhw,
                                  padded_inw);
        long inp_end = get_cur_time();
        // dot

        std::vector<sub_op_task> task_list;
        std::vector<wino_sgemm_param> param_list;
        int inp_loop = (block_hw - 1) / 4 + 1;
        int ker_loop = (output_c - 1) / L2_n + 1;
        if(cpu_number > 1)
            param_list.resize(ker_loop * inp_loop);

        if(cpu_number == 1)
        {
            wino_sgemm_4x16_nhwc(kernel_interleaved, trans_inp, output, bias, bias_term, input_c, cpu_type, 0, output_c,
                                 0, block_hw, block_h, block_w, output_c, output_w, resi_h, resi_w, activation);
        }
        else
        {
            auto f = std::bind(&Conv2dWinogradNHWC::sgemm_aider, this, std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3);
            for(int p = 0; p < output_c; p += L2_n)
            {
                int cout_start = p;
                int cout_end = p + L2_n;
                cout_end = cout_end > output_c ? output_c : cout_end;

                for(int inp_i = 0; inp_i < block_hw; inp_i += 4)
                {
                    int block_start = inp_i;
                    int block_end = inp_i + 4;
                    block_end = block_end > block_hw ? block_hw : block_end;

                    sub_op_task tmp_task;
                    wino_sgemm_param* param = &param_list[task_list.size()];
                    sub_op_task* task = &tmp_task;
                    task->exec_func = f;
                    task->seq = 0;
                    task->data = param;

                    param->ker = kernel_interleaved;
                    param->inp = trans_inp;
                    param->output = output;
                    param->bias = bias;
                    param->bias_term = bias_term;
                    param->input_c = input_c;
                    param->cpu_type = cpu_type;
                    param->cout_start = cout_start;
                    param->cout_end = cout_end;
                    param->block_start = block_start;
                    param->block_end = block_end;
                    param->block_h = block_h;
                    param->block_w = block_w;
                    param->out_c = output_c;
                    param->output_w = output_w;
                    param->resi_h = resi_h;
                    param->resi_w = resi_w;
                    param->activation = activation;

                    task_list.emplace_back(tmp_task);
                }
            }
            task_dispatch(task_list, -1);
            wait_done();
        }

        long gemm_end = get_cur_time();
        time_log.add_log(inp_end - inp_start, gemm_end - inp_end);
    }
    free(input_padded);
    free(trans_inp);

    return true;
}

// postrun
bool Conv2dWinogradNHWC::Postrun(Node* node)
{
    float* addr;
    if(node->ExistAttr("kernel_interleaved"))
    {
        addr = any_cast<float*>(node->GetAttr("kernel_interleaved"));
        mem_free(addr);
        node->RemoveAttr("kernel_interleaved");
    }
    activation = 0;
    return true;
}

static bool isWinogradSupported(const ConvParam* param, const TShape& output_shape)
{
    int output_c = output_shape.GetC();
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;

    if(group != 1 || kernel_h != 3 || kernel_w != 3 || stride_h != 1 || stride_w != 1 || dilation_h != 1 ||
       dilation_w != 1 || output_c % 16 != 0)
    {
        return false;
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    const char* wino_config = std::getenv("NO_WINO");
    if(wino_config)
        return nullptr;
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

#ifdef CONFIG_AUTH_DEVICE
    bool float_enabled = get_auth_float_enabled();

    if(!float_enabled)
        return nullptr;
#endif

    // datatype = fp32, layout = NCHW
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NHWC)

        return nullptr;

    Operator* op = node->GetOp();
    Convolution* conv_op = dynamic_cast<Convolution*>(op);
    ConvParam* param = conv_op->GetParam();
    const TShape& output_shape = node->GetOutputTensor(0)->GetShape();

    if(!isWinogradSupported(param, output_shape))
        return nullptr;

    Conv2dWinogradNHWC* ops = new Conv2dWinogradNHWC();

    ops->activation = param->activation;

    return ops;
}

}    // namespace conv_2d_wino_nhwc

void RegisterConv2dWinogradNHWC(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Convolution", conv_2d_wino_nhwc::SelectFunc,
                                                  conv_2d_wino_nhwc::default_prio);
}

}    // namespace TEngine
