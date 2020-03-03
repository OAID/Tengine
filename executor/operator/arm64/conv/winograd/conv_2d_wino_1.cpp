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

namespace conv_2d_wino_1 {

// name priority
const char* conv_name = "CONV_WINO_1";

int default_prio = 200;

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
    uint64_t inter_out;
    uint64_t trans_out;
    int rep;

    TimeLog()
    {
        trans_inp = gemm_out = inter_out = trans_out = rep = 0;
    }
    void add_log(uint64_t inp_time, uint64_t gemm_time, uint64_t tran_out)
    {
        trans_inp += inp_time;
        gemm_out += gemm_time;
        trans_out += tran_out;
        rep++;
    }
    ~TimeLog()
    {
        const char* env = std::getenv("DEBUG");
        if(rep && env)
        {
            float inp_time = ( float )(trans_inp / 1000.) / rep;
            float gemm_time = ( float )(gemm_out / 1000.) / rep;
            float out_time = ( float )(trans_out / 1000.) / rep;
            float total = inp_time + gemm_time + out_time;
            printf("%.1f (%.2f%%)\t%.1f (%.2f%%)\t%.1f (%.2f%%)\t%.2f\t%d\n", inp_time, inp_time / total * 100,
                   gemm_time, gemm_time / total * 100, out_time, out_time / total * 100, total, rep);
        }
    }
};

TimeLog time_log;

struct Conv2dWinograd_1 : public MTNodeOps
{
    bool Prerun(Node* node) override;
    bool Run(Node* node) override;
    bool Postrun(Node* node) override;
    int activation;
};


// prerun
bool Conv2dWinograd_1::Prerun(Node* node)
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
    transform_kernel_f43_tile(kernel_org, kernel_trans, input_c, output_c);
    interleave_kernel_1(kernel_trans, kernel_interleaved, output_c, input_c);
    (*node)["kernel_interleaved"] = kernel_interleaved;
    mem_free(kernel_trans);

    if(exec_attr->low_mem_mode)
    {
        kernel_tensor->FreeMem();
    }
    return true;
}
void wino_trans_inp_kernel(const int i, const int tid, const void* step, int input_c, int cin_64,
                          const float* input, float* trans_inp, int block_w, int in_hw, int inw, int block_hw)
{
    int my_step = (( int* )step)[0];

    for(int idx = tid; idx < cin_64; idx += my_step)
    {
        int cin_start = idx * 64;
        int cin_end = cin_start + 64;
        cin_end = cin_end > input_c ? input_c : cin_end;

        tran_input_1(input, trans_inp, input_c, cin_start, cin_end, block_w, in_hw, inw, block_hw);
    }
}

void wino_sgemm_kernel(const int i, const int tid,const void* step,float* kernel_interleaved,float* trans_inp,float* trans_out,
    int input_c,int cpu_type,int cout_nn16,int output_c,int block_hw)
{
    int my_step = ((int*)step)[0];
    for(int s = tid; s < ELEM_SIZE; s+=my_step)
    {
        s = (s<ELEM_SIZE)?(s):(ELEM_SIZE-1);
        wino_sgemm_4x16_1(kernel_interleaved, trans_inp, trans_out, input_c, cpu_type, 0, cout_nn16, 0, block_hw,
                            block_hw, output_c, s);
        if(cout_nn16!=output_c)
        {
            wino_sgemm_4x4_1(kernel_interleaved, trans_inp, trans_out, input_c, cpu_type, cout_nn16, output_c, 0, block_hw,
                            block_hw, output_c, s);
        }
    }
}

void wino_trans_out_kernel(const int i, const int tid,const void* step, int cout_16,int output_c,
                    float* trans_out, float* output, float* bias, int bias_term, int block_h, int block_w,
                    int out_hw, int out_w, int resi_h, int resi_w,int activation)
{
    int my_step = ((int*)step)[0];

    for(int cout_idx = tid; cout_idx < cout_16; cout_idx += my_step)
    {
        int cout_start = cout_idx*16;
        int cout_end = cout_start + 16;
        cout_end = cout_end > output_c ? output_c : cout_end;

        trans_output(trans_out, output, bias, bias_term, block_h, block_w, 
                    cout_start, cout_end, 
                    out_hw, out_w, resi_h, resi_w, activation);
    }
}
// run
bool Conv2dWinograd_1::Run(Node* node)
{
    Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    ConvParam* param = conv_op->GetParam();
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;
    // int pad_h1 = param->pad_h1;
    // int pad_w1 = param->pad_w1;

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

    int cpu_number = cpu_info->GetCPUNumber();
    int cout_count16 = output_c/16;
    int cout_nn16 =  cout_count16*16;

    for(int n = 0; n < output_n; n++)
    {
        float* input_padded = ( float* )mem_alloc(inp_padded_size);
        float* trans_inp = ( float* )mem_alloc(sizeof(float) * ELEM_SIZE * input_c * block_hw + 128);
        float* trans_out = ( float* )mem_alloc(sizeof(float) * ELEM_SIZE * output_c * block_hw);
        float* input = input_org + n * inp_chw;
        float* output = output_org + n * out_chw;
        // pad_trans_interleave_inp
        long inp_start = get_cur_time();

        pad_input1(input, input_padded, input_c, input_h, input_w, padded_inh, padded_inw, pad_h0, pad_w0);
        if(cpu_number == 1)
        {
            tran_input_4block_1(input_padded, trans_inp, input_c, 0, nn_block, block_w, pad_inhw, padded_inw, block_hw);
            if(resi_block != block_hw)
            tran_input_resi_block_1(input_padded, trans_inp, input_c, nn_block, resi_block, block_hw, block_w, pad_inhw,
                                    padded_inw);
        }
        else
        {
            // tran_input_1(input_padded, trans_inp, input_c, 0, input_c, block_w, pad_inhw, padded_inw, block_hw);
            int inc_64 = (input_c + 63) / 64;
            MULTI_THREAD_START(cpu_number, cpu_number, tid, param_step)
            wino_trans_inp_kernel(0, tid, param_step, input_c, inc_64,
                            input_padded, trans_inp, block_w, pad_inhw, padded_inw, block_hw);
            MULTI_THREAD_END();
        }
        
        long inp_end = get_cur_time();
        // dot

        free(input_padded);

        if(cpu_number == 1)
        {
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                wino_sgemm_4x16_1(kernel_interleaved, trans_inp, trans_out, input_c, cpu_type, 0, cout_nn16, 0, block_hw,
                                  block_hw, output_c, s);
                if(cout_nn16!=output_c)
                {
                    wino_sgemm_4x4_1(kernel_interleaved, trans_inp, trans_out, input_c, cpu_type, cout_nn16, output_c, 0, block_hw,
                                  block_hw, output_c, s);
                }
            }
        }
        else
        {
            MULTI_THREAD_START(cpu_number, cpu_number, tid, param_step)
                wino_sgemm_kernel(0,tid, param_step,
                             kernel_interleaved, trans_inp, trans_out, 
                             input_c, cpu_type, cout_nn16, output_c, block_hw);
            MULTI_THREAD_END();
        }

        long gemm_end = get_cur_time();
        free(trans_inp);

        if(cpu_number == 1)
        {
            trans_output(trans_out, output, bias, bias_term, block_h, block_w, 0, output_c, out_hw, output_w, resi_h,
                         resi_w, activation);
        }
        else
        {
            int cout_16 = (output_c+15)/16;
            MULTI_THREAD_START(cpu_number, cpu_number, tid, param_step)
            wino_trans_out_kernel(0,tid, param_step,cout_16, output_c,
                                trans_out, output, bias, bias_term, block_h, block_w, 
                                out_hw, output_w, resi_h,resi_w, activation);
            MULTI_THREAD_END();
        }
        long out_end = get_cur_time();
        time_log.add_log(inp_end - inp_start, gemm_end - inp_end, out_end - gemm_end);
        free(trans_out);
    }

    return true;
}

// postrun
bool Conv2dWinograd_1::Postrun(Node* node)
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

static bool isWinogradSupported(const ConvParam* param, const TShape& input_shape)
{
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int input_c = param->input_channel;
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();
    if((input_h <=6) && (input_w <=6))
        return false;  
    if(group != 1 || kernel_h != 3 || kernel_w != 3 || stride_h != 1 || stride_w != 1 || dilation_h != 1 ||
       dilation_w != 1 || input_c < 256)
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
    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)

        return nullptr;

    Operator* op = node->GetOp();
    Convolution* conv_op = dynamic_cast<Convolution*>(op);
    ConvParam* param = conv_op->GetParam();
    const TShape& input_shape = node->GetInputTensor(0)->GetShape();

    if(!isWinogradSupported(param,input_shape))
        return nullptr;

    Conv2dWinograd_1* ops = new Conv2dWinograd_1();

    ops->activation = param->activation;

    return ops;
}

}    // namespace conv_2d_wino_1

void RegisterConv2dWinograd_1(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Convolution", conv_2d_wino_1::SelectFunc,
                                                  conv_2d_wino_1::default_prio);
}

}    // namespace TEngine
