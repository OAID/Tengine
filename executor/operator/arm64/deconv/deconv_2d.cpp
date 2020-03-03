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
 * Author: xiaowei@openailab.com
 */

#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <math.h>
#include <sys/time.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/deconvolution.hpp"

#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif
#include "op_utils.hpp"

#include "deconv_kernel/A72.inl"


namespace TEngine {

namespace DeconvolutionImpl {

#define TYPE_A53 0
#define TYPE_A72 1

// transpose weight matrix and interleave with 16 and 4
void transpose_weight(float* weight, float* weightT, int weight_w, int weight_h)
{
    int i, j, k;
    int weight_w3 = weight_w & 0x3;

    for(i = 0; i < (weight_w & -16); i += 16)
        for(j = 0; j < weight_h; j++)
            for(k = 0; k < 16; k++)
                *weightT++ = *(weight + j * weight_w + i + k);

    for(; i < (weight_w & -4); i += 4)
        for(j = 0; j < weight_h; j++)
            for(k = 0; k < 4; k++)
                *weightT++ = *(weight + j * weight_w + i + k);

    if(weight_w3)
        for(j = 0; j < weight_h; j++)
        {
            for(k = 0; k < weight_w3; k++)
                *weightT++ = *(weight + j * weight_w + i + k);
            for(; k < 4; k++)
                *weightT++ = 0;
        }

    return;
}

// transpose input matrix and interleave with 4
void transpose_input(float* input, float* inputT, int input_w, int input_h, int start_w, int end_w)
{
    int i, j, k;
    int input_w3 = end_w & 0x3;

    // printf("input_w=%d input_h=%d start=%d end=%d\n",
    //		input_w,input_h,start_w,end_w);

    inputT = inputT + input_h * start_w;

    for(i = start_w; i < (end_w & -4); i += 4)
        for(j = 0; j < input_h; j++)
            for(k = 0; k < 4; k++)
                *inputT++ = *(input + j * input_w + i + k);

    if(input_w3)
        for(j = 0; j < input_h; j++)
        {
            for(k = 0; k < input_w3; k++)
                *inputT++ = *(input + j * input_w + i + k);
            for(; k < 4; k++)
                *inputT++ = 0;
        }

    return;
}

static void sgemm4x16(float* inputT, float* weightT, float* col_buf, int input_ch, int input_wh, int weight_size,
                      int input_start, int input_end, int weight_start, int weight_end, int cpu_type)
{
    float result[64];
    float *cur_input, *cur_weight, *cur_col;
    int input_line, weight_num;
    int i, j;
    int input_end3 = input_end & 0x3;

    for(weight_num = (weight_start & -16); weight_num < (weight_end & -16); weight_num += 16)
    {
        cur_weight = weightT + weight_num * input_ch;
        for(input_line = (input_start & -4); input_line < (input_end & -4); input_line += 4)
        {
            cur_input = inputT + input_line * input_ch;
            cur_col = col_buf + input_line * weight_size + weight_num;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x16_deconv_a72(cur_input, cur_weight, input_ch, cur_col, weight_size);
        }

        if(input_end3)
        {
            cur_input = inputT + input_line * input_ch;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x16_deconv_a72(cur_input, cur_weight, input_ch, result, 16);
            for(j = 0; j < input_end3; j++)
                for(i = 0; i < 16; i++)
                    *(col_buf + (input_line + j) * weight_size + weight_num + i) = result[j * 16 + i];
        }
    }

    return;
}

static void sgemm4x4(float* inputT, float* weightT, float* col_buf, int input_ch, int input_wh, int weight_size,
                     int input_start, int input_end, int weight_start, int weight_end, int cpu_type)
{
    float result[16];
    float *cur_input, *cur_weight, *cur_col;
    int input_line, weight_num;
    int i, j;
    int input_end3 = input_end & 0x3;
    int weight_end3 = weight_end & 0x3;

    for(weight_num = (weight_start & -4); weight_num < (weight_end & -4); weight_num += 4)
    {
        cur_weight = weightT + weight_num * input_ch;
        for(input_line = (input_start & -4); input_line < (input_end & -4); input_line += 4)
        {
            cur_input = inputT + input_line * input_ch;
            cur_col = col_buf + input_line * weight_size + weight_num;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_deconv_a72(cur_input, cur_weight, input_ch, cur_col, weight_size);
        }

        if(input_end3)
        {
            cur_input = inputT + input_line * input_ch;
            
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_deconv_a72(cur_input, cur_weight, input_ch, result, 4);
            for(j = 0; j < input_end3; j++)
                for(i = 0; i < 4; i++)
                    *(col_buf + (input_line + j) * weight_size + weight_num + i) = result[j * 4 + i];
        }
    }

    if(weight_end3)
    {
        cur_weight = weightT + weight_num * input_ch;
        for(input_line = (input_start & -4); input_line < (input_end & -4); input_line += 4)
        {
            cur_input = inputT + input_line * input_ch;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_deconv_a72(cur_input, cur_weight, input_ch, result, 4);
            for(j = 0; j < 4; j++)
                for(i = 0; i < weight_end3; i++)
                    *(col_buf + (input_line + j) * weight_size + weight_num + i) = result[j * 4 + i];
        }

        if(input_end3)
        {
            cur_input = inputT + input_line * input_ch;
            if(cpu_type == TYPE_A72 || 1)
                sgemm_4x4_deconv_a72(cur_input, cur_weight, input_ch, result, 4);
            for(j = 0; j < input_end3; j++)
                for(i = 0; i < weight_end3; i++)
                    *(col_buf + (input_line + j) * weight_size + weight_num + i) = result[j * 4 + i];
        }
    }

    return;
}

void initial_output(float* output, float* bias, int output_ch, int output_wh)
{
    int i, j;
    // no bias
    if(bias == nullptr)
        for(i = 0; i < (output_ch * output_wh); i++)
            output[i] = 0;
    else
        for(i = 0; i < output_ch; i++)
            for(j = 0; j < output_wh; j++)
                *output++ = bias[i];
}

void col2im(float* col, float* im, int output_ch_start, int output_ch_end, int output_ch, int output_x, int output_y,
            int kernel_x, int kernel_y, int stride_x, int stride_y, int dilation_x, int dilation_y, int pad_x,
            int pad_y, int input_x, int input_y)
{
    float* cur_col;
    int imx_start, imy_start, ix, iy, kch, kx, ky, imx, imy;
    int output_xy = output_x * output_y;
    int kernel_xy = kernel_x * kernel_y;
    int weight_size = output_ch * kernel_x * kernel_y;
    bool is_nodilation = (dilation_x == 1 && dilation_y == 1);
    bool is_4x4 = (kernel_x == 4 && kernel_y == 4 && is_nodilation);
    bool is_8x8 = (kernel_x == 8 && kernel_y == 8 && is_nodilation);

    if(is_4x4)
    {
        for(iy = 0; iy < input_y; iy++)
        {
            imy_start = iy * stride_y - pad_y;
            for(ix = 0; ix < input_x; ix++)
            {
                imx_start = ix * stride_x - pad_x;
                cur_col = col + (iy * input_x + ix) * weight_size + 16 * output_ch_start;
                if(iy != 0 && iy != (input_y - 1) && ix != 0 && ix != (input_x - 1))
                    for(kch = output_ch_start; kch < output_ch_end; kch++)
                        for(ky = 0; ky < 4; ky++)
                        {
                            imy = imy_start + ky;
                            for(kx = 0; kx < 4; kx++)
                                *(im + output_xy * kch + output_x * imy + imx_start + kx) += *cur_col++;
                        }
                else
                    for(kch = output_ch_start; kch < output_ch_end; kch++)
                        for(ky = 0; ky < 4; ky++)
                        {
                            imy = imy_start + ky;
                            for(kx = 0; kx < 4; kx++)
                            {
                                imx = imx_start + kx;
                                if(imx >= 0 && imx < output_x && imy >= 0 && imy < output_y)
                                    *(im + output_xy * kch + output_x * imy + imx) += *cur_col;
                                cur_col++;
                            }
                        }
            }
        }
    }
    else if(is_8x8)
    {
        for(iy = 0; iy < input_y; iy++)
        {
            imy_start = iy * stride_y - pad_y;
            for(ix = 0; ix < input_x; ix++)
            {
                imx_start = ix * stride_x - pad_x;
                cur_col = col + (iy * input_x + ix) * weight_size + 64 * output_ch_start;
                if(iy != 0 && iy != (input_y - 1) && ix != 0 && ix != (input_x - 1))
                    for(kch = output_ch_start; kch < output_ch_end; kch++)
                        for(ky = 0; ky < 8; ky++)
                        {
                            imy = imy_start + ky;
                            for(kx = 0; kx < 8; kx++)
                                *(im + output_xy * kch + output_x * imy + imx_start + kx) += *cur_col++;
                        }
                else
                    for(kch = output_ch_start; kch < output_ch_end; kch++)
                        for(ky = 0; ky < 8; ky++)
                        {
                            imy = imy_start + ky;
                            for(kx = 0; kx < 8; kx++)
                            {
                                imx = imx_start + kx;
                                if(imx >= 0 && imx < output_x && imy >= 0 && imy < output_y)
                                    *(im + output_xy * kch + output_x * imy + imx) += *cur_col;
                                cur_col++;
                            }
                        }
            }
        }
    }
    // general case
    else
    {
        for(iy = 0; iy < input_y; iy++)
        {
            imy_start = iy * stride_y - pad_y;
            for(ix = 0; ix < input_x; ix++)
            {
                imx_start = ix * stride_x - pad_x;
                cur_col = col + (iy * input_x + ix) * weight_size + kernel_xy * output_ch_start;
                if(iy != 0 && iy != (input_y - 1) && ix != 0 && ix != (input_x - 1))
                    for(kch = output_ch_start; kch < output_ch_end; kch++)
                        for(ky = 0; ky < kernel_y; ky++)
                        {
                            imy = imy_start + ky * dilation_y;
                            for(kx = 0; kx < kernel_x; kx++)
                            {
                                imx = imx_start + kx * dilation_x;
                                *(im + output_xy * kch + output_x * imy + imx) += *cur_col++;
                            }
                        }
                else
                    for(kch = output_ch_start; kch < output_ch_end; kch++)
                        for(ky = 0; ky < kernel_y; ky++)
                        {
                            imy = imy_start + ky * dilation_y;
                            for(kx = 0; kx < kernel_x; kx++)
                            {
                                imx = imx_start + kx * dilation_x;
                                if(imx >= 0 && imx < output_x && imy >= 0 && imy < output_y)
                                    *(im + output_xy * kch + output_x * imy + imx) += *cur_col;
                                cur_col++;
                            }
                        }
            }
        }
    }

    return;
}

using sgemm_func_t = std::function<void(float*, float*, float*, int, int, int, int, int, int, int, int)>;

struct col2im_param
{
    float* col;
    float* im;
    int start_ch;
    int end_ch;
    int output_ch;
    int output_x;
    int output_y;
    int kernel_x;
    int kernel_y;
    int stride_x;
    int stride_y;
    int dilation_x;
    int dilation_y;
    int pad_x;
    int pad_y;
    int input_x;
    int input_y;
};

struct sgemm_param
{
    sgemm_func_t func;

    float* input;
    float* weightT;
    float* col_buf;
    int input_ch;
    int input_wh;
    int weight_size;
    int input_start;
    int input_end;
    int weight_start;
    int weight_end;
};

struct DeconvolutionOps : public MTNodeOps
{
    DeconvolutionOps()
    {
        name_ = "arm_deconv_fp32";
    }

    bool sgemm_aider(int cpu, int seq, void* data)
    {
        int cpu_type = -1;

        if(cpu_info->GetCPUModel(cpu) == CPU_A72)
            cpu_type = TYPE_A72;
        else if(cpu_info->GetCPUModel(cpu) == CPU_A53)
            cpu_type = TYPE_A53;

        sgemm_param* param = ( sgemm_param* )(data);

        param->func(param->input, param->weightT, param->col_buf, param->input_ch, param->input_wh, param->weight_size,
                    param->input_start, param->input_end, param->weight_start, param->weight_end, cpu_type);

        return true;
    }

    bool col2im_aider(int cpu, int seq, void* data)
    {
        col2im_param* param = ( col2im_param* )(data);

        col2im(param->col, param->im, param->start_ch, param->end_ch, param->output_ch, param->output_x,
               param->output_y, param->kernel_x, param->kernel_y, param->stride_x, param->stride_y, param->dilation_x,
               param->dilation_y, param->pad_x, param->pad_y, param->input_x, param->input_y);

        return true;
    }

    bool Prerun(Node* node)
    {
        // param
        Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(node->GetOp());
        DeconvParam* param_ = deconv_op->GetParam();
        int group = param_->group;
        const Tensor* output_tensor = node->GetOutputTensor(0);
        int output_c = output_tensor->GetShape().GetC()/group;

        // input buffer
        const Tensor* input_tensor = node->GetInputTensor(0);
        const TShape& shape = input_tensor->GetShape();
        const std::vector<int> in_dims = shape.GetDim();
        int input_ch = in_dims[1]/group;
        int input_h = in_dims[2];
        int input_w = in_dims[3];
        int inputT_size = input_ch * ((input_h * input_w + 3) & -4);
        float* inputT = ( float* )std::malloc(sizeof(float) * inputT_size + 128);
        (*node)["inputT"] = inputT;

        // weight
        const Tensor* weight_tensor = node->GetInputTensor(1);
        float* weight = ( float* )get_tensor_mem(weight_tensor);

        int weight_w = param_->kernel_h * param_->kernel_w * output_c;
        int weight_size = ((weight_w + 3) & -4) * input_ch;
        float* weightT = ( float* )std::malloc(sizeof(float) * weight_size * group + 128);

        for(int g = 0; g < group;g++)
        {
            float* weight_g = weight + g * param_->kernel_h * param_->kernel_w * output_c * input_ch;
            float* weightT_g = weightT + g * weight_size;
            transpose_weight(weight_g, weightT_g, weight_w, input_ch);
        }
        (*node)["weightT"] = weightT;

        // get col buffer
        int col_size = input_h * input_w * weight_w;
        float* col_buffer = ( float* )std::malloc(sizeof(float) * col_size + 128);
        (*node)["col_buffer"] = col_buffer;

        return true;
    }

    bool Run(Node* node)
    {
        // input
        const Tensor* input_tensor = node->GetInputTensor(0);
        float* input = ( float* )get_tensor_mem(input_tensor);
        const TShape& in_shape = input_tensor->GetShape();
        const std::vector<int> in_dims = in_shape.GetDim();
        float* inputT = any_cast<float*>(node->GetAttr("inputT"));

        // output
        Tensor* output_tensor = node->GetOutputTensor(0);
        float* output = ( float* )get_tensor_mem(output_tensor);
        const TShape& out_shape = output_tensor->GetShape();
        const std::vector<int> out_dims = out_shape.GetDim();

        // weight
        float* weightT = any_cast<float*>(node->GetAttr("weightT"));

        // bias
        bool have_biases = (node->GetInputNum() > 2);
        float* bias = nullptr;
        if(have_biases)
        {
            bias = ( float* )get_tensor_mem(node->GetInputTensor(2));
        }

        // param
        Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(node->GetOp());
        DeconvParam* param_ = deconv_op->GetParam();
        int pad = param_->pad_w0;
        int stride = param_->stride_h;
        int ksize = param_->kernel_h;
        int dilation = param_->dilation_h;
        int group = param_->group;

        // buffer
        float* col_buffer = any_cast<float*>(node->GetAttr("col_buffer"));

        // shape
        int batch = in_dims[0];
        int input_ch = in_dims[1]/group;
        int input_h = in_dims[2];
        int input_w = in_dims[3];
        int output_ch = out_dims[1]/group;
        int output_h = out_dims[2];
        int output_w = out_dims[3];
        int input_wh = input_h * input_w;
        int input_size = input_ch * input_h * input_w;
        int output_size = output_ch * output_h * output_w;
        int output_wh = output_h * output_w;
        int weight_size = ksize * ksize * output_ch;

        int cpu_type;
        int master_cpu = cpu_info->GetMasterCPU();

        if(cpu_info->GetCPUModel(master_cpu) == CPU_A53)
            cpu_type = TYPE_A53;
        else
            cpu_type = TYPE_A72;

        int L2_CACHE_SIZE = (cpu_type == TYPE_A53) ? 512 * 1024 : 1024 * 1024;
        int input_cnt_l2 = L2_CACHE_SIZE / 4 / input_ch * 7 / 8;
        input_cnt_l2 = input_cnt_l2 > 4 ? (input_cnt_l2 & -4) : 4;

        int weight_size_g = ((weight_size + 3) & -4) * input_ch;
        int cpu_number = cpu_info->GetCPUNumber();

        for(int b = 0; b < batch; ++b)
        {
            float* cur_input = input + b * input_size * group;
            float* cur_output = output + b * output_size * group;

            for(int g = 0; g < group; g ++)
            {
                float* cur_input_g = cur_input + g * input_size;
                float* cur_output_g = cur_output + g * output_size;
                float* cur_weight_g = weightT + g * weight_size_g;

                transpose_input(cur_input_g, inputT, input_wh, input_ch, 0, input_wh);

                /* sgemm part*/

                int l2_loop = (input_wh - 1) / input_cnt_l2 + 1;
                int ch16_num = (weight_size - 1) / 16 + 1;
                int max_task_num = ch16_num * l2_loop;

                if(cpu_number == 1 || max_task_num < 4)
                {
                    for(int input_i = 0; input_i < input_wh; input_i += input_cnt_l2)
                    {
                        int input_start = input_i;
                        int input_end = input_i + input_cnt_l2;
                        input_end = input_end > input_wh ? input_wh : input_end;

                        sgemm4x16(inputT, cur_weight_g, col_buffer, input_ch, input_wh, weight_size, input_start, input_end, 0,
                                  weight_size & -16, cpu_type);
                        if(weight_size & 0xf)
                        {
                            sgemm4x4(inputT, cur_weight_g, col_buffer, input_ch, input_wh, weight_size, input_start, input_end,
                                     weight_size & -16, weight_size, cpu_type);
                        }
                    }
                }
                else
                {
                    std::vector<sub_op_task> task_list;
                    std::vector<sgemm_param> param_list(max_task_num);

                    auto f = std::bind(&DeconvolutionOps::sgemm_aider, this, std::placeholders::_1, std::placeholders::_2,
                                       std::placeholders::_3);

                    for(int input_i = 0; input_i < input_wh; input_i += input_cnt_l2)
                    {
                        int input_start = input_i;
                        int input_end = input_i + input_cnt_l2;

                        input_end = input_end > input_wh ? input_wh : input_end;
                        int real_ch16_num = weight_size / 16;

                        for(int i = 0; i < real_ch16_num; i++)
                        {
                            sub_op_task tmp_task;
                            sub_op_task* task = &tmp_task;
                            sgemm_param* param = &param_list[task_list.size()];

                            task->exec_func = f;
                            task->seq = i;
                            task->data = param;

                            param->func = sgemm4x16;
                            param->input = inputT;
                            param->weightT = cur_weight_g;
                            param->col_buf = col_buffer;
                            param->input_ch = input_ch;
                            param->input_wh = input_wh;
                            param->weight_size = weight_size;
                            param->input_start = input_start;
                            param->input_end = input_end;
                            param->weight_start = i * 16;
                            param->weight_end = param->weight_start + 16;

                            if(param->weight_end > weight_size)
                                param->weight_end = weight_size;

                            task_list.emplace_back(tmp_task);
                        }

                        if(real_ch16_num != ch16_num)
                        {
                            sub_op_task tmp_task;
                            sub_op_task* task = &tmp_task;
                            sgemm_param* param = &param_list[task_list.size()];

                            task->exec_func = f;
                            task->seq = real_ch16_num;
                            task->data = param;

                            param->func = sgemm4x4;
                            param->input = inputT;
                            param->weightT = weightT;
                            param->col_buf = col_buffer;
                            param->input_ch = input_ch;
                            param->input_wh = input_wh;
                            param->weight_size = weight_size;
                            param->input_start = input_start;
                            param->input_end = input_end;
                            param->weight_start = real_ch16_num * 16;
                            param->weight_end = weight_size;

                            task_list.emplace_back(tmp_task);
                        }
                    }

                    task_dispatch(task_list, -1);
                    wait_done();
                }

                /* col2im part */

                float* cur_bias_g = bias ? bias + g : bias;
                initial_output(cur_output_g, cur_bias_g, output_ch, output_wh);

                int ch64_number = (output_ch + 63) / 64;

                if(cpu_number == 1 || ch64_number == 1)
                {
                    col2im(col_buffer, cur_output_g, 0, output_ch, output_ch, output_w, output_h, ksize, ksize, stride,
                           stride, dilation, dilation, pad, pad, input_w, input_h);
                }
                else
                {
                    int real_cpu_number = cpu_number;
                    int steps = ch64_number / cpu_number;

                    while(steps == 0)
                    {
                        real_cpu_number--;
                        steps = ch64_number / real_cpu_number;
                    }

                    // printf("output_ch=%d cpu_number=%d real_cpu_number=%d steps=%d\n",
                    //		output_ch,cpu_number,real_cpu_number,steps*64);

                    std::vector<sub_op_task> col2im_task_list;
                    std::vector<col2im_param> col2im_param_list(cpu_number);

                    auto col2im_func = std::bind(&DeconvolutionOps::col2im_aider, this, std::placeholders::_1,
                                                 std::placeholders::_2, std::placeholders::_3);

                    for(int i = 0; i < real_cpu_number; i++)
                    {
                        col2im_param* param = &col2im_param_list[i];
                        sub_op_task tmp_task;
                        sub_op_task* task = &tmp_task;

                        task->exec_func = col2im_func;
                        task->seq = i;
                        task->data = param;

                        param->col = col_buffer;
                        param->im = cur_output_g;
                        param->start_ch = i * steps * 64;
                        param->end_ch = param->start_ch + steps * 64;

                        if(param->end_ch > output_ch)
                            param->end_ch = output_ch;

                        param->output_ch = output_ch;
                        param->output_x = output_w;
                        param->output_y = output_h;
                        param->kernel_x = ksize;
                        param->kernel_y = ksize;
                        param->stride_x = stride;
                        param->stride_y = stride;
                        param->dilation_x = dilation;
                        param->dilation_y = dilation;
                        param->pad_x = pad;
                        param->pad_y = pad;
                        param->input_x = input_w;
                        param->input_y = input_h;

                        col2im_task_list.emplace_back(tmp_task);
                    }

                    task_dispatch(col2im_task_list, -1);
                    wait_done();
                }
            }
        }
        return true;
    }

    bool Postrun(Node* node)
    {
        float* addr;

        addr = any_cast<float*>(node->GetAttr("inputT"));
        std::free(addr);
        addr = any_cast<float*>(node->GetAttr("weightT"));
        std::free(addr);
        addr = any_cast<float*>(node->GetAttr("col_buffer"));
        std::free(addr);
        return true;
    }
};

static bool isDeconvSupported(DeconvParam* param)
{
    if(param->pad_h0 != param->pad_h1 || param->pad_w0 != param->pad_w1 || param->pad_w0 != param->pad_h0 ||
       param->stride_h != param->stride_w || param->dilation_h != param->dilation_w ||
       param->kernel_h != param->kernel_w)
        return false;
    return true;
}
NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
#ifdef CONFIG_AUTH_DEVICE
    if(!get_auth_float_enabled())
        return nullptr;
#endif
    Operator* op = node->GetOp();
    Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(op);
    DeconvParam* param = deconv_op->GetParam();
    if(!isDeconvSupported(param))
        return nullptr;
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));
    if(exec_attr->graph_layout == TENGINE_LAYOUT_NHWC)
        return nullptr;
    DeconvolutionOps* ops = new DeconvolutionOps();
    return ops;
}

}    // namespace DeconvolutionImpl

using namespace DeconvolutionImpl;

void RegisterDeconvNodeExec(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Deconvolution", DeconvolutionImpl::SelectFunc, 100);
}

}    // namespace TEngine
