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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: bhu@openailab.com
 * update：qtang@openailab.com
 */

#include "convolution_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>


static int ref_conv_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* kernel,
                         struct tensor* bias, struct conv_param* conv_param)
{
    int batch = input_tensor->dims[0];
    int group = conv_param->group;
    int input_c = conv_param->input_channel / group;
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];
    int output_c = output_tensor->dims[1] / group;
    int output_h = output_tensor->dims[2];
    int output_w = output_tensor->dims[3];

    int kernel_size = input_c * conv_param->kernel_h * conv_param->kernel_w;
    int n, g, c, h, w, kc, kh, kw;
    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;

    float* input_data = input_tensor->data;
    float* output_data = output_tensor->data;
    float* kernel_data = kernel->data;
    float* bias_data = NULL;
    if (bias != NULL)
        bias_data = bias->data;

    if (conv_param->kernel_h == 0)
        conv_param->kernel_h = 1;
    if (conv_param->kernel_w == 0)
        conv_param->kernel_w = 1;
    if (input_w == 0)
        input_w = 1;

    for (n = 0; n < batch; ++n)
    {
        for (g = 0; g < group; ++g)
        {
            for (c = 0; c < output_c; ++c)
            {
                for (h = 0; h < output_h; ++h)
                {
                    for (w = 0; w < output_w; ++w)
                    {
                        const int h_start = (h * conv_param->stride_h) - conv_param->pad_h0;
                        const int w_start = (w * conv_param->stride_w) - conv_param->pad_w0;
                        float total = 0.f;
                        if (input_tensor->layout == 0)
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            g * output_c * output_h * output_w + c * output_h * output_w +
                                            h * output_w + w;
                        }
                        else
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            h * output_w * group * output_c + w * group * output_c + output_c * g + c;
                        }
                        for (kc = 0; kc < input_c; ++kc)
                        {
                            for (kh = 0; kh < conv_param->kernel_h; ++kh)
                            {
                                for (kw = 0; kw < conv_param->kernel_w; ++kw)
                                {
                                    const int cur_y = h_start + conv_param->dilation_h * kh;
                                    const int cur_x = w_start + conv_param->dilation_w * kw;
                                    // If the location is outside the bounds of the input image,
                                    // use zero as a default value.
                                    if ((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) && (cur_y < input_h))
                                    {
                                        if (input_tensor->layout == 0)
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           g * input_c * input_h * input_w + kc * input_h * input_w +
                                                           cur_y * input_w + cur_x;
                                            kernel_offset = g * output_c * kernel_size + c * kernel_size +
                                                            kc * conv_param->kernel_h * conv_param->kernel_w +
                                                            kh * conv_param->kernel_w + kw;
                                        }
                                        else
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           cur_y * input_w * input_c * group + cur_x * input_c * group +
                                                           g * input_c + kc;
                                            kernel_offset = c * group * kernel_size +
                                                            kh * conv_param->kernel_w * input_c * group +
                                                            kw * input_c * group + g * input_c + kc;
                                        }

                                        total += input_data[input_offset] * kernel_data[kernel_offset];
                                    }
                                }
                            }
                        }

                        if (bias != NULL)
                            total += bias_data[output_c * g + c];

                        if (conv_param->activation >= 0)
                        {
                            if (total < 0 && conv_param->activation != 1)
                            {
                                total = 0;
                            }
                            if (total > 1 && conv_param->activation == 1)
                            {
                                total = 1;
                            }
                            if (total > 6 && conv_param->activation == 6)
                            {
                                total = 6;
                            }
                            if (total < -1 && conv_param->activation == 1)
                            {
                                total = -1;
                            }
                        }
                        output_data[output_offset] = total;
                    }
                }
            }
        }
    }

    return 0;
}


static int ref_conv_fp16(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* kernel,
                         struct tensor* bias, struct conv_param* conv_param)
{
#if MACOS
    TLOG_ERR("FP16 not support under mac os");
#else
    int batch = input_tensor->dims[0];
    int group = conv_param->group;
    int input_c = conv_param->input_channel / group;
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];
    int output_c = output_tensor->dims[1] / group;
    int output_h = output_tensor->dims[2];
    int output_w = output_tensor->dims[3];

    int kernel_size = input_c * conv_param->kernel_h * conv_param->kernel_w;
    int n, g, c, h, w, kc, kh, kw;
    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;

    fp16_t* input_data = input_tensor->data;
    fp16_t* output_data = output_tensor->data;
    fp16_t* kernel_data = kernel->data;
    fp16_t* bias_data = NULL;
    if (bias != NULL)
        bias_data = bias->data;

    if (conv_param->kernel_h == 0)
        conv_param->kernel_h = 1;
    if (conv_param->kernel_w == 0)
        conv_param->kernel_w = 1;
    if (input_w == 0)
        input_w = 1;

    for (n = 0; n < batch; ++n)
    {
        for (g = 0; g < group; ++g)
        {
            for (c = 0; c < output_c; ++c)
            {
                for (h = 0; h < output_h; ++h)
                {
                    for (w = 0; w < output_w; ++w)
                    {
                        const int h_start = (h * conv_param->stride_h) - conv_param->pad_h0;
                        const int w_start = (w * conv_param->stride_w) - conv_param->pad_w0;
                        float total = 0.f;
                        if (input_tensor->layout == 0)
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            g * output_c * output_h * output_w + c * output_h * output_w +
                                            h * output_w + w;
                        }
                        else
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            h * output_w * group * output_c + w * group * output_c + output_c * g + c;
                        }
                        for (kc = 0; kc < input_c; ++kc)
                        {
                            for (kh = 0; kh < conv_param->kernel_h; ++kh)
                            {
                                for (kw = 0; kw < conv_param->kernel_w; ++kw)
                                {
                                    const int cur_y = h_start + conv_param->dilation_h * kh;
                                    const int cur_x = w_start + conv_param->dilation_w * kw;
                                    // If the location is outside the bounds of the input image,
                                    // use zero as a default value.
                                    if ((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) && (cur_y < input_h))
                                    {
                                        if (input_tensor->layout == 0)
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           g * input_c * input_h * input_w + kc * input_h * input_w +
                                                           cur_y * input_w + cur_x;
                                            kernel_offset = g * output_c * kernel_size + c * kernel_size +
                                                            kc * conv_param->kernel_h * conv_param->kernel_w +
                                                            kh * conv_param->kernel_w + kw;
                                        }
                                        else
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           cur_y * input_w * input_c * group + cur_x * input_c * group +
                                                           g * input_c + kc;
                                            kernel_offset = c * group * kernel_size +
                                                            kh * conv_param->kernel_w * input_c * group +
                                                            kw * input_c * group + g * input_c + kc;
                                        }

                                        total += fp16_to_fp32(input_data[input_offset]) *
                                                 fp16_to_fp32(kernel_data[kernel_offset]);
                                    }
                                }
                            }
                        }

                        if (bias != NULL)
                            total += fp16_to_fp32(bias_data[output_c * g + c]);

                        if (conv_param->activation >= 0)
                        {
                            if (total < 0 && conv_param->activation != 1)
                            {
                                total = 0;
                            }
                            if (total > 1 && conv_param->activation == 1)
                            {
                                total = 1;
                            }
                            if (total > 6 && conv_param->activation == 6)
                            {
                                total = 6;
                            }
                            if (total < -1 && conv_param->activation == 1)
                            {
                                total = -1;
                            }
                        }
                        output_data[output_offset] = fp32_to_fp16(total);
                    }
                }
            }
        }
    }
#endif
    return 0;
}

static int ref_conv_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* kernel,
                          struct tensor* bias, struct conv_param* conv_param)
{
    int batch = input_tensor->dims[0];
    int group = conv_param->group;
    int input_c = conv_param->input_channel / group;
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];
    int output_c = output_tensor->dims[1] / group;
    int output_h = output_tensor->dims[2];
    int output_w = output_tensor->dims[3];

    int kernel_size = input_c * conv_param->kernel_h * conv_param->kernel_w;
    int n, g, c, h, w, kc, kh, kw;
    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;

    uint8_t* input_data = input_tensor->data;
    uint8_t* output_data = output_tensor->data;
    uint8_t* kernel_data = kernel->data;
    int32_t* bias_data = NULL;
    if (bias != NULL)
        bias_data = bias->data;

    float input_scale = input_tensor->scale;
    float kernel_scale = kernel->scale;
    float output_scale = output_tensor->scale;
    int32_t kernel_zero = kernel->zero_point;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    /* dequant input  */
    int input_size = batch * group * input_c * input_h * input_w;
    float* input_fp32 = ( float* )sys_malloc(sizeof(float) * input_size);
    for (int i = 0; i < input_size; i++)
        input_fp32[i] = (( float )input_data[i] - input_zero) * input_scale;

    /* dequant kernel  */
    int kernel_total = group * output_c * kernel_size;
    float* kernel_fp32 = ( float* )sys_malloc(sizeof(float) * kernel_total);
    for (int i = 0; i < kernel_total; i++)
        kernel_fp32[i] = (( float )kernel_data[i] - kernel_zero) * kernel_scale;

    /* dequant biases  */
    int bias_size = group * output_c;

    float* bias_fp32 = NULL;
    if (bias != NULL)
    {
        bias_fp32 = ( float* )sys_malloc(sizeof(float) * bias_size);
        for (int i = 0; i < bias_size; i++)
            bias_fp32[i] = ( float )bias_data[i] * input_scale * kernel_scale;
    }

    if (conv_param->kernel_h == 0)
        conv_param->kernel_h = 1;
    if (conv_param->kernel_w == 0)
        conv_param->kernel_w = 1;
    if (input_w == 0)
        input_w = 1;

    for (n = 0; n < batch; ++n)
    {
        for (g = 0; g < group; ++g)
        {
            for (c = 0; c < output_c; ++c)
            {
                for (h = 0; h < output_h; ++h)
                {
                    for (w = 0; w < output_w; ++w)
                    {
                        const int h_start = (h * conv_param->stride_h) - conv_param->pad_h0;
                        const int w_start = (w * conv_param->stride_w) - conv_param->pad_w0;
                        float total = 0.f;
                        if (input_tensor->layout == 0)
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            g * output_c * output_h * output_w + c * output_h * output_w +
                                            h * output_w + w;
                        }
                        else
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            h * output_w * group * output_c + w * group * output_c + output_c * g + c;
                        }
                        for (kc = 0; kc < input_c; ++kc)
                        {
                            for (kh = 0; kh < conv_param->kernel_h; ++kh)
                            {
                                for (kw = 0; kw < conv_param->kernel_w; ++kw)
                                {
                                    const int cur_y = h_start + conv_param->dilation_h * kh;
                                    const int cur_x = w_start + conv_param->dilation_w * kw;
                                    // If the location is outside the bounds of the input image,
                                    // use zero as a default value.
                                    if ((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) && (cur_y < input_h))
                                    {
                                        if (input_tensor->layout == 0)
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           g * input_c * input_h * input_w + kc * input_h * input_w +
                                                           cur_y * input_w + cur_x;
                                            kernel_offset = g * output_c * kernel_size + c * kernel_size +
                                                            kc * conv_param->kernel_h * conv_param->kernel_w +
                                                            kh * conv_param->kernel_w + kw;
                                        }
                                        else
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           cur_y * input_w * input_c * group + cur_x * input_c * group +
                                                           g * input_c + kc;
                                            kernel_offset = c * group * kernel_size +
                                                            kh * conv_param->kernel_w * input_c * group +
                                                            kw * input_c * group + g * input_c + kc;
                                        }

                                        total += input_fp32[input_offset] * kernel_fp32[kernel_offset];
                                    }
                                }
                            }
                        }

                        if (bias != NULL)
                            total += bias_fp32[output_c * g + c];

                        if (conv_param->activation >= 0)
                        {
                            if (total < 0 && conv_param->activation != 1)
                            {
                                total = 0;
                            }
                            if (total > 1 && conv_param->activation == 1)
                            {
                                total = 1;
                            }
                            if (total > 6 && conv_param->activation == 6)
                            {
                                total = 6;
                            }
                            if (total < -1 && conv_param->activation == 1)
                            {
                                total = -1;
                            }
                        }

                        int out = round(total / output_scale) + output_zero;
                        if (out > 255)
                            out = 255;
                        if (out < 0)
                            out = 0;
                        output_data[output_offset] = out;
                    }
                }
            }
        }
    }

    sys_free(input_fp32);
    sys_free(kernel_fp32);
    if (bias != NULL)
        sys_free(bias_fp32);

    return 0;
}


static int ref_conv_int8(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* kernel,
                         struct tensor* bias, struct conv_param* conv_param)
{
    int batch = input_tensor->dims[0];
    int group = conv_param->group;
    int input_c = conv_param->input_channel / group;
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];
    int output_c = output_tensor->dims[1] / group;
    int output_h = output_tensor->dims[2];
    int output_w = output_tensor->dims[3];

    int kernel_size = input_c * conv_param->kernel_h * conv_param->kernel_w;
    int n, g, c, h, w, kc, kh, kw;
    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;

    int8_t* input_i8 = input_tensor->data;
    int8_t* output_i8 = output_tensor->data;
    int8_t* kernel_i8 = kernel->data;
    int32_t* bias_i32 = NULL;
    if (bias != NULL)
        bias_i32 = bias->data;

    float input_scale = input_tensor->scale;
    float* kernel_scales = kernel->scale_list;
    float output_scale = output_tensor->scale;

    /* input and kernel scales */
    int dequant_scales_size = group * output_c;
    float *dequant_scales = (float*)malloc(sizeof(float) * dequant_scales_size);

    for(int i = 0; i < dequant_scales_size; i++)
    {
        dequant_scales[i] = (input_scale * kernel_scales[i]);
    }

    if (conv_param->kernel_h == 0)
        conv_param->kernel_h = 1;
    if (conv_param->kernel_w == 0)
        conv_param->kernel_w = 1;
    if (input_w == 0)
        input_w = 1;

    for (n = 0; n < batch; ++n)
    {
        for (g = 0; g < group; ++g)
        {
            for (c = 0; c < output_c; ++c)
            {
                for (h = 0; h < output_h; ++h)
                {
                    for (w = 0; w < output_w; ++w)
                    {
                        const int h_start = (h * conv_param->stride_h) - conv_param->pad_h0;
                        const int w_start = (w * conv_param->stride_w) - conv_param->pad_w0;
                        int32_t total_i32 = 0;
                        if (input_tensor->layout == 0)
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            g * output_c * output_h * output_w + c * output_h * output_w +
                                            h * output_w + w;
                        }
                        else
                        {
                            output_offset = n * group * output_c * output_h * output_w +
                                            h * output_w * group * output_c + w * group * output_c + output_c * g + c;
                        }
                        for (kc = 0; kc < input_c; ++kc)
                        {
                            for (kh = 0; kh < conv_param->kernel_h; ++kh)
                            {
                                for (kw = 0; kw < conv_param->kernel_w; ++kw)
                                {
                                    const int cur_y = h_start + conv_param->dilation_h * kh;
                                    const int cur_x = w_start + conv_param->dilation_w * kw;
                                    // If the location is outside the bounds of the input image,
                                    // use zero as a default value.
                                    if ((cur_x >= 0) && (cur_x < input_w) && (cur_y >= 0) && (cur_y < input_h))
                                    {
                                        if (input_tensor->layout == 0)
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           g * input_c * input_h * input_w + kc * input_h * input_w +
                                                           cur_y * input_w + cur_x;
                                            kernel_offset = g * output_c * kernel_size + c * kernel_size +
                                                            kc * conv_param->kernel_h * conv_param->kernel_w +
                                                            kh * conv_param->kernel_w + kw;
                                        }
                                        else
                                        {
                                            input_offset = n * group * input_c * input_h * input_w +
                                                           cur_y * input_w * input_c * group + cur_x * input_c * group +
                                                           g * input_c + kc;
                                            kernel_offset = c * group * kernel_size +
                                                            kh * conv_param->kernel_w * input_c * group +
                                                            kw * input_c * group + g * input_c + kc;
                                        }

                                        total_i32 += (int32_t)input_i8[input_offset] * (int32_t)kernel_i8[kernel_offset];
                                    }
                                }
                            }
                        }

                        if (bias != NULL)
                            total_i32 += bias_i32[output_c * g + c];

                        float total = total_i32 * dequant_scales[output_c * g + c];

                        if (conv_param->activation >= 0)
                        {
                            if (total < 0 && conv_param->activation != 1)
                            {
                                total = 0;
                            }
                            if (total > 1 && conv_param->activation == 1)
                            {
                                total = 1;
                            }
                            if (total > 6 && conv_param->activation == 6)
                            {
                                total = 6;
                            }
                            if (total < -1 && conv_param->activation == 1)
                            {
                                total = -1;
                            }
                        }

                        int out = round(total / output_scale);
                        if (out > 127)
                            out = 127;
                        if (out < -127)
                            out = -127;
                        output_i8[output_offset] = (uint8_t)out;
                    }
                }
            }
        }
    }

    sys_free(dequant_scales);

    return 0;
}

// add conv op by wangxinwei for debug conv
//======================================================================================================//

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* bias_tensor = NULL;
    struct tensor* output_tensor = NULL;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    if (ir_node->input_num > 2)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    }
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    int ret = 0;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_conv_fp32(input_tensor, output_tensor, weight_tensor, bias_tensor, conv_param);
    else if (input_tensor->data_type == TENGINE_DT_FP16)
        ret = ref_conv_fp16(input_tensor, output_tensor, weight_tensor, bias_tensor, conv_param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_conv_uint8(input_tensor, output_tensor, weight_tensor, bias_tensor, conv_param);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_conv_int8(input_tensor, output_tensor, weight_tensor, bias_tensor, conv_param);
    else
    {
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);
        return -1;
    }

    return ret;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct conv_param* conv_param = ( struct conv_param* )ir_node->op.param_mem;

    /* dynamic get the shape of output tensor */
    int n = input_tensor->dims[0];
    int h, w;
    int ret = 0;

    if (conv_param->kernel_w == 0)
    {
        conv_param->kernel_w = 1;
        conv_param->pad_w0 = 0;
        conv_param->pad_w1 = 0;
    }
    if (conv_param->kernel_h == 0)
        conv_param->kernel_h = 1;
    if (conv_param->stride_w == 0)
        conv_param->stride_w = 1;
    if (conv_param->stride_h == 0)
        conv_param->stride_h = 1;

    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        h = input_tensor->dims[2];
        w = input_tensor->dims[3];
    }
    else if (ir_graph->graph_layout == TENGINE_LAYOUT_NHWC)
    {
        h = input_tensor->dims[1];
        w = input_tensor->dims[2];
    }
    else
    {
        TLOG_ERR("convolution infer shape: unknown graph layout: %d\n", ir_graph->graph_layout);
        return -1;
    }

    int out_c = conv_param->output_channel;
    int out_h, out_w;

    /* handle the same padding case, which pad_h0 and pad_h1 is -1 (SAME_UPPER)
        -2 (SAME_LOWER) */

    if (conv_param->pad_h0 < 0)
    {
        out_h = (h - 1) / conv_param->stride_h + 1;

        int total_len = (out_h - 1) * conv_param->stride_h + conv_param->kernel_h;
        int pad_num = total_len - h;

        if (conv_param->pad_h0 == -1)
        {
            conv_param->pad_h0 = pad_num / 2;
            conv_param->pad_h1 = pad_num - pad_num / 2;
        }
        else
        {
            conv_param->pad_h1 = pad_num / 2;
            conv_param->pad_h0 = pad_num - pad_num / 2;
        }
    }
    else
    {
        out_h =
                (h - conv_param->dilation_h * (conv_param->kernel_h - 1) - 1 + conv_param->pad_h0 + conv_param->pad_h1) /
                conv_param->stride_h +
                1;
    }

    if (conv_param->pad_w0 < 0)
    {
        out_w = (w - 1) / conv_param->stride_w + 1;

        int total_len = (out_w - 1) * conv_param->stride_w + conv_param->kernel_w;
        int pad_num = total_len - w;

        if (conv_param->pad_w0 == -1)
        {
            conv_param->pad_w0 = pad_num / 2;
            conv_param->pad_w1 = pad_num - pad_num / 2;
        }
        else
        {
            conv_param->pad_w1 = pad_num / 2;
            conv_param->pad_w0 = pad_num - pad_num / 2;
        }
    }
    else
    {
        out_w =
                (w - conv_param->dilation_w * (conv_param->kernel_w - 1) - 1 + conv_param->pad_w0 + conv_param->pad_w1) /
                conv_param->stride_w +
                1;
    }

    int dims[4];
    dims[0] = n;
    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        if (output_tensor->dims[1] != out_c || output_tensor->dims[2] != out_h || output_tensor->dims[3] != out_w)
        {
            dims[1] = out_c;
            dims[2] = out_h;
            dims[3] = out_w;

            for (int i = 0; i < 4; i++)
            {
                if (dims[i] == 0)
                    dims[i] = 1;
            }

            ret = set_ir_tensor_shape(output_tensor, dims, 4);
        }
    }
    else
    {
        if (output_tensor->dims[1] != out_h || output_tensor->dims[2] != out_w || output_tensor->dims[3] != out_c)
        {
            dims[1] = out_h;
            dims[2] = out_w;
            dims[3] = out_c;

            for (int i = 0; i < 4; i++)
            {
                if (dims[i] == 0)
                    dims[i] = 1;
            }

            ret = set_ir_tensor_shape(output_tensor, dims, 4);
        }
    }

    return ret;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
        .run = run,
        .reshape = reshape,
        .postrun = NULL,
        .init_node = init_node,
        .release_node = release_node,
        .score = score};

int register_conv_ref_op()
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

int unregister_conv_ref_op()
{
    unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
    return 0;
}
