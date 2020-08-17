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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: bhu@openailab.com
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "../conv_ref_kernel.h"

static int get_private_mem_size(struct ir_tensor* filter)
{
    if (filter->data_type == TENGINE_DT_UINT8)    // simulator uint8 inference with fp32
        return filter->elem_num * filter->elem_size * 4;
    else
        return filter->elem_num * filter->elem_size;    // caution
}

static void interleave(struct ir_tensor* filter, struct conv_priv_info* priv_info)
{
    /* simply copy the data */
    memcpy(priv_info->interleave_buffer, filter->data, filter->elem_num * filter->elem_size);
}

static void interleave_uint8(struct ir_tensor* filter, struct conv_priv_info* priv_info)
{
    /* dequant uint8 weight to fp32 for simulator */
    float* weight_fp32 = ( float* )priv_info->interleave_buffer;
    uint8_t* weight_uint8 = ( uint8_t* )filter->data;
    float scale = filter->scale;
    int zero_point = filter->zero_point;

    for (int i = 0; i < filter->elem_num; i++)
    {
        weight_fp32[i] = (( float )weight_uint8[i] - ( float )zero_point) * scale;
    }
}

static inline void copy_one_element(void* src, void* dst, int src_off, int dst_off, int elem_type, int zero_point,
                                    float scale)
{
    switch (elem_type)
    {
        case TENGINE_DT_FP32:
        case TENGINE_DT_INT32:
        {
            int32_t* int_dst = dst;
            int32_t* int_src = src;
            int_dst[dst_off] = int_src[src_off];
        }
        break;
        case TENGINE_DT_FP16:
        case TENGINE_DT_INT16:
        {
            int16_t* int_dst = dst;
            int16_t* int_src = src;
            int_dst[dst_off] = int_src[src_off];
        }
        break;
        case TENGINE_DT_INT8:
        {
            int8_t* int_dst = dst;
            int8_t* int_src = src;
            int_dst[dst_off] = int_src[src_off] - zero_point;
        }
        break;
        case TENGINE_DT_UINT8:    // simulator uint8 inference with fp32
        {
            float* int_dst = dst;
            uint8_t* int_src = src;
            int_dst[dst_off] = (( float )int_src[src_off] - ( float )zero_point) * scale;
        }
        break;
    }
}

static inline void zero_one_element(void* dst, int dst_off, int elem_type)
{
    switch (elem_type)
    {
        case TENGINE_DT_FP32:
        case TENGINE_DT_INT32:
        {
            int32_t* int_dst = dst;
            int_dst[dst_off] = 0x0;
        }
        break;
        case TENGINE_DT_FP16:
        case TENGINE_DT_INT16:
        {
            int16_t* int_dst = dst;
            int_dst[dst_off] = 0x0;
        }
        break;
        case TENGINE_DT_INT8:
        case TENGINE_DT_UINT8:
        {
            float* int_dst = dst;   // simulator uint8 inference with fp32
            int_dst[dst_off] = 0x0;
        }
        break;
    }
}

static void im2col_fp32(struct ir_tensor* input, struct ir_tensor* output, struct conv_priv_info* priv_info,
                        struct conv_param* param, int n, int group)
{
    int input_chan = param->input_channel / param->group;
    int image_size = input->dims[1] * input->dims[2] * input->dims[3];
    int group_size = input_chan * input->dims[2] * input->dims[3];

    void* input_base = input->data + (n * image_size + group * group_size) * input->elem_size;
    void* im2col_buf = priv_info->im2col_buffer;

    float scale = input->scale;
    int zero_point = input->zero_point;

    int k_h = param->kernel_h;
    int k_w = param->kernel_w;
    int in_c = input_chan;
    int in_h = input->dims[2];
    int in_w = input->dims[3];
    int out_h = output->dims[2];
    int out_w = output->dims[3];
    int s_h = param->stride_h;
    int s_w = param->stride_w;
    int p_h0 = param->pad_h0;
    int p_w0 = param->pad_w0;
    int d_h = param->dilation_h;
    int d_w = param->dilation_w;
    int data_type = input->data_type;
    int kernel_size = k_h * k_w * in_c;

    for (int i = 0; i < kernel_size; i++)
    {
        int c_off = i / (k_h * k_w);
        int c_left = i % (k_h * k_w);

        int kh_off = c_left / k_w;
        int kw_off = c_left % k_w;

        for (int l = 0; l < out_h; l++)
        {
            for (int m = 0; m < out_w; m++)
            {
                int out_off = (l * out_w + m) * kernel_size + i;
                int img_h = l * s_h - p_h0 + kh_off * d_h;
                int img_w = m * s_w - p_w0 + kw_off * d_w;

                if (img_h >= 0 && img_w >= 0 && img_h < in_h && img_w < in_w)
                {
                    int in_off = c_off * in_h * in_w + img_h * in_w + img_w;
                    copy_one_element(input_base, im2col_buf, in_off, out_off, data_type, zero_point, scale);
                }
                else
                    zero_one_element(im2col_buf, out_off, data_type);
            }
        }
    }
}

static void im2col_uint8(struct ir_tensor* input, struct ir_tensor* output, struct conv_priv_info* priv_info,
                         struct conv_param* param, int n, int group)
{
    int input_chan = param->input_channel / param->group;
    int image_size = input->dims[1] * input->dims[2] * input->dims[3];
    int group_size = input_chan * input->dims[2] * input->dims[3];

    void* input_base = input->data + (n * image_size + group * group_size) * input->elem_size;
    void* im2col_buf = priv_info->im2col_buffer;

    float scale = input->scale;
    int zero_point = input->zero_point;

    int k_h = param->kernel_h;
    int k_w = param->kernel_w;
    int in_c = input_chan;
    int in_h = input->dims[2];
    int in_w = input->dims[3];
    int out_h = output->dims[2];
    int out_w = output->dims[3];
    int s_h = param->stride_h;
    int s_w = param->stride_w;
    int p_h0 = param->pad_h0;
    int p_w0 = param->pad_w0;
    int d_h = param->dilation_h;
    int d_w = param->dilation_w;
    int data_type = input->data_type;
    int kernel_size = k_h * k_w * in_c;

    for (int i = 0; i < kernel_size; i++)
    {
        int c_off = i / (k_h * k_w);
        int c_left = i % (k_h * k_w);

        int kh_off = c_left / k_w;
        int kw_off = c_left % k_w;

        for (int l = 0; l < out_h; l++)
        {
            for (int m = 0; m < out_w; m++)
            {
                int out_off = (l * out_w + m) * kernel_size + i;
                int img_h = l * s_h - p_h0 + kh_off * d_h;
                int img_w = m * s_w - p_w0 + kw_off * d_w;

                if (img_h >= 0 && img_w >= 0 && img_h < in_h && img_w < in_w)
                {
                    int in_off = c_off * in_h * in_w + img_h * in_w + img_w;
                    copy_one_element(input_base, im2col_buf, in_off, out_off, data_type, zero_point, scale);
                }
                else
                    zero_one_element(im2col_buf, out_off, data_type);
            }
        }
    }
}

static void sgemm_fp32(struct ir_tensor* input, struct ir_tensor* filter, struct ir_tensor* bias,
                       struct ir_tensor* output, struct conv_priv_info* priv_info, struct conv_param* param, int n,
                       int group, int num_thread)
{
    int kernel_size = param->kernel_h * param->kernel_w * param->input_channel / param->group;
    int outchan_g = param->output_channel / param->group;

    int out_h = output->dims[2];
    int out_w = output->dims[3];
    int out_image_size = output->dims[1] * output->dims[2] * output->dims[3];

    float* interleave_fp32 = ( float* )priv_info->interleave_buffer + outchan_g * group * kernel_size;
    float* im2col_fp32 = priv_info->im2col_buffer;
    float* output_fp32 = ( float* )output->data + n * out_image_size + outchan_g * group * out_h * out_w;
    float* bias_fp32 = NULL;

    if (bias)
        bias_fp32 = ( float* )bias->data + outchan_g * group;

#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < outchan_g; i++)
    {
        float* kernel = interleave_fp32 + i * kernel_size;
        float* input = im2col_fp32;
        float* output = output_fp32 + i * (out_h * out_w);

        for (int j = 0; j < out_h * out_w; j++)
        {
            int im2col_off = j * kernel_size;

            float sum = 0.f;
            for (int k = 0; k < kernel_size; k++)
            {
                sum += kernel[k] * input[im2col_off + k];
            }
            output[0] = sum;
            output++;
        }
    }

    // process bias
    if (bias)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;
                output_fp32[output_off] += bias_fp32[i];
            }
        }
    }

    // process activation relu
    if (param->activation == 0)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
            }
        }
    }

    // process activation relu6
    if (param->activation > 0)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
                if (output_fp32[output_off] > 6)
                    output_fp32[output_off] = 6;
            }
        }
    }
}

static void sgemm_uint8(struct ir_tensor* input, struct ir_tensor* filter, struct ir_tensor* bias,
                        struct ir_tensor* output, struct conv_priv_info* priv_info, struct conv_param* param, int n,
                        int group, int num_thread)
{
    int kernel_size = param->kernel_h * param->kernel_w * param->input_channel / param->group;
    int outchan_g = param->output_channel / param->group;

    int out_c = output->dims[1];
    int out_h = output->dims[2];
    int out_w = output->dims[3];
    int out_image_size = out_c * out_h * out_w;
    int out_group_size = outchan_g * out_h * out_w;

    float* interleave_fp32 = ( float* )priv_info->interleave_buffer + outchan_g * group * kernel_size;
    float* im2col_fp32 = priv_info->im2col_buffer;
    float* output_fp32 = ( float* )sys_malloc(out_group_size * sizeof(float));
    uint8_t* output_uint8 = ( uint8_t* )output->data + n * out_image_size + outchan_g * group * out_h * out_w;
    int32_t* bias_int32 = NULL;
    float bias_scale = 0.f;

    if (bias)
    {
        bias_int32 = ( int32_t* )bias->data + outchan_g * group;
        bias_scale = input->scale * filter->scale;
    }

#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < outchan_g; i++)
    {
        float* kernel = interleave_fp32 + i * kernel_size;
        float* input = im2col_fp32;
        float* output = output_fp32 + i * (out_h * out_w);

        for (int j = 0; j < out_h * out_w; j++)
        {
            int im2col_off = j * kernel_size;

            float sum = 0.f;
            for (int k = 0; k < kernel_size; k++)
            {
                sum += kernel[k] * input[im2col_off + k];
            }
            output[0] = sum;
            output++;
        }
    }

    // process bias
    if (bias)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;
                output_fp32[output_off] += (float )bias_int32[i] * bias_scale;
            }
        }
    }

    // process activation relu
    if (param->activation == 0)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
            }
        }
    }

    // process activation relu6
    if (param->activation > 0)
    {
        for (int i = 0; i < outchan_g; i++)
        {
            for (int j = 0; j < out_h * out_w; j++)
            {
                int output_off = i * (out_h * out_w) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
                if (output_fp32[output_off] > 6)
                    output_fp32[output_off] = 6;
            }
        }
    }

    /* quant from fp32 to uint8 */
    for (int i = 0; i < outchan_g; i++)
    {
        for (int j = 0; j < out_h * out_w; j++)
        {
            int output_off = i * (out_h * out_w) + j;

            int udata = ( int )(round(output_fp32[output_off] / output->scale) + output->zero_point);
            if (udata > 255)
                udata = 255;
            else if (udata < 0)
                udata = 0;
            output_uint8[output_off] = udata;
        }
    }

    sys_free(output_fp32);
}

int conv_kernel_get_shared_mem_size(struct ir_tensor* input, struct ir_tensor* output, struct conv_param* param)
{
    int group = param->group;
    int input_chan = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int output_xy = output->dims[2] * output->dims[3];
    int elem_size = input->elem_size;

    // simulator uint8 inference with fp32
    if (input->data_type == TENGINE_DT_UINT8)
        elem_size = 4;

    return elem_size * output_xy * kernel_size;
}

int conv_kernel_prerun(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* output_tensor,
                       struct conv_priv_info* priv_info, struct conv_param* param)
{
    if (!priv_info->external_im2col_mem)
    {
        int mem_size = conv_kernel_get_shared_mem_size(input_tensor, output_tensor, param);
        void* mem = sys_malloc(mem_size);
        priv_info->im2col_buffer = mem;
        priv_info->im2col_buffer_size = mem_size;
    }

    if (!priv_info->external_interleave_mem)
    {
        int mem_size = get_private_mem_size(filter_tensor);
        void* mem = sys_malloc(mem_size);
        priv_info->interleave_buffer = mem;
        priv_info->interleave_buffer_size = mem_size;
    }

    if (input_tensor->data_type == TENGINE_DT_UINT8)
        interleave_uint8(filter_tensor, priv_info);
    else
        interleave(filter_tensor, priv_info);

    return 0;
}

int conv_kernel_postrun(struct conv_priv_info* priv_info)
{
    if (!priv_info->external_interleave_mem && priv_info->interleave_buffer != NULL)
    {
        sys_free(priv_info->interleave_buffer);
        priv_info->interleave_buffer = NULL;
    }

    if (!priv_info->external_im2col_mem && priv_info->im2col_buffer != NULL)
    {
        sys_free(priv_info->im2col_buffer);
        priv_info->im2col_buffer = NULL;
    }

    return 0;
}

int conv_kernel_run(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* bias_tensor,
                    struct ir_tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                    int num_thread, int cpu_affinity)
{
    int group = param->group;
    int type = input_tensor->data_type;

    for (int i = 0; i < input_tensor->dims[0]; i++)    // batch size
    {
        for (int j = 0; j < group; j++)
        {
            if (type == TENGINE_DT_FP32)
            {
                im2col_fp32(input_tensor, output_tensor, priv_info, param, i, j);
                sgemm_fp32(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, i, j, num_thread);
            }
            else if (type == TENGINE_DT_UINT8)
            {
                im2col_uint8(input_tensor, output_tensor, priv_info, param, i, j);
                sgemm_uint8(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, i, j, num_thread);
            }
        }
    }

    return 0;
}

int conv_kernel_set_shared_mem(struct conv_priv_info* priv_info, void* mem, int mem_size)
{
    priv_info->external_im2col_mem = 1;
    priv_info->im2col_buffer = mem;
    priv_info->im2col_buffer_size = mem_size;
    return 0;
}
