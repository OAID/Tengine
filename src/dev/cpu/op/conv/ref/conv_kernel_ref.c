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
    return filter->elem_num * filter->elem_size;    // caution
}

static void interleave(struct ir_tensor* filter, struct conv_priv_info* priv_info)
{
    /* simply copy the data */
    memcpy(priv_info->interleave_buffer, filter->data, filter->elem_num * filter->elem_size);
}

static inline void copy_one_element(void* src, void* dst, int src_off, int dst_off, int elem_type, int zero_point)
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
        case TENGINE_DT_UINT8:
        {
            int8_t* int_dst = dst;
            uint8_t* int_src = src;
            int_dst[dst_off] = (int8_t)((int)int_src[src_off] - (int)zero_point);
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
            int8_t* int_dst = dst;
            int_dst[dst_off] = 0x0;
        }
        break;
    }
}

static void im2col(struct ir_tensor* input, struct ir_tensor* output, struct conv_priv_info* priv_info, struct conv_param* param, int n, int group)
{
    int input_chan = param->input_channel / param->group;
    int image_size = input->dims[1] * input->dims[2] * input->dims[3];
    int group_size = input_chan * input->dims[2] * input->dims[3];

    void* input_base  = input->data + (n * image_size + group * group_size) * input->elem_size;
    void* im2col_buf = priv_info->im2col_buffer;

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
    int d_h  = param->dilation_h;
    int d_w  = param->dilation_w;
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
                    copy_one_element(input_base, im2col_buf, in_off, out_off, data_type, zero_point);
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
    for(int i = 0; i < outchan_g; i++)
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

static void sgemm_uint8(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* bias_tensor,
                        struct ir_tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                        int n, int group, int num_thread)
{
    int kernel_size = param->kernel_h * param->kernel_w * param->input_channel / param->group;
    int outchan_g = param->output_channel / param->group;

    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_image_size = output_tensor->dims[1] * output_tensor->dims[2] * output_tensor->dims[3];

    /* data point */
    unsigned char* interleave_uint8 = ( unsigned char* )priv_info->interleave_buffer + outchan_g * group * kernel_size;
    signed char* im2col_int8 = priv_info->im2col_buffer;
    unsigned char* output_uint8 = ( unsigned char* )output_tensor->data + n * out_image_size + outchan_g * group * out_h * out_w;
    int* bias_int32 = NULL;
    if (bias_tensor)
        bias_int32 = ( int* )bias_tensor->data + outchan_g * group;

    /* quantizaion scale and zero-point */
    float input_scale = input_tensor->scale;
    float weight_scale = filter_tensor->scale;
    float output_scale = output_tensor->scale;
//    float bias_scale = 0.f;
//    if (bias_tensor)
//        bias_scale = bias_tensor->scale;

    unsigned char input_zero = input_tensor->zero_point;
    unsigned char weight_zero = filter_tensor->zero_point;
    unsigned char output_zero = output_tensor->zero_point;

    /* int8 sgemm */
    #pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < outchan_g; i++)
    {
        unsigned char* kernel = interleave_uint8 + i * kernel_size;
        signed char* input = im2col_int8;
        unsigned char* output = output_uint8 + i * (out_h * out_w);

        for (int j = 0; j < out_h * out_w; j++)
        {
            int im2col_off = j * kernel_size;
            int sum_int32 = bias_tensor ? bias_int32[i] : 0;

            for (int k = 0; k < kernel_size; k++)
            {
                int input_data = input[im2col_off + k];
                int input_data_u32 = ( unsigned char )input[im2col_off + k];
                int kernel_data = kernel[k] - weight_zero;

                if (input_zero == 0)
                    sum_int32 += input_data_u32 * kernel_data;
                else
                    sum_int32 += input_data * kernel_data;
            }

            // dequant sum from int32 to fp32
            float sum_fp32 = (float)sum_int32 * input_scale * weight_scale;

            // relu
            if (param->activation > 0)
            {
                if (sum_fp32 < 0)
                    sum_fp32 = 0;
            }

            // relu6
            if (param->activation > 0)
            {
                if (sum_fp32 < 0)
                    sum_fp32 = 0;
                if (sum_fp32 > 6)
                    sum_fp32 = 6;
            }

            // quant output from fp32 to uint8
            sum_int32 = round(sum_fp32 / output_scale) + output_zero;
            if (sum_int32 > 255)
                sum_int32 = 255;
            if (sum_int32 < 0)
                sum_int32 = 0;
            output[0] = sum_int32;
            output++;
        }
    }
}

int conv_kernel_get_shared_mem_size(struct ir_tensor* input, struct ir_tensor* output, struct conv_param* param)
{
    int group = param->group;
    int input_chan = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int output_xy = output->dims[2] * output->dims[3];
    int elem_size = input->elem_size;

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
            im2col(input_tensor, output_tensor, priv_info, param, i, j);
            if (type == TENGINE_DT_FP32)
                sgemm_fp32(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, i, j, num_thread);
            else
                sgemm_uint8(input_tensor, filter_tensor, bias_tensor, output_tensor, priv_info, param, i, j, num_thread);
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
