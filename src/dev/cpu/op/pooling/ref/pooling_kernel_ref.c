#include <math.h>
#include "tengine_ir.h"
#include "pooling_kernel_ref.h"

#define HCL_POOL_MAX 0 /* Max pooling     */
#define HCL_POOL_AVG 1 /* Average pooling */

static inline float calc_sum_fp32(const float* input, int layout, int c, int h, int w, int cur_ch, int start_h,
                                  int start_w, int end_h, int end_w)
{
    float sum = 0.0f;
    for (int i = start_h; i < end_h; i++)
    {
        for (int j = start_w; j < end_w; j++)
        {
            if (layout == 0)
                sum += input[cur_ch * h * w + i * w + j];
            else
                sum += input[i * w * c + j * c + cur_ch];
        }
    }

    return sum;
}

static inline float calc_max_fp32(const float* input, int layout, int c, int h, int w, int cur_ch, int start_h,
                                  int start_w, int end_h, int end_w)
{
    float max = 0.0f;
    if (layout == 0)
    {
        max = input[cur_ch * h * w + start_h * w + start_w];
    }
    else
        max = input[start_h * w * c + start_w * c + cur_ch];

    float tmp = 0.0f;
    for (int i = start_h; i < end_h; i++)
    {
        for (int j = start_w; j < end_w; j++)
        {
            if (layout == 0)
            {
                tmp = input[cur_ch * h * w + i * w + j];
            }
            else
                tmp = input[i * w * c + j * c + cur_ch];
            max = max > tmp ? max : tmp;
        }
    }

    return max;
}

static inline int calc_sum_uint8(const uint8_t* input, int layout, int c, int h, int w, int cur_ch, int start_h,
                                 int start_w, int end_h, int end_w)
{
    int sum = 0;
    for (int i = start_h; i < end_h; i++)
        for (int j = start_w; j < end_w; j++)
        {
            if (layout == 0)
                sum += input[cur_ch * h * w + i * w + j];
            else
                sum += input[i * w * c + j * c + cur_ch];
        }

    return sum;
}

static inline uint8_t calc_max_uint8(const uint8_t* input, int layout, int c, int h, int w, int cur_ch, int start_h,
                                     int start_w, int end_h, int end_w)
{
    uint8_t max = 0;
    if (layout == 0)
        max = input[cur_ch * h * w + start_h * w + start_w];
    else
        max = input[start_h * w * c + start_w * c + cur_ch];

    uint8_t tmp = 0;
    for (int i = start_h; i < end_h; i++)
        for (int j = start_w; j < end_w; j++)
        {
            if (layout == 0)
                tmp = input[cur_ch * h * w + i * w + j];
            else
                tmp = input[i * w * c + j * c + cur_ch];

            max = max > tmp ? max : tmp;
        }

    return max;
}

int pooling_kernel_ref_run(struct ir_tensor* input_tensor, struct ir_tensor* output_tensor,
                           struct pool_param* pool_param, int num_thread)
{
    int layout = input_tensor->layout;
    int type = input_tensor->data_type;

    int batch = input_tensor->dims[0];
    int channel = 0;
    int in_h = 0;
    int in_w = 0;
    int out_h = 0;
    int out_w = 0;

    if (layout == TENGINE_LAYOUT_NCHW)
    {
        channel = input_tensor->dims[1];
        in_h = input_tensor->dims[2];
        in_w = input_tensor->dims[3];

        out_h = output_tensor->dims[2];
        out_w = output_tensor->dims[3];
    }
    else
    {
        channel = input_tensor->dims[3];
        in_h = input_tensor->dims[1];
        in_w = input_tensor->dims[2];

        out_h = output_tensor->dims[1];
        out_w = output_tensor->dims[2];
    }

    int input_chw = channel * in_h * in_w;
    int output_chw = channel * out_h * out_w;

    int stride_h = pool_param->stride_h;
    int stride_w = pool_param->stride_w;

    int pad_h = pool_param->pad_h0;
    int pad_w = pool_param->pad_w0;

    int kernel_h = pool_param->kernel_h;
    int kernel_w = pool_param->kernel_w;

    int caffe_flavor = pool_param->caffe_flavor;
    int method = pool_param->pool_method;

    if (type == TENGINE_DT_FP32)
    {
        float* input = input_tensor->data;
        float* output = output_tensor->data;

        for (int n = 0; n < batch; n++)
        {
            const float* input_cur = input + n * input_chw;
            for (int c = 0; c < channel; c++)
            {
                for (int ph = 0; ph < out_h; ph++)
                {
                    for (int pw = 0; pw < out_w; pw++)
                    {
                        int pool_size = 1;
                        int offset = 0;
                        int h_start = ph * stride_h - pad_h;
                        int h_end = h_start + kernel_h;

                        if (h_end > in_h + pad_h)
                            h_end = in_h + pad_h;
                        int w_start = pw * stride_w - pad_w;
                        int w_end = w_start + kernel_w;

                        if (w_end > in_w + pad_w)
                            w_end = in_w + pad_w;

                        if (caffe_flavor)
                            pool_size = (h_end - h_start) * (w_end - w_start);

                        h_start = h_start > 0 ? h_start : 0;
                        w_start = w_start > 0 ? w_start : 0;
                        h_end = h_end < in_h ? h_end : in_h;
                        w_end = w_end < in_w ? w_end : in_w;

                        if (!caffe_flavor)
                            pool_size = (h_end - h_start) * (w_end - w_start);
                        if (layout == TENGINE_LAYOUT_NCHW)    // nchw
                            offset = n * output_chw + c * out_h * out_w + ph * out_w + pw;
                        else
                            offset = n * output_chw + ph * out_w * channel + pw * channel + c;

                        if (method == HCL_POOL_MAX)
                        {
                            float max = calc_max_fp32(input_cur, layout, channel, in_h, in_w, c, h_start, w_start,
                                                      h_end, w_end);
                            output[offset] = max;
                        }
                        else if (method == HCL_POOL_AVG)
                        {
                            float sum = calc_sum_fp32(input_cur, layout, channel, in_h, in_w, c, h_start, w_start,
                                                      h_end, w_end);
                            output[offset] = sum / pool_size;
                        }
                        else
                            return -1;
                    }
                }
            }
        }
    }
    else
    {
        uint8_t* input_uint8 = ( uint8_t* )input_tensor->data;
        uint8_t* output_uint8 = ( uint8_t* )output_tensor->data;

        float input_scale = input_tensor->scale;
        float output_scale = output_tensor->scale;
        int input_zero = input_tensor->zero_point;
        int output_zero = output_tensor->zero_point;

        /* input dequant */
        float* input_fp32 = ( float* )sys_malloc(input_tensor->elem_num * sizeof(float));
        float* output_fp32 = ( float* )sys_malloc(output_tensor->elem_num * sizeof(float));

        for (int i = 0; i < input_tensor->elem_num; i++)
            input_fp32[i] = (input_uint8[i] - input_zero) * input_scale; 

        float* input = input_fp32;
        float* output = output_fp32;

        for (int n = 0; n < batch; n++)
        {
            const float* input_cur = input + n * input_chw;
            for (int c = 0; c < channel; c++)
            {
                for (int ph = 0; ph < out_h; ph++)
                {
                    for (int pw = 0; pw < out_w; pw++)
                    {
                        int pool_size = 1;
                        int offset = 0;
                        int h_start = ph * stride_h - pad_h;
                        int h_end = h_start + kernel_h;

                        if (h_end > in_h + pad_h)
                            h_end = in_h + pad_h;
                        int w_start = pw * stride_w - pad_w;
                        int w_end = w_start + kernel_w;

                        if (w_end > in_w + pad_w)
                            w_end = in_w + pad_w;

                        if (caffe_flavor)
                            pool_size = (h_end - h_start) * (w_end - w_start);

                        h_start = h_start > 0 ? h_start : 0;
                        w_start = w_start > 0 ? w_start : 0;
                        h_end = h_end < in_h ? h_end : in_h;
                        w_end = w_end < in_w ? w_end : in_w;

                        if (!caffe_flavor)
                            pool_size = (h_end - h_start) * (w_end - w_start);
                        if (layout == TENGINE_LAYOUT_NCHW)    // nchw
                            offset = n * output_chw + c * out_h * out_w + ph * out_w + pw;
                        else
                            offset = n * output_chw + ph * out_w * channel + pw * channel + c;

                        if (method == HCL_POOL_MAX)
                        {
                            float max = calc_max_fp32(input_cur, layout, channel, in_h, in_w, c, h_start, w_start,
                                                      h_end, w_end);
                            output[offset] = max;
                        }
                        else if (method == HCL_POOL_AVG)
                        {
                            float sum = calc_sum_fp32(input_cur, layout, channel, in_h, in_w, c, h_start, w_start,
                                                      h_end, w_end);
                            output[offset] = sum / pool_size;
                        }
                        else
                            return -1;
                    }
                }
            }
        }

        /* output quant */
        for (int i = 0; i < output_tensor->elem_num; i++)
        {
            int output_data = round(output_fp32[i] / output_scale) + output_zero;
            output_uint8[i] = output_data > 255 ? 255 : output_data;
        }

        sys_free(input_fp32);
        sys_free(output_fp32); 
    }

    return 0;
}