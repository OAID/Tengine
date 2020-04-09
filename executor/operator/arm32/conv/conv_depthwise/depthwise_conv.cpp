#include <stdio.h>
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

extern "C" void dw_k5s1(float*, float*, float*, float*, int, int, int);

static float elem_activation(float tmp, int type)
{
    if(type == 0)
    {
        if(tmp < 0.0f)
            tmp = 0;
        if(type > 0)
            tmp = tmp < type ? tmp : type;
    }

    return tmp;
}

static float32x4_t vector_activation(float32x4_t tmp, int type)
{
    if(type == 0)
    {
        float32x4_t zero = vdupq_n_f32(0.0);
        tmp = vmaxq_f32(tmp, zero);
        if(type > 0)
        {
            float32x4_t max = vdupq_n_f32((float)type);
            tmp = vminq_f32(tmp, max);
        }
    }

    return tmp;
}
void depthwise_conv_k5s1(float* input, float* weight, float* bias, float* output, int input_h,
                                     int input_w, int channel, int output_h, int output_w, int pad0, int pad1, int activation)
{
    int input_h_pad = input_h + pad0 + pad1;
    int input_w_pad = input_w + pad0 + pad1;
    float* input_buf = nullptr;
    int no_pad = pad0 == 0 && pad1 == 0;
    if(no_pad)
        input_buf = input;
    else
        input_buf = (float*)malloc(sizeof(float) * input_h_pad * input_w_pad + 128);

    for(int c = 0; c < channel; c++)
    {
        if(!no_pad)
        {
            float* input_tmp = input_buf;
            float* input_c = input + c * input_h * input_w;
            memset(input_tmp, 0, sizeof(float) * (input_w_pad * pad0 + pad0));
            input_tmp += input_w_pad * pad0 + pad0;
            for(int h = 0; h < input_h; h++)
            {
                memcpy(input_tmp, input_c + h * input_w, sizeof(float) * input_w);
                input_tmp += input_w;
                memset(input_tmp, 0, sizeof(float) * (pad0 + pad1));
                input_tmp += pad0 + pad1;
            }
            memset(input_tmp, 0, sizeof(float) * (input_w_pad * pad1 - pad0));
        }
        float* weight_cur = weight + c * 25;
        float* output_cur = output + c * output_h * output_w;
        dw_k5s1(input_buf, weight_cur, bias + c, output_cur, output_h, output_w, activation);
        if(no_pad)
           input_buf += input_h * input_w; 

    }
    if(!no_pad)
        free(input_buf);
}

void depthwise_conv_k5s2(float* input_buf, float* weight_buf, float* bias, float* output_buf, int input_h,
                                     int input_w, int channel, int output_h, int output_w, int activation)
{
    int input_hw = input_h * input_w;
    int output_hw = output_h * output_w;
    int h_remain = input_h & 0x1;
    int w_remain = input_w & 0x1;
    int mid_h = output_h - 2;
    int mid_w = output_w - 2;
    int mid_w_block = mid_w & -4;
    int w, h;
    for(int c = 0; c < channel; c++)
    {
        float* input_buf_c = input_buf + c * input_hw;
        float* output_buf_c = output_buf + c * output_hw;
        float* weight_buf_c = weight_buf + c * 25;
        float bias_c = bias ? bias[c] : 0;
        float tmp = bias_c;

        tmp += weight_buf_c[12] * input_buf_c[0];
        tmp += weight_buf_c[13] * input_buf_c[1];
        tmp += weight_buf_c[14] * input_buf_c[2];
        tmp += weight_buf_c[17] * input_buf_c[input_w];
        tmp += weight_buf_c[18] * input_buf_c[input_w + 1];
        tmp += weight_buf_c[19] * input_buf_c[input_w + 2];
        tmp += weight_buf_c[22] * input_buf_c[input_w * 2];
        tmp += weight_buf_c[23] * input_buf_c[input_w * 2 + 1];
        tmp += weight_buf_c[24] * input_buf_c[input_w * 2 + 2];
        output_buf_c[0] = elem_activation(tmp, activation);
        for(w = 0; w < mid_w_block; w += 4)
        {
            float32x4_t sum0 = vdupq_n_f32(bias_c);
            float32x4_t line2_0 = vld1q_f32(input_buf_c + 2 * w);
            float32x4_t line2_1 = vld1q_f32(input_buf_c + 2 * w + 4);
            float32x4_t line2_2 = vld1q_f32(input_buf_c + 2 * w + 8);
            float32x4x2_t line2_01 = vuzpq_f32(line2_0, line2_1);
            float32x4x2_t line2_12 = vuzpq_f32(line2_1, line2_2);
            float32x4_t input2_2 = vextq_f32(line2_01.val[0], line2_2, 1);
            float32x4_t input2_3 = vextq_f32(line2_0, line2_12.val[1], 3);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[10]), line2_01.val[0]);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[11]), line2_01.val[1]);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[12]), input2_2);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[13]), input2_3);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[14]), line2_12.val[0]);
            float32x4_t line3_0 = vld1q_f32(input_buf_c + input_w + 2 * w);
            float32x4_t line3_1 = vld1q_f32(input_buf_c + input_w + 2 * w + 4);
            float32x4_t line3_2 = vld1q_f32(input_buf_c + input_w + 2 * w + 8);
            float32x4x2_t line3_01 = vuzpq_f32(line3_0, line3_1);
            float32x4x2_t line3_12 = vuzpq_f32(line3_1, line3_2);
            float32x4_t input3_2 = vextq_f32(line3_01.val[0], line3_2, 1);
            float32x4_t input3_3 = vextq_f32(line3_0, line3_12.val[1], 3);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[15]), line3_01.val[0]);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[16]), line3_01.val[1]);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[17]), input3_2);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[18]), input3_3);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[19]), line3_12.val[0]);
            float32x4_t line4_0 = vld1q_f32(input_buf_c + input_w * 2 + 2 * w);
            float32x4_t line4_1 = vld1q_f32(input_buf_c + input_w * 2 + 2 * w + 4);
            float32x4_t line4_2 = vld1q_f32(input_buf_c + input_w * 2 + 2 * w + 8);
            float32x4x2_t line4_01 = vuzpq_f32(line4_0, line4_1);
            float32x4x2_t line4_12 = vuzpq_f32(line4_1, line4_2);
            float32x4_t input4_2 = vextq_f32(line4_01.val[0], line4_2, 1);
            float32x4_t input4_3 = vextq_f32(line4_0, line4_12.val[1], 3);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[20]), line4_01.val[0]);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[21]), line4_01.val[1]);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[22]), input4_2);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[23]), input4_3);
            sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[24]), line4_12.val[0]);
            sum0 = vector_activation(sum0, activation);
            vst1q_f32(output_buf_c + w + 1, sum0);
            
        }
        for(w = mid_w_block; w < mid_w; w ++)
        {
            tmp = bias_c;
            tmp += weight_buf_c[10] * input_buf_c[2 * w];
            tmp += weight_buf_c[11] * input_buf_c[2 * w + 1];
            tmp += weight_buf_c[12] * input_buf_c[2 * w + 2];
            tmp += weight_buf_c[13] * input_buf_c[2 * w + 3];
            tmp += weight_buf_c[14] * input_buf_c[2 * w + 4];
            tmp += weight_buf_c[15] * input_buf_c[input_w + 2 * w];
            tmp += weight_buf_c[16] * input_buf_c[input_w + 2 * w + 1];
            tmp += weight_buf_c[17] * input_buf_c[input_w + 2 * w + 2];
            tmp += weight_buf_c[18] * input_buf_c[input_w + 2 * w + 3];
            tmp += weight_buf_c[19] * input_buf_c[input_w + 2 * w + 4];
            tmp += weight_buf_c[20] * input_buf_c[input_w * 2 + 2 * w];
            tmp += weight_buf_c[21] * input_buf_c[input_w * 2 + 2 * w + 1];
            tmp += weight_buf_c[22] * input_buf_c[input_w * 2 + 2 * w + 2];
            tmp += weight_buf_c[23] * input_buf_c[input_w * 2 + 2 * w + 3];
            tmp += weight_buf_c[24] * input_buf_c[input_w * 2 + 2 * w + 4];
            output_buf_c[w + 1] = elem_activation(tmp, activation);
        }
        if(w_remain)
        {
            tmp = bias_c;
            tmp += weight_buf_c[10] * input_buf_c[2 * w];
            tmp += weight_buf_c[11] * input_buf_c[2 * w + 1];
            tmp += weight_buf_c[12] * input_buf_c[2 * w + 2];
            tmp += weight_buf_c[15] * input_buf_c[input_w + 2 * w];
            tmp += weight_buf_c[16] * input_buf_c[input_w + 2 * w + 1];
            tmp += weight_buf_c[17] * input_buf_c[input_w + 2 * w + 2];
            tmp += weight_buf_c[20] * input_buf_c[input_w * 2 + 2 * w];
            tmp += weight_buf_c[21] * input_buf_c[input_w * 2 + 2 * w + 1];
            tmp += weight_buf_c[22] * input_buf_c[input_w * 2 + 2 * w + 2];
            output_buf_c[w + 1] = elem_activation(tmp, activation);
        }
        else
        {
            tmp = bias_c;
            tmp += weight_buf_c[10] * input_buf_c[2 * w];
            tmp += weight_buf_c[11] * input_buf_c[2 * w + 1];
            tmp += weight_buf_c[12] * input_buf_c[2 * w + 2];
            tmp += weight_buf_c[13] * input_buf_c[2 * w + 3];
            tmp += weight_buf_c[15] * input_buf_c[input_w + 2 * w];
            tmp += weight_buf_c[16] * input_buf_c[input_w + 2 * w + 1];
            tmp += weight_buf_c[17] * input_buf_c[input_w + 2 * w + 2];
            tmp += weight_buf_c[18] * input_buf_c[input_w + 2 * w + 3];
            tmp += weight_buf_c[20] * input_buf_c[input_w * 2 + 2 * w];
            tmp += weight_buf_c[21] * input_buf_c[input_w * 2 + 2 * w + 1];
            tmp += weight_buf_c[22] * input_buf_c[input_w * 2 + 2 * w + 2];
            tmp += weight_buf_c[23] * input_buf_c[input_w * 2 + 2 * w + 3];
            output_buf_c[w + 1] = elem_activation(tmp, activation);
        }

        // mid  height
        for(h = 0; h < mid_h; h++)
        {
            tmp = bias_c;
            tmp += weight_buf_c[2] * input_buf_c[input_w * 2 * h];
            tmp += weight_buf_c[3] * input_buf_c[input_w * 2 * h + 1];
            tmp += weight_buf_c[4] * input_buf_c[input_w * 2 * h + 2];
            tmp += weight_buf_c[7] * input_buf_c[input_w * (2 * h + 1)];
            tmp += weight_buf_c[8] * input_buf_c[input_w * (2 * h + 1) + 1];
            tmp += weight_buf_c[9] * input_buf_c[input_w * (2 * h + 1) + 2];
            tmp += weight_buf_c[12] * input_buf_c[input_w * (2 * h + 2)];
            tmp += weight_buf_c[13] * input_buf_c[input_w * (2 * h + 2) + 1];
            tmp += weight_buf_c[14] * input_buf_c[input_w * (2 * h + 2) + 2];
            tmp += weight_buf_c[17] * input_buf_c[input_w * (2 * h + 3)];
            tmp += weight_buf_c[18] * input_buf_c[input_w * (2 * h + 3) + 1];
            tmp += weight_buf_c[19] * input_buf_c[input_w * (2 * h + 3) + 2];
            tmp += weight_buf_c[22] * input_buf_c[input_w * (2 * h + 4)];
            tmp += weight_buf_c[23] * input_buf_c[input_w * (2 * h + 4) + 1];
            tmp += weight_buf_c[24] * input_buf_c[input_w * (2 * h + 4) + 2];
            output_buf_c[output_w * (h + 1)] = elem_activation(tmp, activation);
            for(w = 0; w < mid_w_block; w += 4)
            {
                float32x4_t sum0 = vdupq_n_f32(bias_c);
                float32x4_t line0_0 = vld1q_f32(input_buf_c + input_w * 2 * h + 2 * w);
                float32x4_t line0_1 = vld1q_f32(input_buf_c + input_w * 2 * h + 2 * w + 4);
                float32x4_t line0_2 = vld1q_f32(input_buf_c + input_w * 2 * h + 2 * w + 8);
                float32x4x2_t line0_01 = vuzpq_f32(line0_0, line0_1);
                float32x4x2_t line0_12 = vuzpq_f32(line0_1, line0_2);
                float32x4_t input0_2 = vextq_f32(line0_01.val[0], line0_2, 1);
                float32x4_t input0_3 = vextq_f32(line0_0, line0_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[0]), line0_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[1]), line0_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[2]), input0_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[3]), input0_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[4]), line0_12.val[0]);

                float32x4_t line1_0 = vld1q_f32(input_buf_c + input_w * (2 * h + 1) + 2 * w);
                float32x4_t line1_1 = vld1q_f32(input_buf_c + input_w * (2 * h + 1) + 2 * w + 4);
                float32x4_t line1_2 = vld1q_f32(input_buf_c + input_w * (2 * h + 1) + 2 * w + 8);
                float32x4x2_t line1_01 = vuzpq_f32(line1_0, line1_1);
                float32x4x2_t line1_12 = vuzpq_f32(line1_1, line1_2);
                float32x4_t input1_2 = vextq_f32(line1_01.val[0], line1_2, 1);
                float32x4_t input1_3 = vextq_f32(line1_0, line1_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[5]), line1_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[6]), line1_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[7]), input1_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[8]), input1_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[9]), line1_12.val[0]);

                float32x4_t line2_0 = vld1q_f32(input_buf_c + input_w * (2 * h + 2) + 2 * w);
                float32x4_t line2_1 = vld1q_f32(input_buf_c + input_w * (2 * h + 2) + 2 * w + 4);
                float32x4_t line2_2 = vld1q_f32(input_buf_c + input_w * (2 * h + 2) + 2 * w + 8);
                float32x4x2_t line2_01 = vuzpq_f32(line2_0, line2_1);
                float32x4x2_t line2_12 = vuzpq_f32(line2_1, line2_2);
                float32x4_t input2_2 = vextq_f32(line2_01.val[0], line2_2, 1);
                float32x4_t input2_3 = vextq_f32(line2_0, line2_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[10]), line2_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[11]), line2_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[12]), input2_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[13]), input2_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[14]), line2_12.val[0]);

                float32x4_t line3_0 = vld1q_f32(input_buf_c + input_w * (2 * h + 3) + 2 * w);
                float32x4_t line3_1 = vld1q_f32(input_buf_c + input_w * (2 * h + 3) + 2 * w + 4);
                float32x4_t line3_2 = vld1q_f32(input_buf_c + input_w * (2 * h + 3) + 2 * w + 8);
                float32x4x2_t line3_01 = vuzpq_f32(line3_0, line3_1);
                float32x4x2_t line3_12 = vuzpq_f32(line3_1, line3_2);
                float32x4_t input3_2 = vextq_f32(line3_01.val[0], line3_2, 1);
                float32x4_t input3_3 = vextq_f32(line3_0, line3_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[15]), line3_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[16]), line3_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[17]), input3_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[18]), input3_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[19]), line3_12.val[0]);
                
                float32x4_t line4_0 = vld1q_f32(input_buf_c + input_w * (2 * h + 4) + 2 * w);
                float32x4_t line4_1 = vld1q_f32(input_buf_c + input_w * (2 * h + 4) + 2 * w + 4);
                float32x4_t line4_2 = vld1q_f32(input_buf_c + input_w * (2 * h + 4) + 2 * w + 8);
                float32x4x2_t line4_01 = vuzpq_f32(line4_0, line4_1);
                float32x4x2_t line4_12 = vuzpq_f32(line4_1, line4_2);
                float32x4_t input4_2 = vextq_f32(line4_01.val[0], line4_2, 1);
                float32x4_t input4_3 = vextq_f32(line4_0, line4_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[20]), line4_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[21]), line4_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[22]), input4_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[23]), input4_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[24]), line4_12.val[0]);
                sum0 = vector_activation(sum0, activation);
                vst1q_f32(output_buf_c + output_w * (h + 1) + w + 1, sum0);
                
            }
            for(w = mid_w_block; w < mid_w; w ++)
            {
                tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[input_w * 2 * h + 2 * w];
                tmp += weight_buf_c[1] * input_buf_c[input_w * 2 * h + 2 * w + 1];
                tmp += weight_buf_c[2] * input_buf_c[input_w * 2 * h + 2 * w + 2];
                tmp += weight_buf_c[3] * input_buf_c[input_w * 2 * h + 2 * w + 3];
                tmp += weight_buf_c[4] * input_buf_c[input_w * 2 * h + 2 * w + 4];
                tmp += weight_buf_c[5] * input_buf_c[input_w * (2 * h + 1) + 2 * w];
                tmp += weight_buf_c[6] * input_buf_c[input_w * (2 * h + 1) + 2 * w + 1];
                tmp += weight_buf_c[7] * input_buf_c[input_w * (2 * h + 1) + 2 * w + 2];
                tmp += weight_buf_c[8] * input_buf_c[input_w * (2 * h + 1) + 2 * w + 3];
                tmp += weight_buf_c[9] * input_buf_c[input_w * (2 * h + 1) + 2 * w + 4];
                tmp += weight_buf_c[10] * input_buf_c[input_w * (2 * h + 2) + 2 * w];
                tmp += weight_buf_c[11] * input_buf_c[input_w * (2 * h + 2) + 2 * w + 1];
                tmp += weight_buf_c[12] * input_buf_c[input_w * (2 * h + 2) + 2 * w + 2];
                tmp += weight_buf_c[13] * input_buf_c[input_w * (2 * h + 2) + 2 * w + 3];
                tmp += weight_buf_c[14] * input_buf_c[input_w * (2 * h + 2) + 2 * w + 4];
                tmp += weight_buf_c[15] * input_buf_c[input_w * (2 * h + 3) + 2 * w];
                tmp += weight_buf_c[16] * input_buf_c[input_w * (2 * h + 3) + 2 * w + 1];
                tmp += weight_buf_c[17] * input_buf_c[input_w * (2 * h + 3) + 2 * w + 2];
                tmp += weight_buf_c[18] * input_buf_c[input_w * (2 * h + 3) + 2 * w + 3];
                tmp += weight_buf_c[19] * input_buf_c[input_w * (2 * h + 3) + 2 * w + 4];
                tmp += weight_buf_c[20] * input_buf_c[input_w * (2 * h + 4) + 2 * w];
                tmp += weight_buf_c[21] * input_buf_c[input_w * (2 * h + 4) + 2 * w + 1];
                tmp += weight_buf_c[22] * input_buf_c[input_w * (2 * h + 4) + 2 * w + 2];
                tmp += weight_buf_c[23] * input_buf_c[input_w * (2 * h + 4) + 2 * w + 3];
                tmp += weight_buf_c[24] * input_buf_c[input_w * (2 * h + 4) + 2 * w + 4];
                output_buf_c[output_w * (h + 1) + w + 1] = elem_activation(tmp, activation);
            }
            if(w_remain)
            {
                tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[input_w * 2 * h + 2 * w];
                tmp += weight_buf_c[1] * input_buf_c[input_w * 2 * h + 2 * w + 1];
                tmp += weight_buf_c[2] * input_buf_c[input_w * 2 * h + 2 * w + 2];
                tmp += weight_buf_c[5] * input_buf_c[input_w * (2 * h + 1) + 2 * w];
                tmp += weight_buf_c[6] * input_buf_c[input_w * (2 * h + 1) + 2 * w + 1];
                tmp += weight_buf_c[7] * input_buf_c[input_w * (2 * h + 1) + 2 * w + 2];
                tmp += weight_buf_c[10] * input_buf_c[input_w * (2 * h + 2) + 2 * w];
                tmp += weight_buf_c[11] * input_buf_c[input_w * (2 * h + 2) + 2 * w + 1];
                tmp += weight_buf_c[12] * input_buf_c[input_w * (2 * h + 2) + 2 * w + 2];
                tmp += weight_buf_c[15] * input_buf_c[input_w * (2 * h + 3) + 2 * w];
                tmp += weight_buf_c[16] * input_buf_c[input_w * (2 * h + 3) + 2 * w + 1];
                tmp += weight_buf_c[17] * input_buf_c[input_w * (2 * h + 3) + 2 * w + 2];
                tmp += weight_buf_c[20] * input_buf_c[input_w * (2 * h + 4) + 2 * w];
                tmp += weight_buf_c[21] * input_buf_c[input_w * (2 * h + 4) + 2 * w + 1];
                tmp += weight_buf_c[22] * input_buf_c[input_w * (2 * h + 4) + 2 * w + 2];
                output_buf_c[output_w * (h + 2) - 1] = elem_activation(tmp, activation);
            }
            else
            {
                tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[input_w * 2 * h + 2 * w];
                tmp += weight_buf_c[1] * input_buf_c[input_w * 2 * h + 2 * w + 1];
                tmp += weight_buf_c[2] * input_buf_c[input_w * 2 * h + 2 * w + 2];
                tmp += weight_buf_c[3] * input_buf_c[input_w * 2 * h + 2 * w + 3];
                tmp += weight_buf_c[5] * input_buf_c[input_w * (2 * h + 1) + 2 * w];
                tmp += weight_buf_c[6] * input_buf_c[input_w * (2 * h + 1) + 2 * w + 1];
                tmp += weight_buf_c[7] * input_buf_c[input_w * (2 * h + 1) + 2 * w + 2];
                tmp += weight_buf_c[8] * input_buf_c[input_w * (2 * h + 1) + 2 * w + 3];
                tmp += weight_buf_c[10] * input_buf_c[input_w * (2 * h + 2) + 2 * w];
                tmp += weight_buf_c[11] * input_buf_c[input_w * (2 * h + 2) + 2 * w + 1];
                tmp += weight_buf_c[12] * input_buf_c[input_w * (2 * h + 2) + 2 * w + 2];
                tmp += weight_buf_c[13] * input_buf_c[input_w * (2 * h + 2) + 2 * w + 3];
                tmp += weight_buf_c[15] * input_buf_c[input_w * (2 * h + 3) + 2 * w];
                tmp += weight_buf_c[16] * input_buf_c[input_w * (2 * h + 3) + 2 * w + 1];
                tmp += weight_buf_c[17] * input_buf_c[input_w * (2 * h + 3) + 2 * w + 2];
                tmp += weight_buf_c[18] * input_buf_c[input_w * (2 * h + 3) + 2 * w + 3];
                tmp += weight_buf_c[20] * input_buf_c[input_w * (2 * h + 4) + 2 * w];
                tmp += weight_buf_c[21] * input_buf_c[input_w * (2 * h + 4) + 2 * w + 1];
                tmp += weight_buf_c[22] * input_buf_c[input_w * (2 * h + 4) + 2 * w + 2];
                tmp += weight_buf_c[23] * input_buf_c[input_w * (2 * h + 4) + 2 * w + 3];
                output_buf_c[output_w * (h + 2) - 1] = elem_activation(tmp, activation);
            }
        }
        if(h_remain)
        {
            tmp = bias_c;
            tmp += weight_buf_c[2] * input_buf_c[input_w * (input_h - 3)];
            tmp += weight_buf_c[3] * input_buf_c[input_w * (input_h - 3) + 1];
            tmp += weight_buf_c[4] * input_buf_c[input_w * (input_h - 3) + 2];
            tmp += weight_buf_c[7] * input_buf_c[input_w * (input_h - 2)];
            tmp += weight_buf_c[8] * input_buf_c[input_w * (input_h - 2) + 1];
            tmp += weight_buf_c[9] * input_buf_c[input_w * (input_h - 2) + 2];
            tmp += weight_buf_c[12] * input_buf_c[input_w * (input_h - 1)];
            tmp += weight_buf_c[13] * input_buf_c[input_w * (input_h - 1) + 1];
            tmp += weight_buf_c[14] * input_buf_c[input_w * (input_h - 1) + 2];
            output_buf_c[output_w * (output_h - 1)] = elem_activation(tmp, activation);
            for(w = 0; w < mid_w_block; w += 4)
            {
                float32x4_t sum0 = vdupq_n_f32(bias_c);
                float32x4_t line0_0 = vld1q_f32(input_buf_c + input_w * (input_h - 3) + 2* w);
                float32x4_t line0_1 = vld1q_f32(input_buf_c + input_w * (input_h - 3) + 2* w + 4);
                float32x4_t line0_2 = vld1q_f32(input_buf_c + input_w * (input_h - 3) + 2* w + 8);
                float32x4x2_t line0_01 = vuzpq_f32(line0_0, line0_1);
                float32x4x2_t line0_12 = vuzpq_f32(line0_1, line0_2);
                float32x4_t input0_2 = vextq_f32(line0_01.val[0], line0_2, 1);
                float32x4_t input0_3 = vextq_f32(line0_0, line0_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[0]), line0_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[1]), line0_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[2]), input0_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[3]), input0_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[4]), line0_12.val[0]);

                float32x4_t line1_0 = vld1q_f32(input_buf_c + input_w * (input_h - 2) + 2* w);
                float32x4_t line1_1 = vld1q_f32(input_buf_c + input_w * (input_h - 2) + 2* w + 4);
                float32x4_t line1_2 = vld1q_f32(input_buf_c + input_w * (input_h - 2) + 2* w + 8);
                float32x4x2_t line1_01 = vuzpq_f32(line1_0, line1_1);
                float32x4x2_t line1_12 = vuzpq_f32(line1_1, line1_2);
                float32x4_t input1_2 = vextq_f32(line1_01.val[0], line1_2, 1);
                float32x4_t input1_3 = vextq_f32(line1_0, line1_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[5]), line1_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[6]), line1_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[7]), input1_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[8]), input1_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[9]), line1_12.val[0]);

                float32x4_t line2_0 = vld1q_f32(input_buf_c + input_w * (input_h - 1) + 2* w);
                float32x4_t line2_1 = vld1q_f32(input_buf_c + input_w * (input_h - 1) + 2* w + 4);
                float32x4_t line2_2 = vld1q_f32(input_buf_c + input_w * (input_h - 1) + 2* w + 8);
                float32x4x2_t line2_01 = vuzpq_f32(line2_0, line2_1);
                float32x4x2_t line2_12 = vuzpq_f32(line2_1, line2_2);
                float32x4_t input2_2 = vextq_f32(line2_01.val[0], line2_2, 1);
                float32x4_t input2_3 = vextq_f32(line2_0, line2_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[10]), line2_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[11]), line2_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[12]), input2_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[13]), input2_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[14]), line2_12.val[0]);

                sum0 = vector_activation(sum0, activation);
                vst1q_f32(output_buf_c + output_w * (output_h - 1) + w + 1, sum0);

            }
            for(w = mid_w_block; w < mid_w; w ++)
            {
                tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[input_w * (input_h - 3) + 2 * w];
                tmp += weight_buf_c[1] * input_buf_c[input_w * (input_h - 3) + 2 * w + 1];
                tmp += weight_buf_c[2] * input_buf_c[input_w * (input_h - 3) + 2 * w + 2];
                tmp += weight_buf_c[3] * input_buf_c[input_w * (input_h - 3) + 2 * w + 3];
                tmp += weight_buf_c[4] * input_buf_c[input_w * (input_h - 3) + 2 * w + 4];
                tmp += weight_buf_c[5] * input_buf_c[input_w * (input_h - 2) + 2 * w];
                tmp += weight_buf_c[6] * input_buf_c[input_w * (input_h - 2) + 2 * w + 1];
                tmp += weight_buf_c[7] * input_buf_c[input_w * (input_h - 2) + 2 * w + 2];
                tmp += weight_buf_c[8] * input_buf_c[input_w * (input_h - 2) + 2 * w + 3];
                tmp += weight_buf_c[9] * input_buf_c[input_w * (input_h - 2) + 2 * w + 4];
                tmp += weight_buf_c[10] * input_buf_c[input_w * (input_h - 1) + 2 * w];
                tmp += weight_buf_c[11] * input_buf_c[input_w * (input_h - 1) + 2 * w + 1];
                tmp += weight_buf_c[12] * input_buf_c[input_w * (input_h - 1) + 2 * w + 2];
                tmp += weight_buf_c[13] * input_buf_c[input_w * (input_h - 1) + 2 * w + 3];
                tmp += weight_buf_c[14] * input_buf_c[input_w * (input_h - 1) + 2 * w + 4];
                output_buf_c[output_w * (output_h - 1) + w + 1] = elem_activation(tmp, activation);
            }
            if(w_remain)
            {
                tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[input_w * (input_h - 3) + 2 * w];
                tmp += weight_buf_c[1] * input_buf_c[input_w * (input_h - 3) + 2 * w + 1];
                tmp += weight_buf_c[2] * input_buf_c[input_w * (input_h - 3) + 2 * w + 2];
                tmp += weight_buf_c[5] * input_buf_c[input_w * (input_h - 2) + 2 * w];
                tmp += weight_buf_c[6] * input_buf_c[input_w * (input_h - 2) + 2 * w + 1];
                tmp += weight_buf_c[7] * input_buf_c[input_w * (input_h - 2) + 2 * w + 2];
                tmp += weight_buf_c[10] * input_buf_c[input_w * (input_h - 1) + 2 * w];
                tmp += weight_buf_c[11] * input_buf_c[input_w * (input_h - 1) + 2 * w + 1];
                tmp += weight_buf_c[12] * input_buf_c[input_w * (input_h - 1) + 2 * w + 2];
                output_buf_c[output_hw - 1] = elem_activation(tmp, activation);
            }
            else
            {
                tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[input_w * (input_h - 3) + 2 * w];
                tmp += weight_buf_c[1] * input_buf_c[input_w * (input_h - 3) + 2 * w + 1];
                tmp += weight_buf_c[2] * input_buf_c[input_w * (input_h - 3) + 2 * w + 2];
                tmp += weight_buf_c[3] * input_buf_c[input_w * (input_h - 3) + 2 * w + 3];
                tmp += weight_buf_c[5] * input_buf_c[input_w * (input_h - 2) + 2 * w];
                tmp += weight_buf_c[6] * input_buf_c[input_w * (input_h - 2) + 2 * w + 1];
                tmp += weight_buf_c[7] * input_buf_c[input_w * (input_h - 2) + 2 * w + 2];
                tmp += weight_buf_c[8] * input_buf_c[input_w * (input_h - 2) + 2 * w + 3];
                tmp += weight_buf_c[10] * input_buf_c[input_w * (input_h - 1) + 2 * w];
                tmp += weight_buf_c[11] * input_buf_c[input_w * (input_h - 1) + 2 * w + 1];
                tmp += weight_buf_c[12] * input_buf_c[input_w * (input_h - 1) + 2 * w + 2];
                tmp += weight_buf_c[13] * input_buf_c[input_w * (input_h - 1) + 2 * w + 3];
                output_buf_c[output_hw - 1] = elem_activation(tmp, activation);
            }
        }
        else
        {
            tmp = bias_c;
            tmp += weight_buf_c[2] * input_buf_c[input_w * (input_h - 4)];
            tmp += weight_buf_c[3] * input_buf_c[input_w * (input_h - 4) + 1];
            tmp += weight_buf_c[4] * input_buf_c[input_w * (input_h - 4) + 2];
            tmp += weight_buf_c[7] * input_buf_c[input_w * (input_h - 3)];
            tmp += weight_buf_c[8] * input_buf_c[input_w * (input_h - 3) + 1];
            tmp += weight_buf_c[9] * input_buf_c[input_w * (input_h - 3) + 2];
            tmp += weight_buf_c[12] * input_buf_c[input_w * (input_h - 2)];
            tmp += weight_buf_c[13] * input_buf_c[input_w * (input_h - 2) + 1];
            tmp += weight_buf_c[14] * input_buf_c[input_w * (input_h - 2) + 2];
            tmp += weight_buf_c[17] * input_buf_c[input_w * (input_h - 1)];
            tmp += weight_buf_c[18] * input_buf_c[input_w * (input_h - 1) + 1];
            tmp += weight_buf_c[19] * input_buf_c[input_w * (input_h - 1) + 2];
            output_buf_c[output_w * (output_h - 1)] = elem_activation(tmp, activation);
            for(w = 0; w < mid_w_block; w += 4)
            {
                float32x4_t sum0 = vdupq_n_f32(bias_c);
                float32x4_t line0_0 = vld1q_f32(input_buf_c + input_w * (input_h - 4) + 2* w);
                float32x4_t line0_1 = vld1q_f32(input_buf_c + input_w * (input_h - 4) + 2* w + 4);
                float32x4_t line0_2 = vld1q_f32(input_buf_c + input_w * (input_h - 4) + 2* w + 8);
                float32x4x2_t line0_01 = vuzpq_f32(line0_0, line0_1);
                float32x4x2_t line0_12 = vuzpq_f32(line0_1, line0_2);
                float32x4_t input0_2 = vextq_f32(line0_01.val[0], line0_2, 1);
                float32x4_t input0_3 = vextq_f32(line0_0, line0_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[0]), line0_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[1]), line0_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[2]), input0_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[3]), input0_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[4]), line0_12.val[0]);

                float32x4_t line1_0 = vld1q_f32(input_buf_c + input_w * (input_h - 3) + 2* w);
                float32x4_t line1_1 = vld1q_f32(input_buf_c + input_w * (input_h - 3) + 2* w + 4);
                float32x4_t line1_2 = vld1q_f32(input_buf_c + input_w * (input_h - 3) + 2* w + 8);
                float32x4x2_t line1_01 = vuzpq_f32(line1_0, line1_1);
                float32x4x2_t line1_12 = vuzpq_f32(line1_1, line1_2);
                float32x4_t input1_2 = vextq_f32(line1_01.val[0], line1_2, 1);
                float32x4_t input1_3 = vextq_f32(line1_0, line1_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[5]), line1_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[6]), line1_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[7]), input1_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[8]), input1_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[9]), line1_12.val[0]);

                float32x4_t line2_0 = vld1q_f32(input_buf_c + input_w * (input_h - 2) + 2* w);
                float32x4_t line2_1 = vld1q_f32(input_buf_c + input_w * (input_h - 2) + 2* w + 4);
                float32x4_t line2_2 = vld1q_f32(input_buf_c + input_w * (input_h - 2) + 2* w + 8);
                float32x4x2_t line2_01 = vuzpq_f32(line2_0, line2_1);
                float32x4x2_t line2_12 = vuzpq_f32(line2_1, line2_2);
                float32x4_t input2_2 = vextq_f32(line2_01.val[0], line2_2, 1);
                float32x4_t input2_3 = vextq_f32(line2_0, line2_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[10]), line2_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[11]), line2_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[12]), input2_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[13]), input2_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[14]), line2_12.val[0]);

                float32x4_t line3_0 = vld1q_f32(input_buf_c + input_w * (input_h - 1) + 2* w);
                float32x4_t line3_1 = vld1q_f32(input_buf_c + input_w * (input_h - 1) + 2* w + 4);
                float32x4_t line3_2 = vld1q_f32(input_buf_c + input_w * (input_h - 1) + 2* w + 8);
                float32x4x2_t line3_01 = vuzpq_f32(line3_0, line3_1);
                float32x4x2_t line3_12 = vuzpq_f32(line3_1, line3_2);
                float32x4_t input3_2 = vextq_f32(line3_01.val[0], line3_2, 1);
                float32x4_t input3_3 = vextq_f32(line3_0, line3_12.val[1], 3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[15]), line3_01.val[0]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[16]), line3_01.val[1]);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[17]), input3_2);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[18]), input3_3);
                sum0 = vmlaq_f32(sum0, vdupq_n_f32(weight_buf_c[19]), line3_12.val[0]);
                sum0 = vector_activation(sum0, activation);
                vst1q_f32(output_buf_c + output_w * (output_h - 1) + w + 1, sum0);
                
            }
            for(w = mid_w_block; w < mid_w; w ++)
            {
                tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[input_w * (input_h - 4) + 2 * w];
                tmp += weight_buf_c[1] * input_buf_c[input_w * (input_h - 4) + 2 * w + 1];
                tmp += weight_buf_c[2] * input_buf_c[input_w * (input_h - 4) + 2 * w + 2];
                tmp += weight_buf_c[3] * input_buf_c[input_w * (input_h - 4) + 2 * w + 3];
                tmp += weight_buf_c[4] * input_buf_c[input_w * (input_h - 4) + 2 * w + 4];
                tmp += weight_buf_c[5] * input_buf_c[input_w * (input_h - 3) + 2 * w];
                tmp += weight_buf_c[6] * input_buf_c[input_w * (input_h - 3) + 2 * w + 1];
                tmp += weight_buf_c[7] * input_buf_c[input_w * (input_h - 3) + 2 * w + 2];
                tmp += weight_buf_c[8] * input_buf_c[input_w * (input_h - 3) + 2 * w + 3];
                tmp += weight_buf_c[9] * input_buf_c[input_w * (input_h - 3) + 2 * w + 4];
                tmp += weight_buf_c[10] * input_buf_c[input_w * (input_h - 2) + 2 * w];
                tmp += weight_buf_c[11] * input_buf_c[input_w * (input_h - 2) + 2 * w + 1];
                tmp += weight_buf_c[12] * input_buf_c[input_w * (input_h - 2) + 2 * w + 2];
                tmp += weight_buf_c[13] * input_buf_c[input_w * (input_h - 2) + 2 * w + 3];
                tmp += weight_buf_c[14] * input_buf_c[input_w * (input_h - 2) + 2 * w + 4];
                tmp += weight_buf_c[15] * input_buf_c[input_w * (input_h - 1) + 2 * w];
                tmp += weight_buf_c[16] * input_buf_c[input_w * (input_h - 1) + 2 * w + 1];
                tmp += weight_buf_c[17] * input_buf_c[input_w * (input_h - 1) + 2 * w + 2];
                tmp += weight_buf_c[18] * input_buf_c[input_w * (input_h - 1) + 2 * w + 3];
                tmp += weight_buf_c[19] * input_buf_c[input_w * (input_h - 1) + 2 * w + 4];
                output_buf_c[output_w * (output_h - 1) + w + 1] = elem_activation(tmp, activation);
            }
            if(w_remain)
            {
                tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[input_w * (input_h - 4) + 2 * w];
                tmp += weight_buf_c[1] * input_buf_c[input_w * (input_h - 4) + 2 * w + 1];
                tmp += weight_buf_c[2] * input_buf_c[input_w * (input_h - 4) + 2 * w + 2];
                tmp += weight_buf_c[5] * input_buf_c[input_w * (input_h - 3) + 2 * w];
                tmp += weight_buf_c[6] * input_buf_c[input_w * (input_h - 3) + 2 * w + 1];
                tmp += weight_buf_c[7] * input_buf_c[input_w * (input_h - 3) + 2 * w + 2];
                tmp += weight_buf_c[10] * input_buf_c[input_w * (input_h - 2) + 2 * w];
                tmp += weight_buf_c[11] * input_buf_c[input_w * (input_h - 2) + 2 * w + 1];
                tmp += weight_buf_c[12] * input_buf_c[input_w * (input_h - 2) + 2 * w + 2];
                tmp += weight_buf_c[15] * input_buf_c[input_w * (input_h - 1) + 2 * w];
                tmp += weight_buf_c[16] * input_buf_c[input_w * (input_h - 1) + 2 * w + 1];
                tmp += weight_buf_c[17] * input_buf_c[input_w * (input_h - 1) + 2 * w + 2];
                output_buf_c[output_hw - 1] = elem_activation(tmp, activation);
            }
            else
            {
                tmp = bias_c;
                tmp += weight_buf_c[0] * input_buf_c[input_w * (input_h - 4) + 2 * w];
                tmp += weight_buf_c[1] * input_buf_c[input_w * (input_h - 4) + 2 * w + 1];
                tmp += weight_buf_c[2] * input_buf_c[input_w * (input_h - 4) + 2 * w + 2];
                tmp += weight_buf_c[3] * input_buf_c[input_w * (input_h - 4) + 2 * w + 3];
                tmp += weight_buf_c[5] * input_buf_c[input_w * (input_h - 3) + 2 * w];
                tmp += weight_buf_c[6] * input_buf_c[input_w * (input_h - 3) + 2 * w + 1];
                tmp += weight_buf_c[7] * input_buf_c[input_w * (input_h - 3) + 2 * w + 2];
                tmp += weight_buf_c[8] * input_buf_c[input_w * (input_h - 3) + 2 * w + 3];
                tmp += weight_buf_c[10] * input_buf_c[input_w * (input_h - 2) + 2 * w];
                tmp += weight_buf_c[11] * input_buf_c[input_w * (input_h - 2) + 2 * w + 1];
                tmp += weight_buf_c[12] * input_buf_c[input_w * (input_h - 2) + 2 * w + 2];
                tmp += weight_buf_c[13] * input_buf_c[input_w * (input_h - 2) + 2 * w + 3];
                tmp += weight_buf_c[15] * input_buf_c[input_w * (input_h - 1) + 2 * w];
                tmp += weight_buf_c[16] * input_buf_c[input_w * (input_h - 1) + 2 * w + 1];
                tmp += weight_buf_c[17] * input_buf_c[input_w * (input_h - 1) + 2 * w + 2];
                tmp += weight_buf_c[18] * input_buf_c[input_w * (input_h - 1) + 2 * w + 3];
                output_buf_c[output_hw - 1] = elem_activation(tmp, activation);
            }
        }
    }
}

void depthwise_conv_k7s1(float* input, float* weight, float* bias, float* output, int input_h,
                                     int input_w, int channel, int output_h, int output_w, int activation)
{
    int channel_size = input_h * input_w;
    int mid_w = input_w - 6;
    int mid_block = mid_w >> 2;
    int mid_h = input_h - 6;
    int w = 0;
    float tmp0, tmp1, tmp2;
    for(int c = 0; c < channel; c++)
    {
        float* input_1 = input + c * channel_size;
        float* input_2 = input_1 + input_w;
        float* input_3 = input_1 + input_w * 2;
        float* input_4 = input_1 + input_w * 3;
        float* input_5 = input_1 + input_w * 4;
        float* input_6 = input_1 + input_w * 5;
        float* input_7 = input_1 + input_w * 6;
        float* output_buf = output + c * channel_size;
        float* output_buf_1 = output_buf + output_w;
        float* output_buf_2 = output_buf_1 + output_w;
        float* weight_buf = weight + c * 49;
        float bias_c = bias ? bias[c] : 0;
        
        float32x4_t kernel_0_3 = vld1q_f32(weight_buf);
        float32x4_t kernel_4_7 = vld1q_f32(weight_buf + 4);
        float32x4_t kernel_8_11 = vld1q_f32(weight_buf + 8);
        float32x4_t kernel_12_15 = vld1q_f32(weight_buf + 12);
        float32x4_t kernel_16_19 = vld1q_f32(weight_buf + 16);
        float32x4_t kernel_20_23 = vld1q_f32(weight_buf + 20);
        float32x4_t kernel_24_27 = vld1q_f32(weight_buf + 24);
        float32x4_t kernel_28_31 = vld1q_f32(weight_buf + 28);
        float32x4_t kernel_32_35 = vld1q_f32(weight_buf + 32);
        float32x4_t kernel_36_39 = vld1q_f32(weight_buf + 36);
        float32x4_t kernel_40_43 = vld1q_f32(weight_buf + 40);
        float32x4_t kernel_44_47 = vld1q_f32(weight_buf + 44);
        float32x4_t kernel_48_51 = vld1q_f32(weight_buf + 48);
        float32x4_t line1 = vld1q_f32(input_1);
        float32x4_t line2 = vld1q_f32(input_2);
        float32x4_t line3 = vld1q_f32(input_3);
        float32x4_t line4 = vld1q_f32(input_4);
        float32x4_t line5 = vld1q_f32(input_5);
        float32x4_t line6 = vld1q_f32(input_6);

        float32x4_t kernel_10_13 = vextq_f32(kernel_8_11, kernel_12_15, 2);
        float32x4_t kernel_17_20 = vextq_f32(kernel_16_19, kernel_20_23, 1);
        float32x4_t kernel_31_34 = vextq_f32(kernel_28_31, kernel_32_35, 3);
        float32x4_t kernel_38_41 = vextq_f32(kernel_36_39, kernel_40_43, 2);
        float32x4_t kernel_45_48 = vextq_f32(kernel_44_47, kernel_48_51, 1);
        
        float32x4_t line1_1 = vld1q_f32(input_1 + 4);
        float32x4_t line2_1 = vld1q_f32(input_2 + 4);
        float32x4_t line3_1 = vld1q_f32(input_3 + 4);
        float32x4_t line4_1 = vld1q_f32(input_4 + 4);
        float32x4_t line5_1 = vld1q_f32(input_5 + 4);
        float32x4_t line6_1 = vld1q_f32(input_6 + 4);
        /* top start1 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_24_27);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_31_34);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_38_41);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_45_48);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_17_20);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_24_27);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_31_34);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_38_41);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_45_48);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
            float32x4_t tmp_4_2 = vmulq_f32(line1, kernel_10_13);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line2, kernel_17_20);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line3, kernel_24_27);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_31_34);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_38_41);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_45_48);
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            *output_buf_2++ = elem_activation(tmp2, activation);
        }
        float32x4_t kernel_9_12 = vextq_f32(kernel_8_11, kernel_12_15, 1);
        float32x4_t kernel_23_26 = vextq_f32(kernel_20_23, kernel_24_27, 3);
        float32x4_t kernel_30_33 = vextq_f32(kernel_28_31, kernel_32_35, 2);
        float32x4_t kernel_37_40 = vextq_f32(kernel_36_39, kernel_40_43, 1);
        /* top start2 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_23_26);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_30_33);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_37_40);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_44_47);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            tmp0 += weight_buf[27] * input_1[4];
            tmp0 += weight_buf[34] * input_2[4];
            tmp0 += weight_buf[41] * input_3[4];
            tmp0 += weight_buf[48] * input_4[4];
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_16_19);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_23_26);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_30_33);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_37_40);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_44_47);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            tmp1 += weight_buf[20] * input_1[4];
            tmp1 += weight_buf[27] * input_2[4];
            tmp1 += weight_buf[34] * input_3[4];
            tmp1 += weight_buf[41] * input_4[4];
            tmp1 += weight_buf[48] * input_5[4];
            *output_buf_1++ = elem_activation(tmp1, activation);
            float32x4_t tmp_4_2 = vmulq_f32(line1, kernel_9_12);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line2, kernel_16_19);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line3, kernel_23_26);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_30_33);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_37_40);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_44_47);
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            tmp2 += weight_buf[13] * input_1[4];
            tmp2 += weight_buf[20] * input_2[4];
            tmp2 += weight_buf[27] * input_3[4];
            tmp2 += weight_buf[34] * input_4[4];
            tmp2 += weight_buf[41] * input_5[4];
            tmp2 += weight_buf[48] * input_6[4];
            *output_buf_2++ = elem_activation(tmp2, activation);
        }
        float32x4_t kernel_15_18 = vextq_f32(kernel_12_15, kernel_16_19, 3);
        float32x4_t kernel_22_25 = vextq_f32(kernel_20_23, kernel_24_27, 2);
        float32x4_t kernel_29_32 = vextq_f32(kernel_28_31, kernel_32_35, 1);
        float32x4_t kernel_43_46 = vextq_f32(kernel_40_43, kernel_44_47, 3);
        /* top start3 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_22_25);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_29_32);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_36_39);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_43_46);
            float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_24_27));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_31_34));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_38_41));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_45_48));
            tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);

            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_15_18);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_22_25);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_29_32);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_36_39);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_43_46);
            float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line1_1), vget_high_f32(kernel_17_20));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line2_1), vget_high_f32(kernel_24_27));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_high_f32(kernel_31_34));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_high_f32(kernel_38_41));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_high_f32(kernel_45_48));
            tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);

            float32x4_t tmp_4_2 = vmulq_f32(line1, kernel_8_11);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line2, kernel_15_18);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line3, kernel_22_25);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_29_32);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_36_39);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_43_46);
            float32x2_t tmp_2_2 = vadd_f32(vget_low_f32(tmp_4_2), vget_high_f32(tmp_4_2));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line1_1), vget_high_f32(kernel_10_13));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line2_1), vget_high_f32(kernel_17_20));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line3_1), vget_high_f32(kernel_24_27));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line4_1), vget_high_f32(kernel_31_34));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line5_1), vget_high_f32(kernel_38_41));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line6_1), vget_high_f32(kernel_45_48));
            tmp2 = vget_lane_f32(tmp_2_2, 0) + vget_lane_f32(tmp_2_2, 1) + bias_c;
            *output_buf_2++ = elem_activation(tmp2, activation);
        }
        
        float32x4_t line1_2;
        float32x4_t line2_2;
        float32x4_t line3_2;
        float32x4_t line4_2;
        float32x4_t line5_2;
        float32x4_t line6_2;
        /* top mid */
        for(w = 0; w < mid_block; w++)
        {
            line1_2 = vld1q_f32(input_1 + 8 + 4 * w);
            line2_2 = vld1q_f32(input_2 + 8 + 4 * w);
            line3_2 = vld1q_f32(input_3 + 8 + 4 * w);
            line4_2 = vld1q_f32(input_4 + 8 + 4 * w);
            line5_2 = vld1q_f32(input_5 + 8 + 4 * w);
            line6_2 = vld1q_f32(input_6 + 8 + 4 * w);
            float32x4_t tmp_4_0 = vdupq_n_f32(bias_c);
            float32x4_t tmp_4_1 = vdupq_n_f32(bias_c);
            float32x4_t tmp_4_2 = vdupq_n_f32(bias_c);
            /* line1 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line1, vget_low_f32(kernel_20_23), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line1, vget_high_f32(kernel_12_15), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line1, vget_high_f32(kernel_4_7), 1);
            float32x4_t tmp = vextq_f32(line1, line1_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_12_15), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_8_11), 0);
            tmp = vextq_f32(line1, line1_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_16_19), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_8_11), 1);
            tmp = vextq_f32(line1, line1_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_16_19), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_8_11), 0);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line1_1, vget_low_f32(kernel_24_27), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line1_1, vget_high_f32(kernel_16_19), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line1_1, vget_high_f32(kernel_8_11), 1);
            tmp = vextq_f32(line1_1, line1_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_16_19), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_12_15), 0);
            tmp = vextq_f32(line1_1, line1_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_20_23), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_12_15), 1);
            /* line2 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line2, vget_low_f32(kernel_28_31), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line2, vget_low_f32(kernel_20_23), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line2, vget_high_f32(kernel_12_15), 0);
            tmp = vextq_f32(line2, line2_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_28_31), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_20_23), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_12_15), 1);
            tmp = vextq_f32(line2, line2_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_20_23), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_16_19), 0);
            tmp = vextq_f32(line2, line2_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_24_27), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_16_19), 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line2_1, vget_low_f32(kernel_32_35), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line2_1, vget_low_f32(kernel_24_27), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line2_1, vget_high_f32(kernel_16_19), 0);
            tmp = vextq_f32(line2_1, line2_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_24_27), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_16_19), 1);
            tmp = vextq_f32(line2_1, line2_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_32_35), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_24_27), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_20_23), 0);
            /* line3 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line3, vget_high_f32(kernel_32_35), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line3, vget_low_f32(kernel_28_31), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line3, vget_low_f32(kernel_20_23), 1);
            tmp = vextq_f32(line3, line3_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_28_31), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_20_23), 0);
            tmp = vextq_f32(line3, line3_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_28_31), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_20_23), 1);
            tmp = vextq_f32(line3, line3_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_36_39), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_28_31), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_24_27), 0);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line3_1, vget_high_f32(kernel_36_39), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line3_1, vget_low_f32(kernel_32_35), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line3_1, vget_low_f32(kernel_24_27), 1);
            tmp = vextq_f32(line3_1, line3_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_32_35), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_24_27), 0);
            tmp = vextq_f32(line3_1, line3_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_32_35), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_24_27), 1);
            /* line4 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line4, vget_high_f32(kernel_40_43), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line4, vget_high_f32(kernel_32_35), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line4, vget_low_f32(kernel_28_31), 0);
            tmp = vextq_f32(line4, line4_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_40_43), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_36_39), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_28_31), 1);
            tmp = vextq_f32(line4, line4_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_44_47), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_36_39), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_28_31), 0);
            tmp = vextq_f32(line4, line4_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_44_47), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_36_39), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_28_31), 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line4_1, vget_high_f32(kernel_44_47), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line4_1, vget_high_f32(kernel_36_39), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line4_1, vget_low_f32(kernel_32_35), 0);
            tmp = vextq_f32(line4_1, line4_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_44_47), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_40_43), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_32_35), 1);
            tmp = vextq_f32(line4_1, line4_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_48_51), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_40_43), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_32_35), 0);
            /* line5 */
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line5, vget_high_f32(kernel_40_43), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line5, vget_high_f32(kernel_32_35), 1);
            tmp = vextq_f32(line5, line5_1, 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_40_43), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_36_39), 0);
            tmp = vextq_f32(line5, line5_1, 2);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_44_47), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_36_39), 1);
            tmp = vextq_f32(line5, line5_1, 3);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_44_47), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_36_39), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line5_1, vget_high_f32(kernel_44_47), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line5_1, vget_high_f32(kernel_36_39), 1);
            tmp = vextq_f32(line5_1, line5_2, 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_44_47), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_40_43), 0);
            tmp = vextq_f32(line5_1, line5_2, 2);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_48_51), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_40_43), 1);
            /* line6 */
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line6, vget_high_f32(kernel_40_43), 0);
            tmp = vextq_f32(line6, line6_1, 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_40_43), 1);
            tmp = vextq_f32(line6, line6_1, 2);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_44_47), 0);
            tmp = vextq_f32(line6, line6_1, 3);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_44_47), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line6_1, vget_high_f32(kernel_44_47), 0);
            tmp = vextq_f32(line6_1, line6_2, 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_44_47), 1);
            tmp = vextq_f32(line6_1, line6_2, 2);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_48_51), 0);

            tmp_4_0 = vector_activation(tmp_4_0, activation);
            tmp_4_1 = vector_activation(tmp_4_1, activation);
            tmp_4_2 = vector_activation(tmp_4_2, activation);
            vst1q_f32(output_buf, tmp_4_0);
            vst1q_f32(output_buf_1, tmp_4_1);
            vst1q_f32(output_buf_2, tmp_4_2);
            output_buf += 4;
            output_buf_1 += 4;
            output_buf_2 += 4;
            line1 = line1_1;
            line2 = line2_1;
            line3 = line3_1;
            line4 = line4_1;
            line5 = line5_1;
            line6 = line6_1;
            line1_1 = line1_2;
            line2_1 = line2_2;
            line3_1 = line3_2;
            line4_1 = line4_2;
            line5_1 = line5_2;
            line6_1 = line6_2;
        }
        float32x4_t zero = vdupq_n_f32(0.0);
        float32x4_t kernel_7_10 = vextq_f32(kernel_4_7, kernel_8_11, 3);
        float32x4_t kernel_14_17 = vextq_f32(kernel_12_15, kernel_16_19, 2);
        float32x4_t kernel_21_24 = vextq_f32(kernel_20_23, kernel_24_27, 1);
        float32x4_t kernel_35_38 = vextq_f32(kernel_32_35, kernel_36_39, 3);
        float32x4_t kernel_42_45 = vextq_f32(kernel_40_43, kernel_44_47, 2);
        line1_2 = vld1q_f32(input_1 + 8 + 4 * w);
        line2_2 = vld1q_f32(input_2 + 8 + 4 * w);
        line3_2 = vld1q_f32(input_3 + 8 + 4 * w);
        line4_2 = vld1q_f32(input_4 + 8 + 4 * w);
        line5_2 = vld1q_f32(input_5 + 8 + 4 * w);
        line6_2 = vld1q_f32(input_6 + 8 + 4 * w);
        for(w = mid_block * 4; w < mid_w; w++)
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_21_24);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_28_31);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_35_38);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_42_45);
            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_14_17);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_21_24);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_28_31);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_35_38);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_42_45);
            float32x4_t tmp_4_2 = vmulq_f32(line1, kernel_7_10);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line2, kernel_14_17);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line3, kernel_21_24);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_28_31);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_35_38);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_42_45);
            float32x4_t tmp = vextq_f32(zero, line1_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_24_27);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_17_20);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_10_13);
            tmp = vextq_f32(zero, line2_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_31_34);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_24_27);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_17_20);
            tmp = vextq_f32(zero, line3_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_38_41);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_31_34);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_24_27);
            tmp = vextq_f32(zero, line4_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_45_48);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_38_41);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_31_34);
            tmp = vextq_f32(zero, line5_1, 3);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_45_48);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_38_41);
            tmp = vextq_f32(zero, line6_1, 3);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_45_48);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            *output_buf_1++ = elem_activation(tmp1, activation);
            *output_buf_2++ = elem_activation(tmp2, activation);
            line1 = vextq_f32(line1, line1_1, 1);
            line2 = vextq_f32(line2, line2_1, 1);
            line3 = vextq_f32(line3, line3_1, 1);
            line4 = vextq_f32(line4, line4_1, 1);
            line5 = vextq_f32(line5, line5_1, 1);
            line6 = vextq_f32(line6, line6_1, 1);
            line1_1 = vextq_f32(line1_1, line1_2, 1);
            line2_1 = vextq_f32(line2_1, line2_2, 1);
            line3_1 = vextq_f32(line3_1, line3_2, 1);
            line4_1 = vextq_f32(line4_1, line4_2, 1);
            line5_1 = vextq_f32(line5_1, line5_2, 1);
            line6_1 = vextq_f32(line6_1, line6_2, 1);
        }
        /* top end1 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_21_24);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_28_31);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_35_38);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_42_45);
            float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_23_26));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_30_33));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_37_40));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_44_47));
            tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);

            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_14_17);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_21_24);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_28_31);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_35_38);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_42_45);
            float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line1_1), vget_high_f32(kernel_16_19));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line2_1), vget_high_f32(kernel_23_26));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_high_f32(kernel_30_33));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_high_f32(kernel_37_40));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_high_f32(kernel_44_47));
            tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);

            float32x4_t tmp_4_2 = vmulq_f32(line1, kernel_7_10);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line2, kernel_14_17);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line3, kernel_21_24);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_28_31);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_35_38);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_42_45);
            float32x2_t tmp_2_2 = vadd_f32(vget_low_f32(tmp_4_2), vget_high_f32(tmp_4_2));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line1_1), vget_high_f32(kernel_9_12));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line2_1), vget_high_f32(kernel_16_19));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line3_1), vget_high_f32(kernel_23_26));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line4_1), vget_high_f32(kernel_30_33));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line5_1), vget_high_f32(kernel_37_40));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line6_1), vget_high_f32(kernel_44_47));
            tmp2 = vget_lane_f32(tmp_2_2, 0) + vget_lane_f32(tmp_2_2, 1) + bias_c;
            *output_buf_2++ = elem_activation(tmp2, activation);
            line1 = vextq_f32(line1, line1_1, 1);
            line2 = vextq_f32(line2, line2_1, 1);
            line3 = vextq_f32(line3, line3_1, 1);
            line4 = vextq_f32(line4, line4_1, 1);
            line5 = vextq_f32(line5, line5_1, 1);
            line6 = vextq_f32(line6, line6_1, 1);
            line1_1 = vextq_f32(line1_1, line1_1, 1);
            line2_1 = vextq_f32(line2_1, line2_1, 1);
            line3_1 = vextq_f32(line3_1, line3_1, 1);
            line4_1 = vextq_f32(line4_1, line4_1, 1);
            line5_1 = vextq_f32(line5_1, line5_1, 1);
            line6_1 = vextq_f32(line6_1, line6_1, 1);
        }
        /* top end2 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_21_24);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_28_31);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_35_38);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_42_45);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            tmp0 += vgetq_lane_f32(line1_1, 0) * weight_buf[25];
            tmp0 += vgetq_lane_f32(line2_1, 0) * weight_buf[32];
            tmp0 += vgetq_lane_f32(line3_1, 0) * weight_buf[39];
            tmp0 += vgetq_lane_f32(line4_1, 0) * weight_buf[46];
            *output_buf++ = elem_activation(tmp0, activation);

            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_14_17);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_21_24);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_28_31);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_35_38);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_42_45);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            tmp1 += vgetq_lane_f32(line1_1, 0) * weight_buf[18];
            tmp1 += vgetq_lane_f32(line2_1, 0) * weight_buf[25];
            tmp1 += vgetq_lane_f32(line3_1, 0) * weight_buf[32];
            tmp1 += vgetq_lane_f32(line4_1, 0) * weight_buf[39];
            tmp1 += vgetq_lane_f32(line5_1, 0) * weight_buf[46];
            *output_buf_1++ = elem_activation(tmp1, activation);

            float32x4_t tmp_4_2 = vmulq_f32(line1, kernel_7_10);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line2, kernel_14_17);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line3, kernel_21_24);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_28_31);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_35_38);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_42_45);
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            tmp2 += vgetq_lane_f32(line1_1, 0) * weight_buf[11];
            tmp2 += vgetq_lane_f32(line2_1, 0) * weight_buf[18];
            tmp2 += vgetq_lane_f32(line3_1, 0) * weight_buf[25];
            tmp2 += vgetq_lane_f32(line4_1, 0) * weight_buf[32];
            tmp2 += vgetq_lane_f32(line5_1, 0) * weight_buf[39];
            tmp2 += vgetq_lane_f32(line6_1, 0) * weight_buf[46];
            *output_buf_2++ = elem_activation(tmp2, activation);
            line1 = vextq_f32(line1, line1_1, 1);
            line2 = vextq_f32(line2, line2_1, 1);
            line3 = vextq_f32(line3, line3_1, 1);
            line4 = vextq_f32(line4, line4_1, 1);
            line5 = vextq_f32(line5, line5_1, 1);
            line6 = vextq_f32(line6, line6_1, 1);
        }
        /* top end3 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_21_24);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_28_31);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_35_38);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_42_45);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);

            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_14_17);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_21_24);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_28_31);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_35_38);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_42_45);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);

            float32x4_t tmp_4_2 = vmulq_f32(line1, kernel_7_10);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line2, kernel_14_17);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line3, kernel_21_24);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_28_31);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_35_38);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_42_45);
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            *output_buf_2++ = elem_activation(tmp2, activation);
        }
        float32x4_t kernel_1_4 = vextq_f32(kernel_0_3, kernel_4_7, 1);
        float32x4_t kernel_2_5 = vextq_f32(kernel_0_3, kernel_4_7, 2);
        float32x4_t kernel_3_6 = vextq_f32(kernel_0_3, kernel_4_7, 3);
        output_buf += output_w * 2;
        float32x4_t line7;
        float32x4_t line7_1;
        float32x4_t line7_2;
        /* mid */
        for(int h = 0; h < mid_h; h ++)
        {
            input_1 = input + c * channel_size + h * input_w;
            input_2 = input_1 + input_w;
            input_3 = input_2 + input_w;
            input_4 = input_3 + input_w;
            input_5 = input_4 + input_w;
            input_6 = input_5 + input_w;
            input_7 = input_6 + input_w;
            line1 = vld1q_f32(input_1);
            line2 = vld1q_f32(input_2);
            line3 = vld1q_f32(input_3);
            line4 = vld1q_f32(input_4);
            line5 = vld1q_f32(input_5);
            line6 = vld1q_f32(input_6);
            line7 = vld1q_f32(input_7);
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_3_6);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_10_13);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_17_20);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_24_27);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_31_34);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_38_41);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_45_48);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
            line1_1 = vld1q_f32(input_1 + 4);
            line2_1 = vld1q_f32(input_2 + 4);
            line3_1 = vld1q_f32(input_3 + 4);
            line4_1 = vld1q_f32(input_4 + 4);
            line5_1 = vld1q_f32(input_5 + 4);
            line6_1 = vld1q_f32(input_6 + 4);
            line7_1 = vld1q_f32(input_7 + 4);
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_2_5);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_9_12);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_16_19);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_23_26);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_30_33);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_37_40);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_44_47);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                            
                tmp0 += vgetq_lane_f32(line1_1, 0) * weight_buf[6];
                tmp0 += vgetq_lane_f32(line2_1, 0) * weight_buf[13];
                tmp0 += vgetq_lane_f32(line3_1, 0) * weight_buf[20];
                tmp0 += vgetq_lane_f32(line4_1, 0) * weight_buf[27];
                tmp0 += vgetq_lane_f32(line5_1, 0) * weight_buf[34];
                tmp0 += vgetq_lane_f32(line6_1, 0) * weight_buf[41];
                tmp0 += vgetq_lane_f32(line7_1, 0) * weight_buf[48];
                *output_buf++ = elem_activation(tmp0, activation);
            }
            {
                
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_1_4);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_8_11);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_15_18);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_22_25);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_29_32);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_36_39);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_43_46);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_3_6));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_10_13));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_17_20));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_24_27));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_31_34));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_high_f32(kernel_38_41));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line7_1), vget_high_f32(kernel_45_48));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
            for(w = 0; w < mid_block; w++)
            {
                line1_2 = vld1q_f32(input_1 + 8 + 4 * w);
                line2_2 = vld1q_f32(input_2 + 8 + 4 * w);
                line3_2 = vld1q_f32(input_3 + 8 + 4 * w);
                line4_2 = vld1q_f32(input_4 + 8 + 4 * w);
                line5_2 = vld1q_f32(input_5 + 8 + 4 * w);
                line6_2 = vld1q_f32(input_6 + 8 + 4 * w);
                line7_2 = vld1q_f32(input_7 + 8 + 4 * w);
                float32x4_t tmp_4_0 = vdupq_n_f32(bias_c);
                /* line1 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line1, vget_low_f32(kernel_0_3), 0);
                float32x4_t tmp = vextq_f32(line1, line1_1, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_0_3), 1);
                tmp = vextq_f32(line1, line1_1, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 0);
                tmp = vextq_f32(line1, line1_1, 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line1_1, vget_low_f32(kernel_4_7), 0);
                tmp = vextq_f32(line1_1, line1_2, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_4_7), 1);
                tmp = vextq_f32(line1_1, line1_2, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_4_7), 0);
                /* line2 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line2, vget_high_f32(kernel_4_7), 1);
                tmp = vextq_f32(line2, line2_1, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 0);
                tmp = vextq_f32(line2, line2_1, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 1);
                tmp = vextq_f32(line2, line2_1, 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_8_11), 0);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line2_1, vget_high_f32(kernel_8_11), 1);
                tmp = vextq_f32(line2_1, line2_2, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 0);
                tmp = vextq_f32(line2_1, line2_2, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 1);
                /* line3 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line3, vget_high_f32(kernel_12_15), 0);
                tmp = vextq_f32(line3, line3_1, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_12_15), 1);
                tmp = vextq_f32(line3, line3_1, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 0);
                tmp = vextq_f32(line3, line3_1, 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line3_1, vget_high_f32(kernel_16_19), 0);
                tmp = vextq_f32(line3_1, line3_2, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_16_19), 1);
                tmp = vextq_f32(line3_1, line3_2, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_20_23), 0);
                /* line4 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line4, vget_low_f32(kernel_20_23), 1);
                tmp = vextq_f32(line4, line4_1, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 0);
                tmp = vextq_f32(line4, line4_1, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 1);
                tmp = vextq_f32(line4, line4_1, 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 0);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line4_1, vget_low_f32(kernel_24_27), 1);
                tmp = vextq_f32(line4_1, line4_2, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 0);
                tmp = vextq_f32(line4_1, line4_2, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 1);
                /* line5 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line5, vget_low_f32(kernel_28_31), 0);
                tmp = vextq_f32(line5, line5_1, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_28_31), 1);
                tmp = vextq_f32(line5, line5_1, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 0);
                tmp = vextq_f32(line5, line5_1, 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line5_1, vget_low_f32(kernel_32_35), 0);
                tmp = vextq_f32(line5_1, line5_2, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 1);
                tmp = vextq_f32(line5_1, line5_2, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_32_35), 0);
                /* line6 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line6, vget_high_f32(kernel_32_35), 1);
                tmp = vextq_f32(line6, line6_1, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 0);
                tmp = vextq_f32(line6, line6_1, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 1);
                tmp = vextq_f32(line6, line6_1, 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_36_39), 0);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line6_1, vget_high_f32(kernel_36_39), 1);
                tmp = vextq_f32(line6_1, line6_2, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 0);
                tmp = vextq_f32(line6_1, line6_2, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 1);
                /* line7 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line7, vget_high_f32(kernel_40_43), 0);
                tmp = vextq_f32(line7, line7_1, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_40_43), 1);
                tmp = vextq_f32(line7, line7_1, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_44_47), 0);
                tmp = vextq_f32(line7, line7_1, 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_44_47), 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line7_1, vget_high_f32(kernel_44_47), 0);
                tmp = vextq_f32(line7_1, line7_2, 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_44_47), 1);
                tmp = vextq_f32(line7_1, line7_2, 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_48_51), 0);
                
                tmp_4_0 = vector_activation(tmp_4_0, activation);
                
                vst1q_f32(output_buf, tmp_4_0);
                output_buf += 4;
                line1 = line1_1;
                line2 = line2_1;
                line3 = line3_1;
                line4 = line4_1;
                line5 = line5_1;
                line6 = line6_1;
                line7 = line7_1;
                line1_1 = line1_2;
                line2_1 = line2_2;
                line3_1 = line3_2;
                line4_1 = line4_2;
                line5_1 = line5_2;
                line6_1 = line6_2;
                line7_1 = line7_2;
            }
            
            line1_2 = vld1q_f32(input_1 + 8 + 4 * w);
            line2_2 = vld1q_f32(input_2 + 8 + 4 * w);
            line3_2 = vld1q_f32(input_3 + 8 + 4 * w);
            line4_2 = vld1q_f32(input_4 + 8 + 4 * w);
            line5_2 = vld1q_f32(input_5 + 8 + 4 * w);
            line6_2 = vld1q_f32(input_6 + 8 + 4 * w);
            line7_2 = vld1q_f32(input_7 + 8 + 4 * w);
            for(w = mid_block * 4; w < mid_w; w++)
            {
                    
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_42_45);
                float32x4_t tmp = vextq_f32(zero, line1_1, 3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_3_6);
                tmp = vextq_f32(zero, line2_1, 3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_10_13);
                tmp = vextq_f32(zero, line3_1, 3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_17_20);
                tmp = vextq_f32(zero, line4_1, 3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_24_27);
                tmp = vextq_f32(zero, line5_1, 3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_31_34);
                tmp = vextq_f32(zero, line6_1, 3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_38_41);
                tmp = vextq_f32(zero, line7_1, 3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_45_48);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
                line1 = vextq_f32(line1, line1_1, 1);
                line2 = vextq_f32(line2, line2_1, 1);
                line3 = vextq_f32(line3, line3_1, 1);
                line4 = vextq_f32(line4, line4_1, 1);
                line5 = vextq_f32(line5, line5_1, 1);
                line6 = vextq_f32(line6, line6_1, 1);
                line7 = vextq_f32(line7, line7_1, 1);
                line1_1 = vextq_f32(line1_1, line1_2, 1);
                line2_1 = vextq_f32(line2_1, line2_2, 1);
                line3_1 = vextq_f32(line3_1, line3_2, 1);
                line4_1 = vextq_f32(line4_1, line4_2, 1);
                line5_1 = vextq_f32(line5_1, line5_2, 1);
                line6_1 = vextq_f32(line6_1, line6_2, 1);
                line7_1 = vextq_f32(line7_1, line7_2, 1);
            }
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_42_45);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_2_5));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_9_12));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_16_19));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_23_26));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_30_33));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_high_f32(kernel_37_40));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line7_1), vget_high_f32(kernel_44_47));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
                line1 = vextq_f32(line1, line1_1, 1);
                line2 = vextq_f32(line2, line2_1, 1);
                line3 = vextq_f32(line3, line3_1, 1);
                line4 = vextq_f32(line4, line4_1, 1);
                line5 = vextq_f32(line5, line5_1, 1);
                line6 = vextq_f32(line6, line6_1, 1);
                line7 = vextq_f32(line7, line7_1, 1);
                line1_1 = vextq_f32(line1_1, line1_1, 1);
                line2_1 = vextq_f32(line2_1, line2_1, 1);
                line3_1 = vextq_f32(line3_1, line3_1, 1);
                line4_1 = vextq_f32(line4_1, line4_1, 1);
                line5_1 = vextq_f32(line5_1, line5_1, 1);
                line6_1 = vextq_f32(line6_1, line6_1, 1);
                line7_1 = vextq_f32(line7_1, line7_1, 1);
            }
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_42_45);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                tmp0 += vgetq_lane_f32(line1_1, 0) * weight_buf[4];
                tmp0 += vgetq_lane_f32(line2_1, 0) * weight_buf[11];
                tmp0 += vgetq_lane_f32(line3_1, 0) * weight_buf[18];
                tmp0 += vgetq_lane_f32(line4_1, 0) * weight_buf[25];
                tmp0 += vgetq_lane_f32(line5_1, 0) * weight_buf[32];
                tmp0 += vgetq_lane_f32(line6_1, 0) * weight_buf[39];
                tmp0 += vgetq_lane_f32(line7_1, 0) * weight_buf[46];
                *output_buf++ = elem_activation(tmp0, activation);
                line1 = vextq_f32(line1, line1_1, 1);
                line2 = vextq_f32(line2, line2_1, 1);
                line3 = vextq_f32(line3, line3_1, 1);
                line4 = vextq_f32(line4, line4_1, 1);
                line5 = vextq_f32(line5, line5_1, 1);
                line6 = vextq_f32(line6, line6_1, 1);
                line7 = vextq_f32(line7, line7_1, 1);
            }
            {
                
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_42_45);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
        }
        /* bottom start1 */
        input_1 = input + c * channel_size + input_w * (input_h - 6);
        input_2 = input_1 + input_w;
        input_3 = input_2 + input_w;
        input_4 = input_3 + input_w;
        input_5 = input_4 + input_w;
        input_6 = input_5 + input_w;
        line1 = vld1q_f32(input_1);
        line2 = vld1q_f32(input_2);
        line3 = vld1q_f32(input_3);
        line4 = vld1q_f32(input_4);
        line5 = vld1q_f32(input_5);
        line6 = vld1q_f32(input_6);
        output_buf_1 = output_buf + input_w;
        output_buf_2 = output_buf_1 + input_w;
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_3_6);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_10_13);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_17_20);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_24_27);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_31_34);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_38_41);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line2, kernel_3_6);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_10_13);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_17_20);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_24_27);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_31_34);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
            float32x4_t tmp_4_2 = vmulq_f32(line3, kernel_3_6);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_10_13);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_17_20);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_24_27);
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            *output_buf_2++ = elem_activation(tmp2, activation);
        }
        line1_1 = vld1q_f32(input_1 + 4);
        line2_1 = vld1q_f32(input_2 + 4);
        line3_1 = vld1q_f32(input_3 + 4);
        line4_1 = vld1q_f32(input_4 + 4);
        line5_1 = vld1q_f32(input_5 + 4);
        line6_1 = vld1q_f32(input_6 + 4);
        /* bottom start2 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_2_5);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_9_12);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_16_19);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_23_26);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_30_33);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_37_40);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;                        
            tmp0 += vgetq_lane_f32(line1_1, 0) * weight_buf[6];
            tmp0 += vgetq_lane_f32(line2_1, 0) * weight_buf[13];
            tmp0 += vgetq_lane_f32(line3_1, 0) * weight_buf[20];
            tmp0 += vgetq_lane_f32(line4_1, 0) * weight_buf[27];
            tmp0 += vgetq_lane_f32(line5_1, 0) * weight_buf[34];
            tmp0 += vgetq_lane_f32(line6_1, 0) * weight_buf[41];
            *output_buf++ = elem_activation(tmp0, activation);

            float32x4_t tmp_4_1 = vmulq_f32(line2, kernel_2_5);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_9_12);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_16_19);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_23_26);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_30_33);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            tmp1 += vgetq_lane_f32(line2_1, 0) * weight_buf[6];
            tmp1 += vgetq_lane_f32(line3_1, 0) * weight_buf[13];
            tmp1 += vgetq_lane_f32(line4_1, 0) * weight_buf[20];
            tmp1 += vgetq_lane_f32(line5_1, 0) * weight_buf[27];
            tmp1 += vgetq_lane_f32(line6_1, 0) * weight_buf[34];
            *output_buf_1++ = elem_activation(tmp1, activation);

            float32x4_t tmp_4_2 = vmulq_f32(line3, kernel_2_5);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_9_12);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_16_19);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_23_26);
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            tmp2 += vgetq_lane_f32(line3_1, 0) * weight_buf[6];
            tmp2 += vgetq_lane_f32(line4_1, 0) * weight_buf[13];
            tmp2 += vgetq_lane_f32(line5_1, 0) * weight_buf[20];
            tmp2 += vgetq_lane_f32(line6_1, 0) * weight_buf[27];
            *output_buf_2++ = elem_activation(tmp2, activation);
        }
        /* bottom start3 */
        {
            
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_1_4);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_8_11);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_15_18);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_22_25);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_29_32);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_36_39);
            float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_3_6));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_10_13));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_17_20));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_24_27));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_31_34));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_high_f32(kernel_38_41));
            tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line2, kernel_1_4);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_8_11);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_15_18);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_22_25);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_29_32);
            float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line2_1), vget_high_f32(kernel_3_6));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_high_f32(kernel_10_13));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_high_f32(kernel_17_20));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_high_f32(kernel_24_27));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line6_1), vget_high_f32(kernel_31_34));
            tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
            float32x4_t tmp_4_2 = vmulq_f32(line3, kernel_1_4);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_8_11);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_15_18);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_22_25);
            float32x2_t tmp_2_2 = vadd_f32(vget_low_f32(tmp_4_2), vget_high_f32(tmp_4_2));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line3_1), vget_high_f32(kernel_3_6));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line4_1), vget_high_f32(kernel_10_13));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line5_1), vget_high_f32(kernel_17_20));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line6_1), vget_high_f32(kernel_24_27));
            tmp2 = vget_lane_f32(tmp_2_2, 0) + vget_lane_f32(tmp_2_2, 1) + bias_c;
            *output_buf_2++ = elem_activation(tmp2, activation);
        }
        /* bottom mid */
        for(w = 0; w < mid_block; w++)
        {
            line1_2 = vld1q_f32(input_1 + 8 + 4 * w);
            line2_2 = vld1q_f32(input_2 + 8 + 4 * w);
            line3_2 = vld1q_f32(input_3 + 8 + 4 * w);
            line4_2 = vld1q_f32(input_4 + 8 + 4 * w);
            line5_2 = vld1q_f32(input_5 + 8 + 4 * w);
            line6_2 = vld1q_f32(input_6 + 8 + 4 * w);
            float32x4_t tmp_4_0 = vdupq_n_f32(bias_c);
            float32x4_t tmp_4_1 = vdupq_n_f32(bias_c);
            float32x4_t tmp_4_2 = vdupq_n_f32(bias_c);
            /* line1 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line1, vget_low_f32(kernel_0_3), 0);
            float32x4_t tmp = vextq_f32(line1, line1_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_0_3), 1);
            tmp = vextq_f32(line1, line1_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 0);
            tmp = vextq_f32(line1, line1_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line1_1, vget_low_f32(kernel_4_7), 0);
            tmp = vextq_f32(line1_1, line1_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_4_7), 1);
            tmp = vextq_f32(line1_1, line1_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_4_7), 0);
            /* line2 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line2, vget_high_f32(kernel_4_7), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line2, vget_low_f32(kernel_0_3), 0);
            tmp = vextq_f32(line2, line2_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_0_3), 1);
            tmp = vextq_f32(line2, line2_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_0_3), 0);
            tmp = vextq_f32(line2, line2_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_8_11), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_0_3), 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line2_1, vget_high_f32(kernel_8_11), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line2_1, vget_low_f32(kernel_4_7), 0);
            tmp = vextq_f32(line2_1, line2_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_4_7), 1);
            tmp = vextq_f32(line2_1, line2_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_4_7), 0);
            /* line3 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line3, vget_high_f32(kernel_12_15), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line3, vget_high_f32(kernel_4_7), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line3, vget_low_f32(kernel_0_3), 0);
            tmp = vextq_f32(line3, line3_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_12_15), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_8_11), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_0_3), 1);
            tmp = vextq_f32(line3, line3_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_8_11), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_0_3), 0);
            tmp = vextq_f32(line3, line3_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_8_11), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_0_3), 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line3_1, vget_high_f32(kernel_16_19), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line3_1, vget_high_f32(kernel_8_11), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line3_1, vget_low_f32(kernel_4_7), 0);
            tmp = vextq_f32(line3_1, line3_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_16_19), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_12_15), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_4_7), 1);
            tmp = vextq_f32(line3_1, line3_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_20_23), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_12_15), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_4_7), 0);
            /* line4 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line4, vget_low_f32(kernel_20_23), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line4, vget_high_f32(kernel_12_15), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line4, vget_high_f32(kernel_4_7), 1);
            tmp = vextq_f32(line4, line4_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_12_15), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_8_11), 0);
            tmp = vextq_f32(line4, line4_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_16_19), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_8_11), 1);
            tmp = vextq_f32(line4, line4_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_16_19), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_8_11), 0);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line4_1, vget_low_f32(kernel_24_27), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line4_1, vget_high_f32(kernel_16_19), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line4_1, vget_high_f32(kernel_8_11), 1);
            tmp = vextq_f32(line4_1, line4_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_16_19), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_12_15), 0);
            tmp = vextq_f32(line4_1, line4_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_20_23), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_12_15), 1);
            /* line5 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line5, vget_low_f32(kernel_28_31), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line5, vget_low_f32(kernel_20_23), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line5, vget_high_f32(kernel_12_15), 0);
            tmp = vextq_f32(line5, line5_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_28_31), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_20_23), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_12_15), 1);
            tmp = vextq_f32(line5, line5_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_20_23), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_16_19), 0);
            tmp = vextq_f32(line5, line5_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_24_27), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_16_19), 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line5_1, vget_low_f32(kernel_32_35), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line5_1, vget_low_f32(kernel_24_27), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line5_1, vget_high_f32(kernel_16_19), 0);
            tmp = vextq_f32(line5_1, line5_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_24_27), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_16_19), 1);
            tmp = vextq_f32(line5_1, line5_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_32_35), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_24_27), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_20_23), 0);
            /* line6 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line6, vget_high_f32(kernel_32_35), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line6, vget_low_f32(kernel_28_31), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line6, vget_low_f32(kernel_20_23), 1);
            tmp = vextq_f32(line6, line6_1, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_28_31), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_20_23), 0);
            tmp = vextq_f32(line6, line6_1, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_28_31), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_20_23), 1);
            tmp = vextq_f32(line6, line6_1, 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_36_39), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_28_31), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_low_f32(kernel_24_27), 0);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line6_1, vget_high_f32(kernel_36_39), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line6_1, vget_low_f32(kernel_32_35), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, line6_1, vget_low_f32(kernel_24_27), 1);
            tmp = vextq_f32(line6_1, line6_2, 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_32_35), 1);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_24_27), 0);
            tmp = vextq_f32(line6_1, line6_2, 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_32_35), 0);
            tmp_4_2 = vmlaq_lane_f32(tmp_4_2, tmp, vget_high_f32(kernel_24_27), 1);

            tmp_4_0 = vector_activation(tmp_4_0, activation);
            vst1q_f32(output_buf, tmp_4_0);
            output_buf += 4;

            tmp_4_1 = vector_activation(tmp_4_1, activation);
            vst1q_f32(output_buf_1, tmp_4_1);
            output_buf_1 += 4;

            tmp_4_2 = vector_activation(tmp_4_2, activation);
            vst1q_f32(output_buf_2, tmp_4_2);
            output_buf_2 += 4;

            line1 = line1_1;
            line2 = line2_1;
            line3 = line3_1;
            line4 = line4_1;
            line5 = line5_1;
            line6 = line6_1;
            line1_1 = line1_2;
            line2_1 = line2_2;
            line3_1 = line3_2;
            line4_1 = line4_2;
            line5_1 = line5_2;
            line6_1 = line6_2;
        }

        line1_2 = vld1q_f32(input_1 + 8 + 4 * w);
        line2_2 = vld1q_f32(input_2 + 8 + 4 * w);
        line3_2 = vld1q_f32(input_3 + 8 + 4 * w);
        line4_2 = vld1q_f32(input_4 + 8 + 4 * w);
        line5_2 = vld1q_f32(input_5 + 8 + 4 * w);
        line6_2 = vld1q_f32(input_6 + 8 + 4 * w);
        for(w = mid_block * 4; w < mid_w; w++)
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
            float32x4_t tmp_4_1 = vmulq_f32(line2, kernel_0_3);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_7_10);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_14_17);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_21_24);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_28_31);
            float32x4_t tmp_4_2 = vmulq_f32(line3, kernel_0_3);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_7_10);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_14_17);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_21_24);
            float32x4_t tmp = vextq_f32(zero, line1_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_3_6);
            tmp = vextq_f32(zero, line2_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_10_13);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_3_6);
            tmp = vextq_f32(zero, line3_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_17_20);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_10_13);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_3_6);
            tmp = vextq_f32(zero, line4_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_24_27);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_17_20);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_10_13);
            tmp = vextq_f32(zero, line5_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_31_34);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_24_27);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_17_20);
            tmp = vextq_f32(zero, line6_1, 3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, tmp, kernel_38_41);
            tmp_4_1 = vmlaq_f32(tmp_4_1, tmp, kernel_31_34);
            tmp_4_2 = vmlaq_f32(tmp_4_2, tmp, kernel_24_27);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            *output_buf_2++ = elem_activation(tmp2, activation);
            line1 = vextq_f32(line1, line1_1, 1);
            line2 = vextq_f32(line2, line2_1, 1);
            line3 = vextq_f32(line3, line3_1, 1);
            line4 = vextq_f32(line4, line4_1, 1);
            line5 = vextq_f32(line5, line5_1, 1);
            line6 = vextq_f32(line6, line6_1, 1);
            line1_1 = vextq_f32(line1_1, line1_2, 1);
            line2_1 = vextq_f32(line2_1, line2_2, 1);
            line3_1 = vextq_f32(line3_1, line3_2, 1);
            line4_1 = vextq_f32(line4_1, line4_2, 1);
            line5_1 = vextq_f32(line5_1, line5_2, 1);
            line6_1 = vextq_f32(line6_1, line6_2, 1);
        }
        /* bottom end1 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
            float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_2_5));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_9_12));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_16_19));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_23_26));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_30_33));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_high_f32(kernel_37_40));
            tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line2, kernel_0_3);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_7_10);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_14_17);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_21_24);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_28_31);
            float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line2_1), vget_high_f32(kernel_2_5));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_high_f32(kernel_9_12));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_high_f32(kernel_16_19));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_high_f32(kernel_23_26));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line6_1), vget_high_f32(kernel_30_33));
            tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
            float32x4_t tmp_4_2 = vmulq_f32(line3, kernel_0_3);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_7_10);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_14_17);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_21_24);
            float32x2_t tmp_2_2 = vadd_f32(vget_low_f32(tmp_4_2), vget_high_f32(tmp_4_2));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line3_1), vget_high_f32(kernel_2_5));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line4_1), vget_high_f32(kernel_9_12));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line5_1), vget_high_f32(kernel_16_19));
            tmp_2_2 = vmla_f32(tmp_2_2, vget_low_f32(line6_1), vget_high_f32(kernel_23_26));
            tmp2 = vget_lane_f32(tmp_2_2, 0) + vget_lane_f32(tmp_2_2, 1) + bias_c;
            *output_buf_2++ = elem_activation(tmp2, activation);
            line1 = vextq_f32(line1, line1_1, 1);
            line2 = vextq_f32(line2, line2_1, 1);
            line3 = vextq_f32(line3, line3_1, 1);
            line4 = vextq_f32(line4, line4_1, 1);
            line5 = vextq_f32(line5, line5_1, 1);
            line6 = vextq_f32(line6, line6_1, 1);
            line1_1 = vextq_f32(line1_1, line1_1, 1);
            line2_1 = vextq_f32(line2_1, line2_1, 1);
            line3_1 = vextq_f32(line3_1, line3_1, 1);
            line4_1 = vextq_f32(line4_1, line4_1, 1);
            line5_1 = vextq_f32(line5_1, line5_1, 1);
            line6_1 = vextq_f32(line6_1, line6_1, 1);
        }
        /* bottom end2 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            tmp0 += vgetq_lane_f32(line1_1, 0) * weight_buf[4];
            tmp0 += vgetq_lane_f32(line2_1, 0) * weight_buf[11];
            tmp0 += vgetq_lane_f32(line3_1, 0) * weight_buf[18];
            tmp0 += vgetq_lane_f32(line4_1, 0) * weight_buf[25];
            tmp0 += vgetq_lane_f32(line5_1, 0) * weight_buf[32];
            tmp0 += vgetq_lane_f32(line6_1, 0) * weight_buf[39];
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line2, kernel_0_3);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_7_10);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_14_17);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_21_24);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_28_31);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            tmp1 += vgetq_lane_f32(line2_1, 0) * weight_buf[4];
            tmp1 += vgetq_lane_f32(line3_1, 0) * weight_buf[11];
            tmp1 += vgetq_lane_f32(line4_1, 0) * weight_buf[18];
            tmp1 += vgetq_lane_f32(line5_1, 0) * weight_buf[25];
            tmp1 += vgetq_lane_f32(line6_1, 0) * weight_buf[32];
            *output_buf_1++ = elem_activation(tmp1, activation);
            float32x4_t tmp_4_2 = vmulq_f32(line3, kernel_0_3);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_7_10);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_14_17);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_21_24);
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            tmp2 += vgetq_lane_f32(line3_1, 0) * weight_buf[4];
            tmp2 += vgetq_lane_f32(line4_1, 0) * weight_buf[11];
            tmp2 += vgetq_lane_f32(line5_1, 0) * weight_buf[18];
            tmp2 += vgetq_lane_f32(line6_1, 0) * weight_buf[25];
            *output_buf_2++ = elem_activation(tmp2, activation);
            line1 = vextq_f32(line1, line1_1, 1);
            line2 = vextq_f32(line2, line2_1, 1);
            line3 = vextq_f32(line3, line3_1, 1);
            line4 = vextq_f32(line4, line4_1, 1);
            line5 = vextq_f32(line5, line5_1, 1);
            line6 = vextq_f32(line6, line6_1, 1);
        }
        /* bottom end3 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line2, kernel_0_3);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_7_10);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_14_17);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_21_24);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_28_31);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
            float32x4_t tmp_4_2 = vmulq_f32(line3, kernel_0_3);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line4, kernel_7_10);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line5, kernel_14_17);
            tmp_4_2 = vmlaq_f32(tmp_4_2, line6, kernel_21_24);
            tmp2 = vgetq_lane_f32(tmp_4_2, 0) + vgetq_lane_f32(tmp_4_2, 1) +
                        vgetq_lane_f32(tmp_4_2, 2) + vgetq_lane_f32(tmp_4_2, 3) + bias_c;
            *output_buf_2++ = elem_activation(tmp2, activation);
        }
    }
}

void depthwise_conv_k7s2(float* input, float* weight, float* bias, float* output, int input_h,
                                     int input_w, int channel, int output_h, int output_w, int activation)
{
    int input_hw = input_h * input_w;
    int output_hw = output_h * output_w;
    int mid_w = output_w - 3;
    int mid_h = output_h - 3;
    int remain_h = input_h & 0x1;
    int remain_w = input_w & 0x1;
    if(remain_h)
        mid_h--;
    if(remain_w)
        mid_w--;
    int mid_block = mid_w >> 2;
    int w = 0;
    float tmp0, tmp1;
    for(int c = 0; c < channel; c++)
    {
        float* output_buf = output + c * output_hw;
        float* output_buf_1 = output_buf + output_w;
        float* weight_buf = weight + c * 49;
        float bias_c = bias ? bias[c] : 0;
        float* input_1 = input + c * input_hw;
        float* input_2 = input_1 + input_w;
        float* input_3 = input_2 + input_w;
        float* input_4 = input_3 + input_w;
        float* input_5 = input_4 + input_w;
        float* input_6 = input_5 + input_w;
        float32x4_t kernel_0_3 = vld1q_f32(weight_buf);
        float32x4_t kernel_4_7 = vld1q_f32(weight_buf + 4);
        float32x4_t kernel_8_11 = vld1q_f32(weight_buf + 8);
        float32x4_t kernel_12_15 = vld1q_f32(weight_buf + 12);
        float32x4_t kernel_16_19 = vld1q_f32(weight_buf + 16);
        float32x4_t kernel_20_23 = vld1q_f32(weight_buf + 20);
        float32x4_t kernel_24_27 = vld1q_f32(weight_buf + 24);
        float32x4_t kernel_28_31 = vld1q_f32(weight_buf + 28);
        float32x4_t kernel_32_35 = vld1q_f32(weight_buf + 32);
        float32x4_t kernel_36_39 = vld1q_f32(weight_buf + 36);
        float32x4_t kernel_40_43 = vld1q_f32(weight_buf + 40);
        float32x4_t kernel_44_47 = vld1q_f32(weight_buf + 44);
        float32x4_t kernel_48_51 = vld1q_f32(weight_buf + 48);
        float32x4_t line1 = vld1q_f32(input_1);
        float32x4_t line2 = vld1q_f32(input_2);
        float32x4_t line3 = vld1q_f32(input_3);
        float32x4_t line4 = vld1q_f32(input_4);
        float32x4_t line5 = vld1q_f32(input_5);
        float32x4_t line6 = vld1q_f32(input_6);

        float32x4_t kernel_10_13 = vextq_f32(kernel_8_11, kernel_12_15, 2);
        float32x4_t kernel_17_20 = vextq_f32(kernel_16_19, kernel_20_23, 1);
        float32x4_t kernel_31_34 = vextq_f32(kernel_28_31, kernel_32_35, 3);
        float32x4_t kernel_38_41 = vextq_f32(kernel_36_39, kernel_40_43, 2);
        float32x4_t kernel_45_48 = vextq_f32(kernel_44_47, kernel_48_51, 1);
        /* top left1 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_24_27);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_31_34);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_38_41);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_45_48);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_10_13);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_17_20);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_24_27);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_31_34);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_38_41);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_45_48);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
        }
        
        float32x4_t line1_1 = vld1q_f32(input_1 + 4);
        float32x4_t line2_1 = vld1q_f32(input_2 + 4);
        float32x4_t line3_1 = vld1q_f32(input_3 + 4);
        float32x4_t line4_1 = vld1q_f32(input_4 + 4);
        float32x4_t line5_1 = vld1q_f32(input_5 + 4);
        float32x4_t line6_1 = vld1q_f32(input_6 + 4);
        float32x4_t kernel_15_18 = vextq_f32(kernel_12_15, kernel_16_19, 3);
        float32x4_t kernel_22_25 = vextq_f32(kernel_20_23, kernel_24_27, 2);
        float32x4_t kernel_29_32 = vextq_f32(kernel_28_31, kernel_32_35, 1);
        float32x4_t kernel_43_46 = vextq_f32(kernel_40_43, kernel_44_47, 3);
        /* top left2 */
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_22_25);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_29_32);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_36_39);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_43_46);
            float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_24_27));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_31_34));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_38_41));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_45_48));
            tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);

            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_8_11);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_15_18);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_22_25);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_29_32);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_36_39);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_43_46);
            float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line1_1), vget_high_f32(kernel_10_13));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line2_1), vget_high_f32(kernel_17_20));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_high_f32(kernel_24_27));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_high_f32(kernel_31_34));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_high_f32(kernel_38_41));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line6_1), vget_high_f32(kernel_45_48));
            tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
        }
        
        /* top  mid  */
        float32x4x2_t line_1_01 = vuzpq_f32(line1, line1_1);
        float32x4x2_t line_2_01 = vuzpq_f32(line2, line2_1);
        float32x4x2_t line_3_01 = vuzpq_f32(line3, line3_1);
        float32x4x2_t line_4_01 = vuzpq_f32(line4, line4_1);
        float32x4x2_t line_5_01 = vuzpq_f32(line5, line5_1);
        float32x4x2_t line_6_01 = vuzpq_f32(line6, line6_1);
        for(w = 0; w < mid_block; w++)
        {
            float32x4x2_t line_1_23 = vld2q_f32(input_1 + 8 + 8 * w);
            float32x4x2_t line_2_23 = vld2q_f32(input_2 + 8 + 8 * w);
            float32x4x2_t line_3_23 = vld2q_f32(input_3 + 8 + 8 * w);
            float32x4x2_t line_4_23 = vld2q_f32(input_4 + 8 + 8 * w);
            float32x4x2_t line_5_23 = vld2q_f32(input_5 + 8 + 8 * w);
            float32x4x2_t line_6_23 = vld2q_f32(input_6 + 8 + 8 * w);
            float32x4_t tmp_4_0 = vdupq_n_f32(bias_c);
            float32x4_t tmp_4_1 = vdupq_n_f32(bias_c);
            
            /* line1 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_1_01.val[1], vget_low_f32(kernel_20_23), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_1_01.val[1], vget_high_f32(kernel_4_7), 1);
            float32x4_t tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_8_11), 0);
            tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_8_11), 1);
            tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_8_11), 0);
            tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_8_11), 1);
            tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_12_15), 0);
            tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_12_15), 1);
            /* line2 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_2_01.val[1], vget_low_f32(kernel_28_31), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_2_01.val[1], vget_high_f32(kernel_12_15), 0);
            tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_28_31), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_12_15), 1);
            tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_16_19), 0);
            tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_16_19), 1);
            tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_16_19), 0);
            tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_16_19), 1);
            tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_32_35), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_20_23), 0);
            /* line3 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_3_01.val[1], vget_high_f32(kernel_32_35), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_3_01.val[1], vget_low_f32(kernel_20_23), 1);
            tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_20_23), 0);
            tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_20_23), 1);
            tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_36_39), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_24_27), 0);
            tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_36_39), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_24_27), 1);
            tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_24_27), 0);
            tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_24_27), 1);
            /* line4 */
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_4_01.val[1], vget_high_f32(kernel_40_43), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_4_01.val[1], vget_low_f32(kernel_28_31), 0);
            tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_40_43), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_28_31), 1);
            tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 1);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_44_47), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_28_31), 0);
            tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_44_47), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_28_31), 1);
            tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 2);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_44_47), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_32_35), 0);
            tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_44_47), 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_32_35), 1);
            tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 3);
            tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_48_51), 0);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_32_35), 0);
            /* line5 */
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_5_01.val[1], vget_high_f32(kernel_32_35), 1);
            tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_36_39), 0);
            tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_36_39), 1);
            tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 2);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_36_39), 0);
            tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 2);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_36_39), 1);
            tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 3);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_40_43), 0);
            tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 3);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_40_43), 1);
            /* line6 */
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_6_01.val[1], vget_high_f32(kernel_40_43), 0);
            tmp = vextq_f32(line_6_01.val[0], line_6_23.val[0], 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_40_43), 1);
            tmp = vextq_f32(line_6_01.val[1], line_6_23.val[1], 1);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_44_47), 0);
            tmp = vextq_f32(line_6_01.val[0], line_6_23.val[0], 2);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_44_47), 1);
            tmp = vextq_f32(line_6_01.val[1], line_6_23.val[1], 2);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_44_47), 0);
            tmp = vextq_f32(line_6_01.val[0], line_6_23.val[0], 3);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_44_47), 1);
            tmp = vextq_f32(line_6_01.val[1], line_6_23.val[1], 3);
            tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_48_51), 0);
            tmp_4_0 = vector_activation(tmp_4_0, activation);
            tmp_4_1 = vector_activation(tmp_4_1, activation);
            vst1q_f32(output_buf, tmp_4_0);
            vst1q_f32(output_buf_1, tmp_4_1);
            output_buf += 4;
            output_buf_1 += 4;
            line_1_01 = line_1_23;
            line_2_01 = line_2_23;
            line_3_01 = line_3_23;
            line_4_01 = line_4_23;
            line_5_01 = line_5_23;
            line_6_01 = line_6_23;
        }
        line_1_01 = vzipq_f32(line_1_01.val[0], line_1_01.val[1]);
        line_2_01 = vzipq_f32(line_2_01.val[0], line_2_01.val[1]);
        line_3_01 = vzipq_f32(line_3_01.val[0], line_3_01.val[1]);
        line_4_01 = vzipq_f32(line_4_01.val[0], line_4_01.val[1]);
        line_5_01 = vzipq_f32(line_5_01.val[0], line_5_01.val[1]);
        line_6_01 = vzipq_f32(line_6_01.val[0], line_6_01.val[1]);
        line1 = line_1_01.val[0];
        line1_1 = line_1_01.val[1];
        line2 = line_2_01.val[0];
        line2_1 = line_2_01.val[1];
        line3 = line_3_01.val[0];
        line3_1 = line_3_01.val[1];
        line4 = line_4_01.val[0];
        line4_1 = line_4_01.val[1];
        line5 = line_5_01.val[0];
        line5_1 = line_5_01.val[1];
        line6 = line_6_01.val[0];
        line6_1 = line_6_01.val[1];
        float32x4_t kernel_7_10 = vextq_f32(kernel_4_7, kernel_8_11, 3);
        float32x4_t kernel_14_17 = vextq_f32(kernel_12_15, kernel_16_19, 2);
        float32x4_t kernel_21_24 = vextq_f32(kernel_20_23, kernel_24_27, 1);
        float32x4_t kernel_35_38 = vextq_f32(kernel_32_35, kernel_36_39, 3);
        float32x4_t kernel_42_45 = vextq_f32(kernel_40_43, kernel_44_47, 2);
        float32x4_t zero = vdupq_n_f32(0.0);
        float32x4_t kernel_0789 = vextq_f32(zero, kernel_7_10, 3);
        float32x4_t kernel_0141516 = vextq_f32(zero, kernel_14_17, 3);
        float32x4_t kernel_0212223 = vextq_f32(zero, kernel_21_24, 3);
        float32x4_t kernel_0282930 = vextq_f32(zero, kernel_28_31, 3);
        float32x4_t kernel_0353637 = vextq_f32(zero, kernel_35_38, 3);
        float32x4_t kernel_0424344 = vextq_f32(zero, kernel_42_45, 3);
        for(w = mid_block * 4; w < mid_w; w ++)
        {
            float32x4_t line1_2 = vld1q_f32(input_1 + 8 + 2 * w);
            float32x4_t line2_2 = vld1q_f32(input_2 + 8 + 2 * w);
            float32x4_t line3_2 = vld1q_f32(input_3 + 8 + 2 * w);
            float32x4_t line4_2 = vld1q_f32(input_4 + 8 + 2 * w);
            float32x4_t line5_2 = vld1q_f32(input_5 + 8 + 2 * w);
            float32x4_t line6_2 = vld1q_f32(input_6 + 8 + 2 * w);
            
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0212223);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_0282930);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_0353637);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_0424344);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line1_1, kernel_24_27);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2_1, kernel_31_34);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3_1, kernel_38_41);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4_1, kernel_45_48);
            tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_0789);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_0141516);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_0212223);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_0282930);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_0353637);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_0424344);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line1_1, kernel_10_13);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2_1, kernel_17_20);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3_1, kernel_24_27);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4_1, kernel_31_34);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5_1, kernel_38_41);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6_1, kernel_45_48);
            tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
            
            line1 = vextq_f32(line1, line1_1, 2);
            line2 = vextq_f32(line2, line2_1, 2);
            line3 = vextq_f32(line3, line3_1, 2);
            line4 = vextq_f32(line4, line4_1, 2);
            line5 = vextq_f32(line5, line5_1, 2);
            line6 = vextq_f32(line6, line6_1, 2);
            line1_1 = vextq_f32(line1_1, line1_2, 2);
            line2_1 = vextq_f32(line2_1, line2_2, 2);
            line3_1 = vextq_f32(line3_1, line3_2, 2);
            line4_1 = vextq_f32(line4_1, line4_2, 2);
            line5_1 = vextq_f32(line5_1, line5_2, 2);
            line6_1 = vextq_f32(line6_1, line6_2, 2);
        }
        /* top right */
        if(remain_w)
        {
            float32x4_t kernel_9_12 = vextq_f32(kernel_8_11, kernel_12_15, 1);
            float32x4_t kernel_23_26 = vextq_f32(kernel_20_23, kernel_24_27, 3);
            float32x4_t kernel_30_33 = vextq_f32(kernel_28_31, kernel_32_35, 2);
            float32x4_t kernel_37_40 = vextq_f32(kernel_36_39, kernel_40_43, 1);
            line1 = vextq_f32(line1, line1_1, 1);
            line2 = vextq_f32(line2, line2_1, 1);
            line3 = vextq_f32(line3, line3_1, 1);
            line4 = vextq_f32(line4, line4_1, 1);
            line5 = vextq_f32(line5, line5_1, 1);
            line6 = vextq_f32(line6, line6_1, 1);
            line1_1 = vextq_f32(line1_1, line1_1, 1);
            line2_1 = vextq_f32(line2_1, line2_1, 1);
            line3_1 = vextq_f32(line3_1, line3_1, 1);
            line4_1 = vextq_f32(line4_1, line4_1, 1);
            line5_1 = vextq_f32(line5_1, line5_1, 1);
            line6_1 = vextq_f32(line6_1, line6_1, 1);
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_21_24);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_28_31);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_35_38);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_42_45);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_23_26));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_30_33));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_37_40));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_44_47));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
                
                float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_7_10);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_14_17);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_21_24);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_28_31);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_35_38);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_42_45);
                float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line1_1), vget_high_f32(kernel_9_12));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line2_1), vget_high_f32(kernel_16_19));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_high_f32(kernel_23_26));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_high_f32(kernel_30_33));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_high_f32(kernel_37_40));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line6_1), vget_high_f32(kernel_44_47));
                tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
                *output_buf_1++ = elem_activation(tmp1, activation);
            }

            line1 = vextq_f32(line1, line1_1, 2);
            line2 = vextq_f32(line2, line2_1, 2);
            line3 = vextq_f32(line3, line3_1, 2);
            line4 = vextq_f32(line4, line4_1, 2);
            line5 = vextq_f32(line5, line5_1, 2);
            line6 = vextq_f32(line6, line6_1, 2);
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_21_24);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_28_31);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_35_38);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_42_45);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) + 
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
                float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_7_10);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_14_17);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_21_24);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_28_31);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_35_38);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_42_45);
                tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) + 
                        vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
                *output_buf_1++ = elem_activation(tmp1, activation);
            }
        }
        else
        {
            float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0212223);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_0282930);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_0353637);
            tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_0424344);
            float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_low_f32(kernel_24_27));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_low_f32(kernel_31_34));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_low_f32(kernel_38_41));
            tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_low_f32(kernel_45_48));
            tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
            *output_buf++ = elem_activation(tmp0, activation);
            float32x4_t tmp_4_1 = vmulq_f32(line1, kernel_0789);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line2, kernel_0141516);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line3, kernel_0212223);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_0282930);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_0353637);
            tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_0424344);
            float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line1_1), vget_low_f32(kernel_10_13));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line2_1), vget_low_f32(kernel_17_20));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_low_f32(kernel_24_27));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_low_f32(kernel_31_34));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_low_f32(kernel_38_41));
            tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line6_1), vget_low_f32(kernel_45_48));
            tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
            *output_buf_1++ = elem_activation(tmp1, activation);
        }
        float* input_7;
        output_buf = output_buf_1;
        float32x4_t kernel_3_6 = vextq_f32(kernel_0_3, kernel_4_7, 3);
        float32x4_t kernel_1_4 = vextq_f32(kernel_0_3, kernel_4_7, 1);
        float32x4_t kernel_0012 = vextq_f32(zero, kernel_0_3, 3);
        /*  mid */
        for(int h = 0; h < mid_h; h++)
        {
            input_1 = input + c * input_hw + input_w * (1 + 2 * h);
            input_2 = input_1 + input_w;
            input_3 = input_2 + input_w;
            input_4 = input_3 + input_w;
            input_5 = input_4 + input_w;
            input_6 = input_5 + input_w;
            input_7 = input_6 + input_w;
            line1 = vld1q_f32(input_1);
            line2 = vld1q_f32(input_2);
            line3 = vld1q_f32(input_3);
            line4 = vld1q_f32(input_4);
            line5 = vld1q_f32(input_5);
            line6 = vld1q_f32(input_6);
            float32x4_t line7 = vld1q_f32(input_7);
            /* mid left 1 */
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_3_6);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_10_13);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_17_20);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_24_27);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_31_34);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_38_41);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_45_48);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
            line1_1 = vld1q_f32(input_1 + 4);
            line2_1 = vld1q_f32(input_2 + 4);
            line3_1 = vld1q_f32(input_3 + 4);
            line4_1 = vld1q_f32(input_4 + 4);
            line5_1 = vld1q_f32(input_5 + 4);
            line6_1 = vld1q_f32(input_6 + 4);
            /* mid left 2 */
            float32x4_t line7_1 = vld1q_f32(input_7 + 4);
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_1_4);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_8_11);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_15_18);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_22_25);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_29_32);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_36_39);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_43_46);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_3_6));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_10_13));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_17_20));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_24_27));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_31_34));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_high_f32(kernel_38_41));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line7_1), vget_high_f32(kernel_45_48));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
            
            line_1_01 = vuzpq_f32(line1, line1_1);
            line_2_01 = vuzpq_f32(line2, line2_1);
            line_3_01 = vuzpq_f32(line3, line3_1);
            line_4_01 = vuzpq_f32(line4, line4_1);
            line_5_01 = vuzpq_f32(line5, line5_1);
            line_6_01 = vuzpq_f32(line6, line6_1);
            float32x4x2_t line_7_01 = vuzpq_f32(line7, line7_1);
            /* mid mid */
            for(w = 0; w < mid_block; w++)
            {
                float32x4x2_t line_1_23 = vld2q_f32(input_1 + 8 + 8 * w);
                float32x4x2_t line_2_23 = vld2q_f32(input_2 + 8 + 8 * w);
                float32x4x2_t line_3_23 = vld2q_f32(input_3 + 8 + 8 * w);
                float32x4x2_t line_4_23 = vld2q_f32(input_4 + 8 + 8 * w);
                float32x4x2_t line_5_23 = vld2q_f32(input_5 + 8 + 8 * w);
                float32x4x2_t line_6_23 = vld2q_f32(input_6 + 8 + 8 * w);
                float32x4x2_t line_7_23 = vld2q_f32(input_7 + 8 + 8 * w);
                float32x4_t tmp_4_0 = vdupq_n_f32(bias_c);
                
                /* line1 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_1_01.val[1], vget_low_f32(kernel_0_3), 0);
                float32x4_t tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_0_3), 1);
                tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 0);
                tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 1);
                tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_4_7), 0);
                tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_4_7), 1);
                tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_4_7), 0);
                /* line2 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_2_01.val[1], vget_high_f32(kernel_4_7), 1);
                tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 0);
                tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 1);
                tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_8_11), 0);
                tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_8_11), 1);
                tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 0);
                tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 1);
                /* line3 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_3_01.val[1], vget_high_f32(kernel_12_15), 0);
                tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_12_15), 1);
                tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 0);
                tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 1);
                tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_16_19), 0);
                tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_16_19), 1);
                tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_20_23), 0);
                /* line4 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_4_01.val[1], vget_low_f32(kernel_20_23), 1);
                tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 0);
                tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 1);
                tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 0);
                tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 1);
                tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 0);
                tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 1);
                /* line5 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_5_01.val[1], vget_low_f32(kernel_28_31), 0);
                tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_28_31), 1);
                tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 0);
                tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 1);
                tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 0);
                tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 1);
                tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_32_35), 0);
                /* line6 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_6_01.val[1], vget_high_f32(kernel_32_35), 1);
                tmp = vextq_f32(line_6_01.val[0], line_6_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 0);
                tmp = vextq_f32(line_6_01.val[1], line_6_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 1);
                tmp = vextq_f32(line_6_01.val[0], line_6_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_36_39), 0);
                tmp = vextq_f32(line_6_01.val[1], line_6_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_36_39), 1);
                tmp = vextq_f32(line_6_01.val[0], line_6_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 0);
                tmp = vextq_f32(line_6_01.val[1], line_6_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 1);
                /* line7 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_7_01.val[1], vget_high_f32(kernel_40_43), 0);
                tmp = vextq_f32(line_7_01.val[0], line_7_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_40_43), 1);
                tmp = vextq_f32(line_7_01.val[1], line_7_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_44_47), 0);
                tmp = vextq_f32(line_7_01.val[0], line_7_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_44_47), 1);
                tmp = vextq_f32(line_7_01.val[1], line_7_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_44_47), 0);
                tmp = vextq_f32(line_7_01.val[0], line_7_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_44_47), 1);
                tmp = vextq_f32(line_7_01.val[1], line_7_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_48_51), 0);
                tmp_4_0 = vector_activation(tmp_4_0, activation);
                vst1q_f32(output_buf, tmp_4_0);
                output_buf += 4;
                line_1_01 = line_1_23;
                line_2_01 = line_2_23;
                line_3_01 = line_3_23;
                line_4_01 = line_4_23;
                line_5_01 = line_5_23;
                line_6_01 = line_6_23;
                line_7_01 = line_7_23;
            }
            line_1_01 = vzipq_f32(line_1_01.val[0], line_1_01.val[1]);
            line_2_01 = vzipq_f32(line_2_01.val[0], line_2_01.val[1]);
            line_3_01 = vzipq_f32(line_3_01.val[0], line_3_01.val[1]);
            line_4_01 = vzipq_f32(line_4_01.val[0], line_4_01.val[1]);
            line_5_01 = vzipq_f32(line_5_01.val[0], line_5_01.val[1]);
            line_6_01 = vzipq_f32(line_6_01.val[0], line_6_01.val[1]);
            line_7_01 = vzipq_f32(line_7_01.val[0], line_7_01.val[1]);
            line1 = line_1_01.val[0];
            line1_1 = line_1_01.val[1];
            line2 = line_2_01.val[0];
            line2_1 = line_2_01.val[1];
            line3 = line_3_01.val[0];
            line3_1 = line_3_01.val[1];
            line4 = line_4_01.val[0];
            line4_1 = line_4_01.val[1];
            line5 = line_5_01.val[0];
            line5_1 = line_5_01.val[1];
            line6 = line_6_01.val[0];
            line6_1 = line_6_01.val[1];
            line7 = line_7_01.val[0];
            line7_1 = line_7_01.val[1];
            for(w = mid_block * 4; w < mid_w; w ++)
            {
                float32x4_t line1_2 = vld1q_f32(input_1 + 8 + 2 * w);
                float32x4_t line2_2 = vld1q_f32(input_2 + 8 + 2 * w);
                float32x4_t line3_2 = vld1q_f32(input_3 + 8 + 2 * w);
                float32x4_t line4_2 = vld1q_f32(input_4 + 8 + 2 * w);
                float32x4_t line5_2 = vld1q_f32(input_5 + 8 + 2 * w);
                float32x4_t line6_2 = vld1q_f32(input_6 + 8 + 2 * w);
                float32x4_t line7_2 = vld1q_f32(input_7 + 8 + 2 * w);
                
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0012);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_0789);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_0141516);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_0212223);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_0282930);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_0353637);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_0424344);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line1_1, kernel_3_6);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2_1, kernel_10_13);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3_1, kernel_17_20);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4_1, kernel_24_27);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5_1, kernel_31_34);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6_1, kernel_38_41);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7_1, kernel_45_48);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);

                line1 = vextq_f32(line1, line1_1, 2);
                line2 = vextq_f32(line2, line2_1, 2);
                line3 = vextq_f32(line3, line3_1, 2);
                line4 = vextq_f32(line4, line4_1, 2);
                line5 = vextq_f32(line5, line5_1, 2);
                line6 = vextq_f32(line6, line6_1, 2);
                line7 = vextq_f32(line7, line7_1, 2);
                line1_1 = vextq_f32(line1_1, line1_2, 2);
                line2_1 = vextq_f32(line2_1, line2_2, 2);
                line3_1 = vextq_f32(line3_1, line3_2, 2);
                line4_1 = vextq_f32(line4_1, line4_2, 2);
                line5_1 = vextq_f32(line5_1, line5_2, 2);
                line6_1 = vextq_f32(line6_1, line6_2, 2);
                line7_1 = vextq_f32(line7_1, line7_2, 2);
            }
            /* mid right */
            if(remain_w)
            {
                float32x4_t kernel_9_12 = vextq_f32(kernel_8_11, kernel_12_15, 1);
                float32x4_t kernel_23_26 = vextq_f32(kernel_20_23, kernel_24_27, 3);
                float32x4_t kernel_30_33 = vextq_f32(kernel_28_31, kernel_32_35, 2);
                float32x4_t kernel_37_40 = vextq_f32(kernel_36_39, kernel_40_43, 1);
                line1 = vextq_f32(line1, line1_1, 1);
                line2 = vextq_f32(line2, line2_1, 1);
                line3 = vextq_f32(line3, line3_1, 1);
                line4 = vextq_f32(line4, line4_1, 1);
                line5 = vextq_f32(line5, line5_1, 1);
                line6 = vextq_f32(line6, line6_1, 1);
                line7 = vextq_f32(line7, line7_1, 1);
                line1_1 = vextq_f32(line1_1, line1_1, 1);
                line2_1 = vextq_f32(line2_1, line2_1, 1);
                line3_1 = vextq_f32(line3_1, line3_1, 1);
                line4_1 = vextq_f32(line4_1, line4_1, 1);
                line5_1 = vextq_f32(line5_1, line5_1, 1);
                line6_1 = vextq_f32(line6_1, line6_1, 1);
                line7_1 = vextq_f32(line7_1, line7_1, 1);
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_42_45);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_low_f32(kernel_4_7));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_9_12));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_16_19));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_23_26));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_30_33));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_high_f32(kernel_37_40));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line7_1), vget_high_f32(kernel_44_47));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);

                line1 = vextq_f32(line1, line1_1, 2);
                line2 = vextq_f32(line2, line2_1, 2);
                line3 = vextq_f32(line3, line3_1, 2);
                line4 = vextq_f32(line4, line4_1, 2);
                line5 = vextq_f32(line5, line5_1, 2);
                line6 = vextq_f32(line6, line6_1, 2);
                line7 = vextq_f32(line7, line7_1, 2);
                tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_42_45);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) + 
                        vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
            else
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0012);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_0789);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_0141516);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_0212223);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_0282930);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_0353637);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line7, kernel_0424344);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_low_f32(kernel_3_6));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_low_f32(kernel_10_13));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_low_f32(kernel_17_20));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_low_f32(kernel_24_27));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_low_f32(kernel_31_34));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_low_f32(kernel_38_41));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line7_1), vget_low_f32(kernel_45_48));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
            
        }
        /*  bottom  */
        if(remain_h)
        {
            output_buf_1 = output_buf + output_w;
            input_1 = input + c * input_hw + (input_h - 6) * input_w;
            input_2 = input_1 + input_w;
            input_3 = input_2 + input_w;
            input_4 = input_3 + input_w;
            input_5 = input_4 + input_w;
            input_6 = input_5 + input_w;
            line1 = vld1q_f32(input_1);
            line2 = vld1q_f32(input_2);
            line3 = vld1q_f32(input_3);
            line4 = vld1q_f32(input_4);
            line5 = vld1q_f32(input_5);
            line6 = vld1q_f32(input_6);
            
            /* bottom 1 left */
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_3_6);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_10_13);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_17_20);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_24_27);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_31_34);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_38_41);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
                float32x4_t tmp_4_1 = vmulq_f32(line3, kernel_3_6);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_10_13);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_17_20);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_24_27);
                tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                            vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
                *output_buf_1++ = elem_activation(tmp1, activation);
            }
            line1_1 = vld1q_f32(input_1 + 4);
            line2_1 = vld1q_f32(input_2 + 4);
            line3_1 = vld1q_f32(input_3 + 4);
            line4_1 = vld1q_f32(input_4 + 4);
            line5_1 = vld1q_f32(input_5 + 4);
            line6_1 = vld1q_f32(input_6 + 4);
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_1_4);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_8_11);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_15_18);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_22_25);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_29_32);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_36_39);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_3_6));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_10_13));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_17_20));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_24_27));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_31_34));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_high_f32(kernel_38_41));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
                
                float32x4_t tmp_4_1 = vmulq_f32(line3, kernel_1_4);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_8_11);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_15_18);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_22_25);
                float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_high_f32(kernel_3_6));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_high_f32(kernel_10_13));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_high_f32(kernel_17_20));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line6_1), vget_high_f32(kernel_24_27));
                tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
                *output_buf_1++ = elem_activation(tmp1, activation);
            }
            line_1_01 = vuzpq_f32(line1, line1_1);
            line_2_01 = vuzpq_f32(line2, line2_1);
            line_3_01 = vuzpq_f32(line3, line3_1);
            line_4_01 = vuzpq_f32(line4, line4_1);
            line_5_01 = vuzpq_f32(line5, line5_1);
            line_6_01 = vuzpq_f32(line6, line6_1);
            /* bottom 1 mid  */
            for(w = 0; w < mid_block; w++)
            {
                float32x4x2_t line_1_23 = vld2q_f32(input_1 + 8 + 8 * w);
                float32x4x2_t line_2_23 = vld2q_f32(input_2 + 8 + 8 * w);
                float32x4x2_t line_3_23 = vld2q_f32(input_3 + 8 + 8 * w);
                float32x4x2_t line_4_23 = vld2q_f32(input_4 + 8 + 8 * w);
                float32x4x2_t line_5_23 = vld2q_f32(input_5 + 8 + 8 * w);
                float32x4x2_t line_6_23 = vld2q_f32(input_6 + 8 + 8 * w);
                float32x4_t tmp_4_0 = vdupq_n_f32(bias_c);
                float32x4_t tmp_4_1 = vdupq_n_f32(bias_c);
                
                /* line1 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_1_01.val[1], vget_low_f32(kernel_0_3), 0);
                float32x4_t tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_0_3), 1);
                tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 0);
                tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 1);
                tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_4_7), 0);
                tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_4_7), 1);
                tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_4_7), 0);
                /* line2 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_2_01.val[1], vget_high_f32(kernel_4_7), 1);
                tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 0);
                tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 1);
                tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_8_11), 0);
                tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_8_11), 1);
                tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 0);
                tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 1);
                /* line3 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_3_01.val[1], vget_high_f32(kernel_12_15), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_3_01.val[1], vget_low_f32(kernel_0_3), 0);
                tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_12_15), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_0_3), 1);
                tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_0_3), 0);
                tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_0_3), 1);
                tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_16_19), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_4_7), 0);
                tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_16_19), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_4_7), 1);
                tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_20_23), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_4_7), 0);
                /* line4 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_4_01.val[1], vget_low_f32(kernel_20_23), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_4_01.val[1], vget_high_f32(kernel_4_7), 1);
                tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_8_11), 0);
                tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_8_11), 1);
                tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_8_11), 0);
                tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_8_11), 1);
                tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_12_15), 0);
                tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_12_15), 1);
                /* line5 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_5_01.val[1], vget_low_f32(kernel_28_31), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_5_01.val[1], vget_high_f32(kernel_12_15), 0);
                tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_28_31), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_12_15), 1);
                tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_16_19), 0);
                tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_16_19), 1);
                tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_16_19), 0);
                tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_16_19), 1);
                tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_32_35), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_20_23), 0);
                /* line6 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_6_01.val[1], vget_high_f32(kernel_32_35), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, line_6_01.val[1], vget_low_f32(kernel_20_23), 1);
                tmp = vextq_f32(line_6_01.val[0], line_6_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_20_23), 0);
                tmp = vextq_f32(line_6_01.val[1], line_6_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_36_39), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_20_23), 1);
                tmp = vextq_f32(line_6_01.val[0], line_6_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_36_39), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_24_27), 0);
                tmp = vextq_f32(line_6_01.val[1], line_6_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_36_39), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_low_f32(kernel_24_27), 1);
                tmp = vextq_f32(line_6_01.val[0], line_6_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 0);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_24_27), 0);
                tmp = vextq_f32(line_6_01.val[1], line_6_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_40_43), 1);
                tmp_4_1 = vmlaq_lane_f32(tmp_4_1, tmp, vget_high_f32(kernel_24_27), 1);
                tmp_4_0 = vector_activation(tmp_4_0, activation);
                vst1q_f32(output_buf, tmp_4_0);
                output_buf += 4;
                tmp_4_1 = vector_activation(tmp_4_1, activation);
                vst1q_f32(output_buf_1, tmp_4_1);
                output_buf_1 += 4;
                line_1_01 = line_1_23;
                line_2_01 = line_2_23;
                line_3_01 = line_3_23;
                line_4_01 = line_4_23;
                line_5_01 = line_5_23;
                line_6_01 = line_6_23;
            }
            line_1_01 = vzipq_f32(line_1_01.val[0], line_1_01.val[1]);
            line_2_01 = vzipq_f32(line_2_01.val[0], line_2_01.val[1]);
            line_3_01 = vzipq_f32(line_3_01.val[0], line_3_01.val[1]);
            line_4_01 = vzipq_f32(line_4_01.val[0], line_4_01.val[1]);
            line_5_01 = vzipq_f32(line_5_01.val[0], line_5_01.val[1]);
            line_6_01 = vzipq_f32(line_6_01.val[0], line_6_01.val[1]);
            line1 = line_1_01.val[0];
            line1_1 = line_1_01.val[1];
            line2 = line_2_01.val[0];
            line2_1 = line_2_01.val[1];
            line3 = line_3_01.val[0];
            line3_1 = line_3_01.val[1];
            line4 = line_4_01.val[0];
            line4_1 = line_4_01.val[1];
            line5 = line_5_01.val[0];
            line5_1 = line_5_01.val[1];
            line6 = line_6_01.val[0];
            line6_1 = line_6_01.val[1];
            for(w = mid_block * 4; w < mid_w; w ++)
            {
                float32x4_t line1_2 = vld1q_f32(input_1 + 8 + 2 * w);
                float32x4_t line2_2 = vld1q_f32(input_2 + 8 + 2 * w);
                float32x4_t line3_2 = vld1q_f32(input_3 + 8 + 2 * w);
                float32x4_t line4_2 = vld1q_f32(input_4 + 8 + 2 * w);
                float32x4_t line5_2 = vld1q_f32(input_5 + 8 + 2 * w);
                float32x4_t line6_2 = vld1q_f32(input_6 + 8 + 2 * w);

                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0012);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_0789);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_0141516);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_0212223);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_0282930);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_0353637);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line1_1, kernel_3_6);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2_1, kernel_10_13);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3_1, kernel_17_20);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4_1, kernel_24_27);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5_1, kernel_31_34);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6_1, kernel_38_41);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);

                float32x4_t tmp_4_1 = vmulq_f32(line3, kernel_0012);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_0789);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_0141516);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_0212223);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line3_1, kernel_3_6);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line4_1, kernel_10_13);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line5_1, kernel_17_20);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line6_1, kernel_24_27);
                tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) +
                            vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
                *output_buf_1++ = elem_activation(tmp1, activation);

                line1 = vextq_f32(line1, line1_1, 2);
                line2 = vextq_f32(line2, line2_1, 2);
                line3 = vextq_f32(line3, line3_1, 2);
                line4 = vextq_f32(line4, line4_1, 2);
                line5 = vextq_f32(line5, line5_1, 2);
                line6 = vextq_f32(line6, line6_1, 2);
                line1_1 = vextq_f32(line1_1, line1_2, 2);
                line2_1 = vextq_f32(line2_1, line2_2, 2);
                line3_1 = vextq_f32(line3_1, line3_2, 2);
                line4_1 = vextq_f32(line4_1, line4_2, 2);
                line5_1 = vextq_f32(line5_1, line5_2, 2);
                line6_1 = vextq_f32(line6_1, line6_2, 2);
            }
            /* bottom 1 right */
            if(remain_w)
            {
                float32x4_t kernel_9_12 = vextq_f32(kernel_8_11, kernel_12_15, 1);
                float32x4_t kernel_23_26 = vextq_f32(kernel_20_23, kernel_24_27, 3);
                float32x4_t kernel_30_33 = vextq_f32(kernel_28_31, kernel_32_35, 2);
                float32x4_t kernel_37_40 = vextq_f32(kernel_36_39, kernel_40_43, 1);
                line1 = vextq_f32(line1, line1_1, 1);
                line2 = vextq_f32(line2, line2_1, 1);
                line3 = vextq_f32(line3, line3_1, 1);
                line4 = vextq_f32(line4, line4_1, 1);
                line5 = vextq_f32(line5, line5_1, 1);
                line6 = vextq_f32(line6, line6_1, 1);
                line1_1 = vextq_f32(line1_1, line1_1, 1);
                line2_1 = vextq_f32(line2_1, line2_1, 1);
                line3_1 = vextq_f32(line3_1, line3_1, 1);
                line4_1 = vextq_f32(line4_1, line4_1, 1);
                line5_1 = vextq_f32(line5_1, line5_1, 1);
                line6_1 = vextq_f32(line6_1, line6_1, 1);
                {
                    float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
                    float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_low_f32(kernel_4_7));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_9_12));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_16_19));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_23_26));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_30_33));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_high_f32(kernel_37_40));
                    tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                    *output_buf++ = elem_activation(tmp0, activation);

                    float32x4_t tmp_4_1 = vmulq_f32(line3, kernel_0_3);
                    tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_7_10);
                    tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_14_17);
                    tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_21_24);
                    float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
                    tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_low_f32(kernel_4_7));
                    tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_high_f32(kernel_9_12));
                    tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_high_f32(kernel_16_19));
                    tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line6_1), vget_high_f32(kernel_23_26));
                    tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
                    *output_buf_1++ = elem_activation(tmp1, activation);
                }

                line1 = vextq_f32(line1, line1_1, 2);
                line2 = vextq_f32(line2, line2_1, 2);
                line3 = vextq_f32(line3, line3_1, 2);
                line4 = vextq_f32(line4, line4_1, 2);
                line5 = vextq_f32(line5, line5_1, 2);
                line6 = vextq_f32(line6, line6_1, 2);
                {
                    float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_35_38);
                    tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) + 
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                    *output_buf++ = elem_activation(tmp0, activation);
                    float32x4_t tmp_4_1 = vmulq_f32(line3, kernel_0_3);
                    tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_7_10);
                    tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_14_17);
                    tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_21_24);
                    tmp1 = vgetq_lane_f32(tmp_4_1, 0) + vgetq_lane_f32(tmp_4_1, 1) + 
                            vgetq_lane_f32(tmp_4_1, 2) + vgetq_lane_f32(tmp_4_1, 3) + bias_c;
                    *output_buf_1++ = elem_activation(tmp1, activation);
                }
            }
            else
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0012);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_0789);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_0141516);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_0212223);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_0282930);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line6, kernel_0353637);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_low_f32(kernel_3_6));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_low_f32(kernel_10_13));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_low_f32(kernel_17_20));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_low_f32(kernel_24_27));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_low_f32(kernel_31_34));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line6_1), vget_low_f32(kernel_38_41));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
                
                float32x4_t tmp_4_1 = vmulq_f32(line3, kernel_0012);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line4, kernel_0789);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line5, kernel_0141516);
                tmp_4_1 = vmlaq_f32(tmp_4_1, line6, kernel_0212223);
                float32x2_t tmp_2_1 = vadd_f32(vget_low_f32(tmp_4_1), vget_high_f32(tmp_4_1));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line3_1), vget_low_f32(kernel_3_6));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line4_1), vget_low_f32(kernel_10_13));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line5_1), vget_low_f32(kernel_17_20));
                tmp_2_1 = vmla_f32(tmp_2_1, vget_low_f32(line6_1), vget_low_f32(kernel_24_27));
                tmp1 = vget_lane_f32(tmp_2_1, 0) + vget_lane_f32(tmp_2_1, 1) + bias_c;
                *output_buf_1++ = elem_activation(tmp1, activation);
            }
            
        }
        else
        {
            input_1 = input + c * input_hw + (input_h - 5) * input_w;
            input_2 = input_1 + input_w;
            input_3 = input_2 + input_w;
            input_4 = input_3 + input_w;
            input_5 = input_4 + input_w;
            line1 = vld1q_f32(input_1);
            line2 = vld1q_f32(input_2);
            line3 = vld1q_f32(input_3);
            line4 = vld1q_f32(input_4);
            line5 = vld1q_f32(input_5);
            /* bottom 0 left */
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_3_6);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_10_13);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_17_20);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_24_27);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_31_34);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
            line1_1 = vld1q_f32(input_1 + 4);
            line2_1 = vld1q_f32(input_2 + 4);
            line3_1 = vld1q_f32(input_3 + 4);
            line4_1 = vld1q_f32(input_4 + 4);
            line5_1 = vld1q_f32(input_5 + 4);
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_1_4);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_8_11);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_15_18);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_22_25);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_29_32);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_high_f32(kernel_3_6));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_10_13));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_17_20));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_24_27));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_31_34));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
            line_1_01 = vuzpq_f32(line1, line1_1);
            line_2_01 = vuzpq_f32(line2, line2_1);
            line_3_01 = vuzpq_f32(line3, line3_1);
            line_4_01 = vuzpq_f32(line4, line4_1);
            line_5_01 = vuzpq_f32(line5, line5_1);
            /* bottom 0 mid  */
            for(w = 0; w < mid_block; w++)
            {
                float32x4x2_t line_1_23 = vld2q_f32(input_1 + 8 + 8 * w);
                float32x4x2_t line_2_23 = vld2q_f32(input_2 + 8 + 8 * w);
                float32x4x2_t line_3_23 = vld2q_f32(input_3 + 8 + 8 * w);
                float32x4x2_t line_4_23 = vld2q_f32(input_4 + 8 + 8 * w);
                float32x4x2_t line_5_23 = vld2q_f32(input_5 + 8 + 8 * w);
                float32x4_t tmp_4_0 = vdupq_n_f32(bias_c);
                
                /* line1 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_1_01.val[1], vget_low_f32(kernel_0_3), 0);
                float32x4_t tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_0_3), 1);
                tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 0);
                tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_0_3), 1);
                tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_4_7), 0);
                tmp = vextq_f32(line_1_01.val[0], line_1_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_4_7), 1);
                tmp = vextq_f32(line_1_01.val[1], line_1_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_4_7), 0);
                /* line2 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_2_01.val[1], vget_high_f32(kernel_4_7), 1);
                tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 0);
                tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_8_11), 1);
                tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_8_11), 0);
                tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_8_11), 1);
                tmp = vextq_f32(line_2_01.val[0], line_2_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 0);
                tmp = vextq_f32(line_2_01.val[1], line_2_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_12_15), 1);
                /* line3 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_3_01.val[1], vget_high_f32(kernel_12_15), 0);
                tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_12_15), 1);
                tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 0);
                tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_16_19), 1);
                tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_16_19), 0);
                tmp = vextq_f32(line_3_01.val[0], line_3_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_16_19), 1);
                tmp = vextq_f32(line_3_01.val[1], line_3_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_20_23), 0);
                /* line4 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_4_01.val[1], vget_low_f32(kernel_20_23), 1);
                tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 0);
                tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_20_23), 1);
                tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 0);
                tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_24_27), 1);
                tmp = vextq_f32(line_4_01.val[0], line_4_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 0);
                tmp = vextq_f32(line_4_01.val[1], line_4_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_24_27), 1);
                /* line5 */
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, line_5_01.val[1], vget_low_f32(kernel_28_31), 0);
                tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_28_31), 1);
                tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 1);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 0);
                tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_28_31), 1);
                tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 2);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 0);
                tmp = vextq_f32(line_5_01.val[0], line_5_23.val[0], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_low_f32(kernel_32_35), 1);
                tmp = vextq_f32(line_5_01.val[1], line_5_23.val[1], 3);
                tmp_4_0 = vmlaq_lane_f32(tmp_4_0, tmp, vget_high_f32(kernel_32_35), 0);
                tmp_4_0 = vector_activation(tmp_4_0, activation);
                vst1q_f32(output_buf, tmp_4_0);
                output_buf += 4;
                line_1_01 = line_1_23;
                line_2_01 = line_2_23;
                line_3_01 = line_3_23;
                line_4_01 = line_4_23;
                line_5_01 = line_5_23;
            }
            line_1_01 = vzipq_f32(line_1_01.val[0], line_1_01.val[1]);
            line_2_01 = vzipq_f32(line_2_01.val[0], line_2_01.val[1]);
            line_3_01 = vzipq_f32(line_3_01.val[0], line_3_01.val[1]);
            line_4_01 = vzipq_f32(line_4_01.val[0], line_4_01.val[1]);
            line_5_01 = vzipq_f32(line_5_01.val[0], line_5_01.val[1]);
            line1 = line_1_01.val[0];
            line1_1 = line_1_01.val[1];
            line2 = line_2_01.val[0];
            line2_1 = line_2_01.val[1];
            line3 = line_3_01.val[0];
            line3_1 = line_3_01.val[1];
            line4 = line_4_01.val[0];
            line4_1 = line_4_01.val[1];
            line5 = line_5_01.val[0];
            line5_1 = line_5_01.val[1];
            for(w = mid_block * 4; w < mid_w; w ++)
            {
                float32x4_t line1_2 = vld1q_f32(input_1 + 8 + 2 * w);
                float32x4_t line2_2 = vld1q_f32(input_2 + 8 + 2 * w);
                float32x4_t line3_2 = vld1q_f32(input_3 + 8 + 2 * w);
                float32x4_t line4_2 = vld1q_f32(input_4 + 8 + 2 * w);
                float32x4_t line5_2 = vld1q_f32(input_5 + 8 + 2 * w);
                
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0012);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_0789);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_0141516);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_0212223);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_0282930);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line1_1, kernel_3_6);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2_1, kernel_10_13);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3_1, kernel_17_20);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4_1, kernel_24_27);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5_1, kernel_31_34);
                tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) +
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);

                line1 = vextq_f32(line1, line1_1, 2);
                line2 = vextq_f32(line2, line2_1, 2);
                line3 = vextq_f32(line3, line3_1, 2);
                line4 = vextq_f32(line4, line4_1, 2);
                line5 = vextq_f32(line5, line5_1, 2);
                line1_1 = vextq_f32(line1_1, line1_2, 2);
                line2_1 = vextq_f32(line2_1, line2_2, 2);
                line3_1 = vextq_f32(line3_1, line3_2, 2);
                line4_1 = vextq_f32(line4_1, line4_2, 2);
                line5_1 = vextq_f32(line5_1, line5_2, 2);
            }
            /*  bottom 0 right */
            if(remain_w)
            {
                float32x4_t kernel_9_12 = vextq_f32(kernel_8_11, kernel_12_15, 1);
                float32x4_t kernel_23_26 = vextq_f32(kernel_20_23, kernel_24_27, 3);
                float32x4_t kernel_30_33 = vextq_f32(kernel_28_31, kernel_32_35, 2);
                line1 = vextq_f32(line1, line1_1, 1);
                line2 = vextq_f32(line2, line2_1, 1);
                line3 = vextq_f32(line3, line3_1, 1);
                line4 = vextq_f32(line4, line4_1, 1);
                line5 = vextq_f32(line5, line5_1, 1);
                line1_1 = vextq_f32(line1_1, line1_1, 1);
                line2_1 = vextq_f32(line2_1, line2_1, 1);
                line3_1 = vextq_f32(line3_1, line3_1, 1);
                line4_1 = vextq_f32(line4_1, line4_1, 1);
                line5_1 = vextq_f32(line5_1, line5_1, 1);
                {
                    float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                    float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_low_f32(kernel_4_7));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_high_f32(kernel_9_12));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_high_f32(kernel_16_19));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_high_f32(kernel_23_26));
                    tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_high_f32(kernel_30_33));
                    tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                    *output_buf++ = elem_activation(tmp0, activation);
                }

                line1 = vextq_f32(line1, line1_1, 2);
                line2 = vextq_f32(line2, line2_1, 2);
                line3 = vextq_f32(line3, line3_1, 2);
                line4 = vextq_f32(line4, line4_1, 2);
                line5 = vextq_f32(line5, line5_1, 2);
                {
                    float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0_3);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_7_10);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_14_17);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_21_24);
                    tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_28_31);
                    tmp0 = vgetq_lane_f32(tmp_4_0, 0) + vgetq_lane_f32(tmp_4_0, 1) + 
                            vgetq_lane_f32(tmp_4_0, 2) + vgetq_lane_f32(tmp_4_0, 3) + bias_c;
                    *output_buf++ = elem_activation(tmp0, activation);
                }
            }
            else
            {
                float32x4_t tmp_4_0 = vmulq_f32(line1, kernel_0012);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line2, kernel_0789);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line3, kernel_0141516);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line4, kernel_0212223);
                tmp_4_0 = vmlaq_f32(tmp_4_0, line5, kernel_0282930);
                float32x2_t tmp_2_0 = vadd_f32(vget_low_f32(tmp_4_0), vget_high_f32(tmp_4_0));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line1_1), vget_low_f32(kernel_3_6));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line2_1), vget_low_f32(kernel_10_13));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line3_1), vget_low_f32(kernel_17_20));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line4_1), vget_low_f32(kernel_24_27));
                tmp_2_0 = vmla_f32(tmp_2_0, vget_low_f32(line5_1), vget_low_f32(kernel_31_34));
                tmp0 = vget_lane_f32(tmp_2_0, 0) + vget_lane_f32(tmp_2_0, 1) + bias_c;
                *output_buf++ = elem_activation(tmp0, activation);
            }
        }
    }
}

