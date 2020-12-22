/*
 * Author: 1091545398@qq.com
 */
#include "conv_kernel_arm.h"
#include <arm_neon.h>
#define PER_OUT_CHAN 8
#define PER_INPUT_COL 8

static inline void sgemm_8x8(int32_t* biases, int8_t* input, int8_t* kernel, long kernel_size, int8_t* output,
                             long output_xy, int activation, int layout, float* dequant_scales, float outputscales)
{
    int kernel_4 = kernel_size >> 2;
    int remian = kernel_4 << 2;
    int32x4_t out0 = {0, 0, 0, 0}, out1 = {0, 0, 0, 0}, out2 = {0, 0, 0, 0}, out3 = {0, 0, 0, 0}, out4 = {0, 0, 0, 0},
              out5 = {0, 0, 0, 0}, out6 = {0, 0, 0, 0}, out7 = {0, 0, 0, 0}, out8 = {0, 0, 0, 0}, out9 = {0, 0, 0, 0},
              out10 = {0, 0, 0, 0}, out11 = {0, 0, 0, 0}, out12 = {0, 0, 0, 0}, out13 = {0, 0, 0, 0},
              out14 = {0, 0, 0, 0}, out15 = {0, 0, 0, 0};
    int16x8_t col_0, col_1, col_2, col_3, col_4, col_5, col_6, col_7;
    int8x8_t kernel_0, input_0;
    int8_t* kernel_ptr = kernel;
    int8_t* input_ptr = input;
    for (int i = 0; i < kernel_4; i++)
    {
        kernel_0 = vld1_s8(kernel_ptr);
        input_0 = vld1_s8(input_ptr);
        col_0 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 0)));
        col_1 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 1)));
        col_2 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 2)));
        col_3 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 3)));
        col_4 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 4)));
        col_5 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 5)));
        col_6 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 6)));
        col_7 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 7)));

        kernel_0 = vld1_s8(kernel_ptr + 8);
        input_0 = vld1_s8(input_ptr + 8);
        col_0 = vmlal_s8(col_0, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 0)));
        col_1 = vmlal_s8(col_1, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 1)));
        col_2 = vmlal_s8(col_2, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 2)));
        col_3 = vmlal_s8(col_3, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 3)));
        col_4 = vmlal_s8(col_4, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 4)));
        col_5 = vmlal_s8(col_5, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 5)));
        col_6 = vmlal_s8(col_6, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 6)));
        col_7 = vmlal_s8(col_7, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 7)));

        out0 = vaddq_s32(out0, vmovl_s16(vget_low_s16(col_0)));
        out1 = vaddq_s32(out1, vmovl_s16(vget_high_s16(col_0)));
        out2 = vaddq_s32(out2, vmovl_s16(vget_low_s16(col_1)));
        out3 = vaddq_s32(out3, vmovl_s16(vget_high_s16(col_1)));
        out4 = vaddq_s32(out4, vmovl_s16(vget_low_s16(col_2)));
        out5 = vaddq_s32(out5, vmovl_s16(vget_high_s16(col_2)));
        out6 = vaddq_s32(out6, vmovl_s16(vget_low_s16(col_3)));
        out7 = vaddq_s32(out7, vmovl_s16(vget_high_s16(col_3)));
        out8 = vaddq_s32(out8, vmovl_s16(vget_low_s16(col_4)));
        out9 = vaddq_s32(out9, vmovl_s16(vget_high_s16(col_4)));
        out10 = vaddq_s32(out10, vmovl_s16(vget_low_s16(col_5)));
        out11 = vaddq_s32(out11, vmovl_s16(vget_high_s16(col_5)));
        out12 = vaddq_s32(out12, vmovl_s16(vget_low_s16(col_6)));
        out13 = vaddq_s32(out13, vmovl_s16(vget_high_s16(col_6)));
        out14 = vaddq_s32(out14, vmovl_s16(vget_low_s16(col_7)));
        out15 = vaddq_s32(out15, vmovl_s16(vget_high_s16(col_7)));

        kernel_0 = vld1_s8(kernel_ptr + 16);
        input_0 = vld1_s8(input_ptr + 16);
        col_0 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 0)));
        col_1 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 1)));
        col_2 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 2)));
        col_3 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 3)));
        col_4 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 4)));
        col_5 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 5)));
        col_6 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 6)));
        col_7 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 7)));

        kernel_0 = vld1_s8(kernel_ptr + 24);
        input_0 = vld1_s8(input_ptr + 24);
        col_0 = vmlal_s8(col_0, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 0)));
        col_1 = vmlal_s8(col_1, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 1)));
        col_2 = vmlal_s8(col_2, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 2)));
        col_3 = vmlal_s8(col_3, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 3)));
        col_4 = vmlal_s8(col_4, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 4)));
        col_5 = vmlal_s8(col_5, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 5)));
        col_6 = vmlal_s8(col_6, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 6)));
        col_7 = vmlal_s8(col_7, input_0, vdup_n_s8(vget_lane_s8(kernel_0, 7)));

        out0 = vaddq_s32(out0, vmovl_s16(vget_low_s16(col_0)));
        out1 = vaddq_s32(out1, vmovl_s16(vget_high_s16(col_0)));
        out2 = vaddq_s32(out2, vmovl_s16(vget_low_s16(col_1)));
        out3 = vaddq_s32(out3, vmovl_s16(vget_high_s16(col_1)));
        out4 = vaddq_s32(out4, vmovl_s16(vget_low_s16(col_2)));
        out5 = vaddq_s32(out5, vmovl_s16(vget_high_s16(col_2)));
        out6 = vaddq_s32(out6, vmovl_s16(vget_low_s16(col_3)));
        out7 = vaddq_s32(out7, vmovl_s16(vget_high_s16(col_3)));
        out8 = vaddq_s32(out8, vmovl_s16(vget_low_s16(col_4)));
        out9 = vaddq_s32(out9, vmovl_s16(vget_high_s16(col_4)));
        out10 = vaddq_s32(out10, vmovl_s16(vget_low_s16(col_5)));
        out11 = vaddq_s32(out11, vmovl_s16(vget_high_s16(col_5)));
        out12 = vaddq_s32(out12, vmovl_s16(vget_low_s16(col_6)));
        out13 = vaddq_s32(out13, vmovl_s16(vget_high_s16(col_6)));
        out14 = vaddq_s32(out14, vmovl_s16(vget_low_s16(col_7)));
        out15 = vaddq_s32(out15, vmovl_s16(vget_high_s16(col_7)));

        kernel_ptr += 32;
        input_ptr += 32;
    }

    for (int i = remian; i < kernel_size; ++i)
    {
        kernel_0 = vld1_s8(kernel_ptr);
        input_0 = vld1_s8(input_ptr);
        col_0 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 0)));
        col_1 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 1)));
        col_2 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 2)));
        col_3 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 3)));
        col_4 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 4)));
        col_5 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 5)));
        col_6 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 6)));
        col_7 = vmull_s8(input_0, vdup_n_s8(vget_lane_s8(kernel_0, 7)));

        out0 = vaddq_s32(out0, vmovl_s16(vget_low_s16(col_0)));
        out1 = vaddq_s32(out1, vmovl_s16(vget_high_s16(col_0)));
        out2 = vaddq_s32(out2, vmovl_s16(vget_low_s16(col_1)));
        out3 = vaddq_s32(out3, vmovl_s16(vget_high_s16(col_1)));
        out4 = vaddq_s32(out4, vmovl_s16(vget_low_s16(col_2)));
        out5 = vaddq_s32(out5, vmovl_s16(vget_high_s16(col_2)));
        out6 = vaddq_s32(out6, vmovl_s16(vget_low_s16(col_3)));
        out7 = vaddq_s32(out7, vmovl_s16(vget_high_s16(col_3)));
        out8 = vaddq_s32(out8, vmovl_s16(vget_low_s16(col_4)));
        out9 = vaddq_s32(out9, vmovl_s16(vget_high_s16(col_4)));
        out10 = vaddq_s32(out10, vmovl_s16(vget_low_s16(col_5)));
        out11 = vaddq_s32(out11, vmovl_s16(vget_high_s16(col_5)));
        out12 = vaddq_s32(out12, vmovl_s16(vget_low_s16(col_6)));
        out13 = vaddq_s32(out13, vmovl_s16(vget_high_s16(col_6)));
        out14 = vaddq_s32(out14, vmovl_s16(vget_low_s16(col_7)));
        out15 = vaddq_s32(out15, vmovl_s16(vget_high_s16(col_7)));

        input_ptr += 8;
        kernel_ptr += 8;
    }

    if (biases)
    {
        int32x4_t biases0 = vld1q_s32(biases);
        int32x4_t biases1 = vld1q_s32(biases + 4);
        out0 = vaddq_s32(out0, vdupq_n_s32(vgetq_lane_s32(biases0, 0)));
        out1 = vaddq_s32(out1, vdupq_n_s32(vgetq_lane_s32(biases0, 0)));
        out2 = vaddq_s32(out2, vdupq_n_s32(vgetq_lane_s32(biases0, 1)));
        out3 = vaddq_s32(out3, vdupq_n_s32(vgetq_lane_s32(biases0, 1)));
        out4 = vaddq_s32(out4, vdupq_n_s32(vgetq_lane_s32(biases0, 2)));
        out5 = vaddq_s32(out5, vdupq_n_s32(vgetq_lane_s32(biases0, 2)));
        out6 = vaddq_s32(out6, vdupq_n_s32(vgetq_lane_s32(biases0, 3)));
        out7 = vaddq_s32(out7, vdupq_n_s32(vgetq_lane_s32(biases0, 3)));
        out8 = vaddq_s32(out8, vdupq_n_s32(vgetq_lane_s32(biases1, 0)));
        out9 = vaddq_s32(out9, vdupq_n_s32(vgetq_lane_s32(biases1, 0)));
        out10 = vaddq_s32(out10, vdupq_n_s32(vgetq_lane_s32(biases1, 1)));
        out11 = vaddq_s32(out11, vdupq_n_s32(vgetq_lane_s32(biases1, 1)));
        out12 = vaddq_s32(out12, vdupq_n_s32(vgetq_lane_s32(biases1, 2)));
        out13 = vaddq_s32(out13, vdupq_n_s32(vgetq_lane_s32(biases1, 2)));
        out14 = vaddq_s32(out14, vdupq_n_s32(vgetq_lane_s32(biases1, 3)));
        out15 = vaddq_s32(out15, vdupq_n_s32(vgetq_lane_s32(biases1, 3)));
    }

    float32x4_t dsclae0 = vld1q_f32(dequant_scales);
    float32x4_t dsclae1 = vld1q_f32(dequant_scales + 4);
    float32x4_t out0_f = vmulq_n_f32(vcvtq_f32_s32(out0), vgetq_lane_f32(dsclae0, 0));
    float32x4_t out1_f = vmulq_n_f32(vcvtq_f32_s32(out1), vgetq_lane_f32(dsclae0, 0));
    float32x4_t out2_f = vmulq_n_f32(vcvtq_f32_s32(out2), vgetq_lane_f32(dsclae0, 1));
    float32x4_t out3_f = vmulq_n_f32(vcvtq_f32_s32(out3), vgetq_lane_f32(dsclae0, 1));
    float32x4_t out4_f = vmulq_n_f32(vcvtq_f32_s32(out4), vgetq_lane_f32(dsclae0, 2));
    float32x4_t out5_f = vmulq_n_f32(vcvtq_f32_s32(out5), vgetq_lane_f32(dsclae0, 2));
    float32x4_t out6_f = vmulq_n_f32(vcvtq_f32_s32(out6), vgetq_lane_f32(dsclae0, 3));
    float32x4_t out7_f = vmulq_n_f32(vcvtq_f32_s32(out7), vgetq_lane_f32(dsclae0, 3));
    float32x4_t out8_f = vmulq_n_f32(vcvtq_f32_s32(out8), vgetq_lane_f32(dsclae1, 0));
    float32x4_t out9_f = vmulq_n_f32(vcvtq_f32_s32(out9), vgetq_lane_f32(dsclae1, 0));
    float32x4_t out10_f = vmulq_n_f32(vcvtq_f32_s32(out10), vgetq_lane_f32(dsclae1, 1));
    float32x4_t out11_f = vmulq_n_f32(vcvtq_f32_s32(out11), vgetq_lane_f32(dsclae1, 1));
    float32x4_t out12_f = vmulq_n_f32(vcvtq_f32_s32(out12), vgetq_lane_f32(dsclae1, 2));
    float32x4_t out13_f = vmulq_n_f32(vcvtq_f32_s32(out13), vgetq_lane_f32(dsclae1, 2));
    float32x4_t out14_f = vmulq_n_f32(vcvtq_f32_s32(out14), vgetq_lane_f32(dsclae1, 3));
    float32x4_t out15_f = vmulq_n_f32(vcvtq_f32_s32(out15), vgetq_lane_f32(dsclae1, 3));

    float32x4_t f_0 = vdupq_n_f32(0);
    float32x4_t f_1 = vdupq_n_f32(-1);
    float32x4_t f6 = vdupq_n_f32(6);
    float32x4_t f1 = vdupq_n_f32(1);
    if (activation >= 0)
    {
        if (activation != 1)
        {
            out0_f = vmaxq_f32(out0_f, f_0);
            out1_f = vmaxq_f32(out1_f, f_0);
            out2_f = vmaxq_f32(out2_f, f_0);
            out3_f = vmaxq_f32(out3_f, f_0);
            out4_f = vmaxq_f32(out4_f, f_0);
            out5_f = vmaxq_f32(out5_f, f_0);
            out6_f = vmaxq_f32(out6_f, f_0);
            out7_f = vmaxq_f32(out7_f, f_0);
            out8_f = vmaxq_f32(out8_f, f_0);
            out9_f = vmaxq_f32(out9_f, f_0);
            out10_f = vmaxq_f32(out10_f, f_0);
            out11_f = vmaxq_f32(out11_f, f_0);
            out12_f = vmaxq_f32(out12_f, f_0);
            out13_f = vmaxq_f32(out13_f, f_0);
            out14_f = vmaxq_f32(out14_f, f_0);
            out15_f = vmaxq_f32(out15_f, f_0);
        }
        if (activation == 1)
        {
            out0_f = vminq_f32(out0_f, f1);
            out1_f = vminq_f32(out1_f, f1);
            out2_f = vminq_f32(out2_f, f1);
            out3_f = vminq_f32(out3_f, f1);
            out4_f = vminq_f32(out4_f, f1);
            out5_f = vminq_f32(out5_f, f1);
            out6_f = vminq_f32(out6_f, f1);
            out7_f = vminq_f32(out7_f, f1);
            out8_f = vminq_f32(out8_f, f1);
            out9_f = vminq_f32(out9_f, f1);
            out10_f = vminq_f32(out10_f, f1);
            out11_f = vminq_f32(out11_f, f1);
            out12_f = vminq_f32(out12_f, f1);
            out13_f = vminq_f32(out13_f, f1);
            out14_f = vminq_f32(out14_f, f1);
            out15_f = vminq_f32(out15_f, f1);

            out0_f = vmaxq_f32(out0_f, f_1);
            out1_f = vmaxq_f32(out1_f, f_1);
            out2_f = vmaxq_f32(out2_f, f_1);
            out3_f = vmaxq_f32(out3_f, f_1);
            out4_f = vmaxq_f32(out4_f, f_1);
            out5_f = vmaxq_f32(out5_f, f_1);
            out6_f = vmaxq_f32(out6_f, f_1);
            out7_f = vmaxq_f32(out7_f, f_1);
            out8_f = vmaxq_f32(out8_f, f_1);
            out9_f = vmaxq_f32(out9_f, f_1);
            out10_f = vmaxq_f32(out10_f, f_1);
            out11_f = vmaxq_f32(out11_f, f_1);
            out12_f = vmaxq_f32(out12_f, f_1);
            out13_f = vmaxq_f32(out13_f, f_1);
            out14_f = vmaxq_f32(out14_f, f_1);
            out15_f = vmaxq_f32(out15_f, f_1);
        }
        if (activation == 6)
        {
            out0_f = vminq_f32(out0_f, f6);
            out1_f = vminq_f32(out1_f, f6);
            out2_f = vminq_f32(out2_f, f6);
            out3_f = vminq_f32(out3_f, f6);
            out4_f = vminq_f32(out4_f, f6);
            out5_f = vminq_f32(out5_f, f6);
            out6_f = vminq_f32(out6_f, f6);
            out7_f = vminq_f32(out7_f, f6);
            out8_f = vminq_f32(out8_f, f6);
            out9_f = vminq_f32(out9_f, f6);
            out10_f = vminq_f32(out10_f, f6);
            out11_f = vminq_f32(out11_f, f6);
            out12_f = vminq_f32(out12_f, f6);
            out13_f = vminq_f32(out13_f, f6);
            out14_f = vminq_f32(out14_f, f6);
            out15_f = vminq_f32(out15_f, f6);
        }
    }

    float32x4_t f_0_5 = vdupq_n_f32(0.5);
    int32x4_t d127 = vdupq_n_s32(127);
    int32x4_t d_127 = vdupq_n_s32(-127);
    uint32x4_t u_1 = vmovq_n_u32(1);

    out0_f = vmulq_n_f32(out0_f, outputscales);
    out1_f = vmulq_n_f32(out1_f, outputscales);
    out2_f = vmulq_n_f32(out2_f, outputscales);
    out3_f = vmulq_n_f32(out3_f, outputscales);
    out4_f = vmulq_n_f32(out4_f, outputscales);
    out5_f = vmulq_n_f32(out5_f, outputscales);
    out6_f = vmulq_n_f32(out6_f, outputscales);
    out7_f = vmulq_n_f32(out7_f, outputscales);
    out8_f = vmulq_n_f32(out8_f, outputscales);
    out9_f = vmulq_n_f32(out9_f, outputscales);
    out10_f = vmulq_n_f32(out10_f, outputscales);
    out11_f = vmulq_n_f32(out11_f, outputscales);
    out12_f = vmulq_n_f32(out12_f, outputscales);
    out13_f = vmulq_n_f32(out13_f, outputscales);
    out14_f = vmulq_n_f32(out14_f, outputscales);
    out15_f = vmulq_n_f32(out15_f, outputscales);

    /* round pos */
    out0_f = vaddq_f32(out0_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out0_f),u_1)),0.5));
    out1_f = vaddq_f32(out1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out1_f),u_1)),0.5));
    out2_f = vaddq_f32(out2_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out2_f),u_1)),0.5));
    out3_f = vaddq_f32(out3_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out3_f),u_1)),0.5));
    out4_f = vaddq_f32(out4_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out4_f),u_1)),0.5));
    out5_f = vaddq_f32(out5_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out5_f),u_1)),0.5));
    out6_f = vaddq_f32(out6_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out6_f),u_1)),0.5));
    out7_f = vaddq_f32(out7_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out7_f),u_1)),0.5));
    out8_f = vaddq_f32(out8_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out8_f),u_1)),0.5));
    out9_f = vaddq_f32(out9_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out9_f),u_1)),0.5));
    out10_f = vaddq_f32(out10_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out10_f),u_1)),0.5));
    out11_f = vaddq_f32(out11_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out11_f),u_1)),0.5));
    out12_f = vaddq_f32(out12_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out12_f),u_1)),0.5));
    out13_f = vaddq_f32(out13_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out13_f),u_1)),0.5));
    out14_f = vaddq_f32(out14_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out14_f),u_1)),0.5));
    out15_f = vaddq_f32(out15_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f_0,out15_f),u_1)),0.5));

    /* round neg */
    out0_f = vaddq_f32(out0_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out0_f),u_1)),-0.5));
    out1_f = vaddq_f32(out1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out1_f),u_1)),-0.5));
    out2_f = vaddq_f32(out2_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out2_f),u_1)),-0.5));
    out3_f = vaddq_f32(out3_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out3_f),u_1)),-0.5));
    out4_f = vaddq_f32(out4_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out4_f),u_1)),-0.5));
    out5_f = vaddq_f32(out5_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out5_f),u_1)),-0.5));
    out6_f = vaddq_f32(out6_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out6_f),u_1)),-0.5));
    out7_f = vaddq_f32(out7_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out7_f),u_1)),-0.5));
    out8_f = vaddq_f32(out8_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out8_f),u_1)),-0.5));
    out9_f = vaddq_f32(out9_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out9_f),u_1)),-0.5));
    out10_f = vaddq_f32(out10_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out10_f),u_1)),-0.5));
    out11_f = vaddq_f32(out11_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out11_f),u_1)),-0.5));
    out12_f = vaddq_f32(out12_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out12_f),u_1)),-0.5));
    out13_f = vaddq_f32(out13_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out13_f),u_1)),-0.5));
    out14_f = vaddq_f32(out14_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out14_f),u_1)),-0.5));
    out15_f = vaddq_f32(out15_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f_0,out15_f),u_1)),-0.5));

    out0 = vcvtq_s32_f32(out0_f);
    out1 = vcvtq_s32_f32(out1_f);
    out2 = vcvtq_s32_f32(out2_f);
    out3 = vcvtq_s32_f32(out3_f);
    out4 = vcvtq_s32_f32(out4_f);
    out5 = vcvtq_s32_f32(out5_f);
    out6 = vcvtq_s32_f32(out6_f);
    out7 = vcvtq_s32_f32(out7_f);
    out8 = vcvtq_s32_f32(out8_f);
    out9 = vcvtq_s32_f32(out9_f);
    out10 = vcvtq_s32_f32(out10_f);
    out11 = vcvtq_s32_f32(out11_f);
    out12 = vcvtq_s32_f32(out12_f);
    out13 = vcvtq_s32_f32(out13_f);
    out14 = vcvtq_s32_f32(out14_f);
    out15 = vcvtq_s32_f32(out15_f);

    out0 = vminq_s32(d127, vmaxq_s32(out0, d_127));
    out1 = vminq_s32(d127, vmaxq_s32(out1, d_127));
    out2 = vminq_s32(d127, vmaxq_s32(out2, d_127));
    out3 = vminq_s32(d127, vmaxq_s32(out3, d_127));
    out4 = vminq_s32(d127, vmaxq_s32(out4, d_127));
    out5 = vminq_s32(d127, vmaxq_s32(out5, d_127));
    out6 = vminq_s32(d127, vmaxq_s32(out6, d_127));
    out7 = vminq_s32(d127, vmaxq_s32(out7, d_127));
    out8 = vminq_s32(d127, vmaxq_s32(out8, d_127));
    out9 = vminq_s32(d127, vmaxq_s32(out9, d_127));
    out10 = vminq_s32(d127, vmaxq_s32(out10, d_127));
    out11 = vminq_s32(d127, vmaxq_s32(out11, d_127));
    out12 = vminq_s32(d127, vmaxq_s32(out12, d_127));
    out13 = vminq_s32(d127, vmaxq_s32(out13, d_127));
    out14 = vminq_s32(d127, vmaxq_s32(out14, d_127));
    out15 = vminq_s32(d127, vmaxq_s32(out15, d_127));

    int8x8_t col0 = vmovn_s16(vcombine_s16(vmovn_s32(out0), vmovn_s32(out1)));
    int8x8_t col1 = vmovn_s16(vcombine_s16(vmovn_s32(out2), vmovn_s32(out3)));
    int8x8_t col2 = vmovn_s16(vcombine_s16(vmovn_s32(out4), vmovn_s32(out5)));
    int8x8_t col3 = vmovn_s16(vcombine_s16(vmovn_s32(out6), vmovn_s32(out7)));
    int8x8_t col4 = vmovn_s16(vcombine_s16(vmovn_s32(out8), vmovn_s32(out9)));
    int8x8_t col5 = vmovn_s16(vcombine_s16(vmovn_s32(out10), vmovn_s32(out11)));
    int8x8_t col6 = vmovn_s16(vcombine_s16(vmovn_s32(out12), vmovn_s32(out13)));
    int8x8_t col7 = vmovn_s16(vcombine_s16(vmovn_s32(out14), vmovn_s32(out15)));

    vst1_s8(output, col0);
    vst1_s8(output + output_xy, col1);
    vst1_s8(output + 2 * output_xy, col2);
    vst1_s8(output + 3 * output_xy, col3);
    vst1_s8(output + 4 * output_xy, col4);
    vst1_s8(output + 5 * output_xy, col5);
    vst1_s8(output + 6 * output_xy, col6);
    vst1_s8(output + 7 * output_xy, col7);
}

static void sgemm_set(int8_t* col, int8_t* kernel, int32_t* biases, int8_t* output, int kernel_size, int ch_start,
                      int ch_end, int output_xy, int activation, int num_thread, int cpu_affinity,
                      float* dequant_scales, float output_scales)
{
    int nn_outch = ch_end / PER_OUT_CHAN;
    int col_end7 = output_xy & 0x7;
    if (col_end7)
    {
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * PER_OUT_CHAN;

            int32_t* biasptr = biases ? ( int32_t* )(biases + p) : NULL;
            int8_t* kernel_tmp = ( int8_t* )(kernel + p * kernel_size);
            int8_t* output_tmp = ( int8_t* )(output + p * output_xy);
            float* dequant_tmp = dequant_scales + p;

            int col_line = 0;
            for (col_line = 0; col_line + 7 < output_xy; col_line += 8)
            {
                int8_t* col_tmp = ( int8_t* )(col + col_line * kernel_size);
                sgemm_8x8(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, output_xy, activation, 0,
                          dequant_tmp, output_scales);
            }
            {
                int8_t result[64];
                int8_t* col_tmp = ( int8_t* )(col + col_line * kernel_size);

                sgemm_8x8(biasptr, col_tmp, kernel_tmp, kernel_size, result, 8, activation, 0, dequant_tmp,
                          output_scales);
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < (col_end7); j++)
                        *(output + (p + i) * output_xy + col_line + j) = result[(i << 3) + j];
                }
            }
        }
    }
    else
    {
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * PER_OUT_CHAN;

            int32_t* biasptr = biases ? ( int32_t* )(biases + p) : NULL;
            int8_t* kernel_tmp = ( int8_t* )(kernel + p * kernel_size);
            int8_t* output_tmp = ( int8_t* )(output + p * output_xy);
            float* dequant_tmp = dequant_scales + p;

            int col_line = 0;
            for (col_line = 0; col_line + 7 < output_xy; col_line += 8)
            {
                int8_t* col_tmp = ( int8_t* )(col + col_line * kernel_size);
                sgemm_8x8(biasptr, col_tmp, kernel_tmp, kernel_size, output_tmp + col_line, output_xy, activation, 0,
                          dequant_tmp, output_scales);
            }
        }
    }
}

static void im2col(int8_t* input, int8_t* col, int in_c, int in_w, int in_h, int k_w, int k_h, int s_w, int s_h,
                   int d_w, int d_h, int pad_w0, int pad_w1, int pad_h0, int pad_h1, int out_w, int out_h,
                   int num_thread)
{
    if (k_w == 1 && k_h == 1 && s_w == 1 && s_h == 1)
    {
        int kernel_size = in_c;
        int in_xy = in_w * in_h;
        int out_xy = out_w * out_h;
        int col_end_ = out_xy & 7;
        int8_t* cur_col_ = col;
        for (int col_i = 0; col_i < out_xy - 7; col_i += PER_INPUT_COL)
        {
            for (int inp_c = 0; inp_c < kernel_size; ++inp_c)
            {
                for (int oup_i = 0; oup_i < PER_INPUT_COL; ++oup_i)
                {
                    *(cur_col_++) = *(input + in_xy * inp_c + col_i + oup_i);
                }
            }
        }
        int col_i = out_xy & -PER_INPUT_COL;
        int8_t* cur_col;
        // final 8 input
        if (col_end_)
        {
            cur_col = col + col_i * kernel_size;
            for (int col_j = 0; col_j < kernel_size; col_j++)
            {
                for (int i = 0; i < 8; i++)
                {
                    if (i < col_end_)
                        *cur_col++ = *(input + col_j * in_xy + col_i + i);
                    else
                        *cur_col++ = 0;
                }
            }
        }
    }
    else
    {
        int out_xy = out_w * out_h;
        for (int col_i = 0; col_i < out_xy - 7; col_i += 8)
        {
            int kernel_size = k_w * k_h * in_c;
            int in_xy = in_w * in_h;
            int col_end7 = out_xy & 7;
            int8_t* cur_col = col + col_i * kernel_size;
            int cnt_y[8] = {col_i / out_w,       (col_i + 1) / out_w, (col_i + 2) / out_w, (col_i + 3) / out_w,
                            (col_i + 4) / out_w, (col_i + 5) / out_w, (col_i + 6) / out_w, (col_i + 7) / out_w};
            int cnt_x[8] = {col_i - cnt_y[0] * out_w,     col_i - cnt_y[1] * out_w + 1, col_i - cnt_y[2] * out_w + 2,
                            col_i - cnt_y[3] * out_w + 3, col_i - cnt_y[4] * out_w + 4, col_i - cnt_y[5] * out_w + 5,
                            col_i - cnt_y[6] * out_w + 6, col_i - cnt_y[7] * out_w + 7};
            int imx_start[8] = {cnt_x[0] * s_w - pad_w0, cnt_x[1] * s_w - pad_w0, cnt_x[2] * s_w - pad_w0,
                                cnt_x[3] * s_w - pad_w0, cnt_x[4] * s_w - pad_w0, cnt_x[5] * s_w - pad_w0,
                                cnt_x[6] * s_w - pad_w0, cnt_x[7] * s_w - pad_w0};
            int imy_start[8] = {cnt_y[0] * s_h - pad_h0, cnt_y[1] * s_h - pad_h0, cnt_y[2] * s_h - pad_h0,
                                cnt_y[3] * s_h - pad_h0, cnt_y[4] * s_h - pad_h0, cnt_y[5] * s_h - pad_h0,
                                cnt_y[6] * s_h - pad_h0, cnt_y[7] * s_h - pad_h0};
            for (int kch = 0; kch < in_c; kch++)
                for (int ky = 0; ky < (k_h * d_h); ky += d_h)
                    for (int kx = 0; kx < (k_w * d_w); kx += d_w)
                    {
                        int imx[8] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx,
                                      imx_start[4] + kx, imx_start[5] + kx, imx_start[6] + kx, imx_start[7] + kx};
                        int imy[8] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky,
                                      imy_start[4] + ky, imy_start[5] + ky, imy_start[6] + ky, imy_start[7] + ky};
                        for (int i = 0; i < 8; i++)
                        {
                            if (imx[i] >= 0 && imx[i] < in_w && imy[i] >= 0 && imy[i] < in_h)
                                *cur_col++ = *(input + in_xy * kch + in_w * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0;
                        }
                    }
        }
        int col_i = out_xy & -8;
        int8_t* cur_col;
        int kernel_size = k_w * k_h * in_c;
        int in_xy = in_w * in_h;
        int col_end7 = out_xy & 7;
        if (col_end7)
        {
            cur_col = col + col_i * kernel_size;
            int cnt_y[8] = {col_i / out_w,       (col_i + 1) / out_w, (col_i + 2) / out_w, (col_i + 3) / out_w,
                            (col_i + 4) / out_w, (col_i + 5) / out_w, (col_i + 6) / out_w, (col_i + 7) / out_w};
            int cnt_x[8] = {col_i - cnt_y[0] * out_w,     col_i - cnt_y[1] * out_w + 1, col_i - cnt_y[2] * out_w + 2,
                            col_i - cnt_y[3] * out_w + 3, col_i - cnt_y[4] * out_w + 4, col_i - cnt_y[5] * out_w + 5,
                            col_i - cnt_y[6] * out_w + 6, col_i - cnt_y[7] * out_w + 7};
            int imx_start[8] = {cnt_x[0] * s_w - pad_w0, cnt_x[1] * s_w - pad_w0, cnt_x[2] * s_w - pad_w0,
                                cnt_x[3] * s_w - pad_w0, cnt_x[4] * s_w - pad_w0, cnt_x[5] * s_w - pad_w0,
                                cnt_x[6] * s_w - pad_w0, cnt_x[7] * s_w - pad_w0};
            int imy_start[8] = {cnt_y[0] * s_h - pad_h0, cnt_y[1] * s_h - pad_h0, cnt_y[2] * s_h - pad_h0,
                                cnt_y[3] * s_h - pad_h0, cnt_y[4] * s_h - pad_h0, cnt_y[5] * s_h - pad_h0,
                                cnt_y[6] * s_h - pad_h0, cnt_y[7] * s_h - pad_h0};
            for (int kch = 0; kch < in_c; kch++)
                for (int ky = 0; ky < (k_h * d_h); ky += d_h)
                    for (int kx = 0; kx < (k_w * d_w); kx += d_w)
                    {
                        int imx[8] = {imx_start[0] + kx, imx_start[1] + kx, imx_start[2] + kx, imx_start[3] + kx,
                                      imx_start[4] + kx, imx_start[5] + kx, imx_start[6] + kx, imx_start[7] + kx};
                        int imy[8] = {imy_start[0] + ky, imy_start[1] + ky, imy_start[2] + ky, imy_start[3] + ky,
                                      imy_start[4] + ky, imy_start[5] + ky, imy_start[6] + ky, imy_start[7] + ky};
                        for (int i = 0; i < 8; i++)
                        {
                            if (i < col_end7 && imx[i] >= 0 && imx[i] < in_w && imy[i] >= 0 && imy[i] < in_h)
                                *cur_col++ = *(input + in_xy * kch + in_w * imy[i] + imx[i]);
                            else
                                *cur_col++ = 0;
                        }
                    }
        }
    }
}

static void inline interleave_kernel(int8_t* kernel, int8_t* kernel_interleave, int out_chan, int kernelsize)
{
    int8_t* kernel_head[PER_OUT_CHAN];
    int8_t* cur_kernel_intervel = kernel_interleave;
    int i, j, k;
    for (i = 0; i + PER_OUT_CHAN - 1 < out_chan; i = i + PER_OUT_CHAN)
    {
        for (j = 0; j < PER_OUT_CHAN; ++j)
        {
            kernel_head[j] = kernel + (j + i) * kernelsize;
        }
        for (k = 0; k < kernelsize; ++k)
        {
            for (j = 0; j < PER_OUT_CHAN; ++j)
            {
                *(cur_kernel_intervel++) = kernel_head[j][k];
            }
        }
    }
    int remian = out_chan % 8;
    if (remian)
    {
        for (k = 0; k < remian; ++k)
        {
            kernel_head[k] = kernel + kernelsize * (i + k);
        }
        for (j = 0; j < kernelsize; ++j)
        {
            for (k = 0; k < remian; k++)
            {
                *(cur_kernel_intervel++) = kernel_head[k][j];
            }
            for (; k < PER_OUT_CHAN; ++k)
            {
                *(cur_kernel_intervel++) = 0;
            }
        }
    }
}

static inline void interleave(struct ir_tensor* filter, struct conv_priv_info* priv_info, struct conv_param* param)
{
    int group = param->group;
    int out_chan = filter->dims[0] / group;
    int kernel_size = filter->dims[1] * filter->dims[2] * filter->dims[3];

    int kernel_size_g = kernel_size * out_chan;
    int kernel_interleaved_size_g = kernel_size * ((out_chan + 7) & -8);

    int8_t* kernel = ( int8_t* )filter->data;

    int8_t* interleave_buf = ( int8_t* )priv_info->interleave_buffer;
    for (int g = 0; g < group; g++)
    {
        int8_t* cur_kernel = kernel + g * kernel_size_g;
        int8_t* cur_interleave = interleave_buf + g * kernel_interleaved_size_g;
        interleave_kernel(cur_kernel, cur_interleave, out_chan, kernel_size);
    }
}

int int8_conv_hcl_get_shared_mem_size(struct ir_tensor* input_tensor, struct ir_tensor* output,
                                      struct conv_param* param)
{
    int group = param->group;
    int input_chan = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;

    int output_xy = output->dims[2] * output->dims[3];
    int mem_size = kernel_size * ((output_xy + 7) & -8) + 128;

    return mem_size;
}

int int8_conv_hcl_prerun(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor,
                         struct ir_tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param)
{
    int group = param->group;
    int input_chan = param->input_channel / group;
    int kernel_size = input_chan * param->kernel_h * param->kernel_w;
    int output_xy = output_tensor->dims[2] * output_tensor->dims[3];
    int im2col_size = int8_conv_hcl_get_shared_mem_size(input_tensor, output_tensor, param);

    int out_chan = filter_tensor->dims[0] / group;
    int kernel_mem_size = kernel_size * ((out_chan + 7) & -8) * group + 128;

    if (!priv_info->external_im2col_mem)
    {
        void* mem = sys_malloc(im2col_size);
        priv_info->im2col_buffer = mem;
        priv_info->im2col_buffer_size = im2col_size;
    }

    if (!priv_info->external_interleave_mem)
    {
        void* mem = sys_malloc(kernel_mem_size);
        priv_info->interleave_buffer = mem;
        priv_info->interleave_buffer_size = kernel_mem_size;
    }
    interleave(filter_tensor, priv_info, param);
    return 0;
}

int int8_conv_hcl_postrun(struct conv_priv_info* priv_info)
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

int int8_conv_hcl_run(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* bias_tensor,
                      struct ir_tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                      int num_thread, int cpu_affinity)
{
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_h0 = param->pad_h0;
    int pad_h1 = param->pad_h1;
    int pad_w0 = param->pad_w0;
    int pad_w1 = param->pad_w1;
    int activation = param->activation;
    int input_image_size = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];

    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1] / group;
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int input_size = in_c * in_h * in_w;
    int kernel_size = in_c * kernel_h * kernel_w;

    int out_c = output_tensor->dims[1] / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_hw = out_h * out_w;
    int output_size = out_c * out_h * out_w;
    int out_c_align = ((out_c + 7) & -8);
    int output_image_size = output_tensor->dims[1] * output_tensor->dims[2] * output_tensor->dims[3];
    /* about int8 */
    float input_scale = input_tensor->scale;
    float* kernel_scales = filter_tensor->scale_list;
    float output_scale = 1 / output_tensor->scale;
    /* input and kernel scales */
    int dequant_scales_size = group * out_c;
    float* dequant_scales = ( float* )malloc(sizeof(float) * dequant_scales_size);
    for (int i = 0; i < dequant_scales_size; i++)
    {
        dequant_scales[i] = (input_scale * kernel_scales[i]);
    }

    /* buffer addr */
    int8_t* input_buf = ( int8_t* )input_tensor->data;
    int8_t* output_buf = ( int8_t* )output_tensor->data;
    int32_t* biases_buf = NULL;
    if (bias_tensor != NULL)
        biases_buf = ( int32_t* )bias_tensor->data;
    int8_t* col_buf = ( int8_t* )priv_info->im2col_buffer;
    int8_t* interleave_buf = ( int8_t* )priv_info->interleave_buffer;

    for (int n = 0; n < batch; ++n)
    {
        for (int g = 0; g < group; ++g)
        {
            /* im2col */
            int8_t* cur_input = input_buf + n * input_image_size + g * input_size;
            im2col(cur_input, col_buf, in_c, in_w, in_h, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
                   pad_w0, pad_w1, pad_h0, pad_h1, out_w, out_h, num_thread);

            /* gemm */
            int8_t* cur_kernel = interleave_buf + g * kernel_size * out_c_align;
            int8_t* cur_output = output_buf + n * output_image_size + g * output_size;
            int32_t* cur_bias = biases_buf ? (biases_buf + g * out_c) : NULL;
            sgemm_set(col_buf, cur_kernel, cur_bias, cur_output, kernel_size, 0, out_c_align, out_hw, activation,
                      num_thread, cpu_affinity, dequant_scales, output_scale);
        }
    }
    return 0;
}
