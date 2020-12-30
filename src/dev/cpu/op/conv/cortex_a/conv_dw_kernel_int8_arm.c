/*
 * Author: 1091545398@qq.com
 */

#include "conv_dw_kernel_int8_arm.h"
#include "tengine_ir.h"
#include "sys_port.h"
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <arm_neon.h>

static inline signed char float2int8(float v)
{
    int32_t int32 = round(v);
    if (int32 > 127)
        return 127;
    if (int32 < -127)
        return -127;
    return int32;
}

static inline int8_t sum2int8(int sum, float descale, const int32_t bias0, float output_scales, int activation)
{
    float desum = (( float )(sum + bias0)) * descale;
    if (activation >= 0)
    {
        if (desum < 0 && activation != 1)
        {
            desum = 0;
        }
        if (desum > 1 && activation == 1)
        {
            desum = 1;
        }
        if (desum > 6 && activation == 6)
        {
            desum = 6;
        }
        if (desum < -1 && activation == 1)
        {
            desum = -1;
        }
    }
    float outsum = desum * output_scales;
    return ( int8_t )float2int8(outsum);
}

static inline void pad_input(const int8_t* input, int8_t* inp_padded, int inc, int inh, int inw, int padded_h,
                             int padded_w, int pad0, int pad1)
{
    int padded_hw = padded_h * padded_w;
    int8_t* pad_ptr;
    int8_t* inp_ptr = ( int8_t* )input;
    int resi_h = padded_h - pad0 - inh;
    int resi_w = padded_w - pad1 - inw;
    for (int c = 0; c < inc; c++)
    {
        pad_ptr = inp_padded + c * padded_hw;
        // pad h_top
        memset(pad_ptr, 0, padded_w * pad0);
        pad_ptr += pad0 * padded_w;
        // pad h_mid
        for (int h = 0; h < inh; h++)
        {
            // pad w_left
            memset(pad_ptr, 0, pad1);
            // pad w_mid
            memcpy(pad_ptr + pad1, inp_ptr, inw);
            // pad w_end
            memset(pad_ptr + pad1 + inw, 0, resi_w);

            inp_ptr += inw;
            pad_ptr += padded_w;
        }
        // pad h_bottom
        memset(pad_ptr, 0, padded_w * resi_h);
    }
}

static inline void conv_dw_int8_3x3s1(const int8_t* input, int8_t* kernel, const int32_t* bias, int8_t* output,
                                      const float* dequant_scales, float output_scales, int c, int h, int w, int outc,
                                      int outh, int outw, int activation)
{
    for (int p = 0; p < outc; ++p)
    {
        const int32_t bias0 = bias ? bias[p] : 0.f;
        const float descale = dequant_scales[p];
        int8_t* outptr0 = output + p * outh * outw;
        int8_t* outptr0n = outptr0 + outw;
        const int8_t* kernel_ptr = kernel + 9 * p;

        const int8_t* r0 = input + p * h * w;
        const int8_t* r1 = input + p * h * w + w;
        const int8_t* r2 = input + p * h * w + w * 2;
        const int8_t* r3 = input + p * h * w + w * 3;

        int i = 0;
        int8x16_t _k0123456789x = vld1q_s8(kernel_ptr);
        int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
        int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

        int16x4_t _k0123 = vget_low_s16(_k_s16);
        int16x4_t _k4567 = vget_high_s16(_k_s16);
        int16x4_t _k8xxx = vget_low_s16(_kn_s16);
        int8x8_t input_0_7, input_8_15, input_1_8, input_2_9;
        int16x8_t input_0_7_16, input_1_8_16, input_2_9_16;
        int32x4_t sum0, sum0_1, sum1, sum1_1, sum2, sum2_1, sum3, sum3_1;

        int32x4_t qbias = vdupq_n_s32(bias0);
        float32x4_t qdescale = vdupq_n_f32(descale);
        float32x4_t outscale = vdupq_n_f32(output_scales);
        float32x4_t f_0_5 = vdupq_n_f32(0.5f);
        int8x8_t d127 = vdup_n_s8(127);
        int8x8_t d_127 = vdup_n_s8(-127);
        float32x4_t f0 = vdupq_n_f32(0);
        float32x4_t f1 = vdupq_n_f32(1);
        float32x4_t f6 = vdupq_n_f32(6);
        float32x4_t f_1 = vdupq_n_f32(-1);
        uint32x4_t u_1 = vmovq_n_u32(1);

        for (; i + 1 < outh; i += 2)
        {
            int nn = outw >> 3;
            int remain = outw & 7;
            for (int j = 0; j < nn; ++j)
            {
                // line0  outline0
                input_0_7 = vld1_s8(r0);
                input_8_15 = vld1_s8(r0 + 8);
                input_1_8 = vext_s8(input_0_7, input_8_15, 1);
                input_2_9 = vext_s8(input_0_7, input_8_15, 2);
                input_0_7_16 = vmovl_s8(input_0_7);
                input_1_8_16 = vmovl_s8(input_1_8);
                input_2_9_16 = vmovl_s8(input_2_9);
                sum0 = vmull_lane_s16(vget_low_s16(input_0_7_16), _k0123, 0);
                sum0_1 = vmull_lane_s16(vget_high_s16(input_0_7_16), _k0123, 0);
                sum1 = vmull_lane_s16(vget_low_s16(input_1_8_16), _k0123, 1);
                sum1_1 = vmull_lane_s16(vget_high_s16(input_1_8_16), _k0123, 1);
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_2_9_16), vget_lane_s16(_k0123, 2));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_2_9_16), vget_lane_s16(_k0123, 2));

                // line1 outline0
                input_0_7 = vld1_s8(r1);
                input_8_15 = vld1_s8(r1 + 8);
                input_1_8 = vext_s8(input_0_7, input_8_15, 1);
                input_2_9 = vext_s8(input_0_7, input_8_15, 2);
                input_0_7_16 = vmovl_s8(input_0_7);
                input_1_8_16 = vmovl_s8(input_1_8);
                input_2_9_16 = vmovl_s8(input_2_9);
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_0_7_16), vget_lane_s16(_k0123, 3));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_0_7_16), vget_lane_s16(_k0123, 3));
                sum1 = vmlal_n_s16(sum1, vget_low_s16(input_1_8_16), vget_lane_s16(_k4567, 0));
                sum1_1 = vmlal_n_s16(sum1_1, vget_high_s16(input_1_8_16), vget_lane_s16(_k4567, 0));
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_2_9_16), vget_lane_s16(_k4567, 1));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_2_9_16), vget_lane_s16(_k4567, 1));

                // line0 outline1
                sum2 = vmull_lane_s16(vget_low_s16(input_0_7_16), _k0123, 0);
                sum2_1 = vmull_lane_s16(vget_high_s16(input_0_7_16), _k0123, 0);
                sum3 = vmull_lane_s16(vget_low_s16(input_1_8_16), _k0123, 1);
                sum3_1 = vmull_lane_s16(vget_high_s16(input_1_8_16), _k0123, 1);
                sum2 = vmlal_n_s16(sum2, vget_low_s16(input_2_9_16), vget_lane_s16(_k0123, 2));
                sum2_1 = vmlal_n_s16(sum2_1, vget_high_s16(input_2_9_16), vget_lane_s16(_k0123, 2));

                // line2 outline0
                input_0_7 = vld1_s8(r2);
                input_8_15 = vld1_s8(r2 + 8);
                input_1_8 = vext_s8(input_0_7, input_8_15, 1);
                input_2_9 = vext_s8(input_0_7, input_8_15, 2);
                input_0_7_16 = vmovl_s8(input_0_7);
                input_1_8_16 = vmovl_s8(input_1_8);
                input_2_9_16 = vmovl_s8(input_2_9);
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_0_7_16), vget_lane_s16(_k4567, 2));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_0_7_16), vget_lane_s16(_k4567, 2));
                sum1 = vmlal_n_s16(sum1, vget_low_s16(input_1_8_16), vget_lane_s16(_k4567, 3));
                sum1_1 = vmlal_n_s16(sum1_1, vget_high_s16(input_1_8_16), vget_lane_s16(_k4567, 3));
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_2_9_16), vget_lane_s16(_k8xxx, 0));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_2_9_16), vget_lane_s16(_k8xxx, 0));

                // line1 outline1
                sum2 = vmlal_n_s16(sum2, vget_low_s16(input_0_7_16), vget_lane_s16(_k0123, 3));
                sum2_1 = vmlal_n_s16(sum2_1, vget_high_s16(input_0_7_16), vget_lane_s16(_k0123, 3));
                sum3 = vmlal_n_s16(sum3, vget_low_s16(input_1_8_16), vget_lane_s16(_k4567, 0));
                sum3_1 = vmlal_n_s16(sum3_1, vget_high_s16(input_1_8_16), vget_lane_s16(_k4567, 0));
                sum2 = vmlal_n_s16(sum2, vget_low_s16(input_2_9_16), vget_lane_s16(_k4567, 1));
                sum2_1 = vmlal_n_s16(sum2_1, vget_high_s16(input_2_9_16), vget_lane_s16(_k4567, 1));

                // line2 outline1
                input_0_7 = vld1_s8(r3);
                input_8_15 = vld1_s8(r3 + 8);
                input_1_8 = vext_s8(input_0_7, input_8_15, 1);
                input_2_9 = vext_s8(input_0_7, input_8_15, 2);
                input_0_7_16 = vmovl_s8(input_0_7);
                input_1_8_16 = vmovl_s8(input_1_8);
                input_2_9_16 = vmovl_s8(input_2_9);
                sum2 = vmlal_n_s16(sum2, vget_low_s16(input_0_7_16), vget_lane_s16(_k4567, 2));
                sum2_1 = vmlal_n_s16(sum2_1, vget_high_s16(input_0_7_16), vget_lane_s16(_k4567, 2));
                sum3 = vmlal_n_s16(sum3, vget_low_s16(input_1_8_16), vget_lane_s16(_k4567, 3));
                sum3_1 = vmlal_n_s16(sum3_1, vget_high_s16(input_1_8_16), vget_lane_s16(_k4567, 3));
                sum2 = vmlal_n_s16(sum2, vget_low_s16(input_2_9_16), vget_lane_s16(_k8xxx, 0));
                sum2_1 = vmlal_n_s16(sum2_1, vget_high_s16(input_2_9_16), vget_lane_s16(_k8xxx, 0));

                sum0 = vaddq_s32(sum0, sum1);
                sum0_1 = vaddq_s32(sum0_1, sum1_1);
                sum2 = vaddq_s32(sum2, sum3);
                sum2_1 = vaddq_s32(sum2_1, sum3_1);

                sum0 = vaddq_s32(sum0, qbias);
                sum0_1 = vaddq_s32(sum0_1, qbias);
                sum2 = vaddq_s32(sum2, qbias);
                sum2_1 = vaddq_s32(sum2_1, qbias);

                float32x4_t sum0_f = vcvtq_f32_s32(sum0);
                float32x4_t sum0_1_f = vcvtq_f32_s32(sum0_1);

                sum0_f = vmulq_f32(sum0_f, qdescale);
                sum0_1_f = vmulq_f32(sum0_1_f, qdescale);
                if (activation >= 0)
                {
                    if (activation != 1)
                    {
                        sum0_f = vmaxq_f32(sum0_f, f0);
                        sum0_1_f = vmaxq_f32(sum0_1_f, f0);
                    }
                    if (activation == 1)
                    {
                        sum0_f = vminq_f32(sum0_f, f1);
                        sum0_1_f = vminq_f32(sum0_1_f, f1);
                        sum0_f = vmaxq_f32(sum0_f, f_1);
                        sum0_1_f = vmaxq_f32(sum0_1_f, f_1);
                    }
                    if (activation == 6)
                    {
                        sum0_f = vminq_f32(sum0_f, f6);
                        sum0_1_f = vminq_f32(sum0_1_f, f6);
                    }
                }

                sum0_f = vmulq_f32(sum0_f, outscale);
                sum0_1_f = vmulq_f32(sum0_1_f, outscale);

                sum0_f = vaddq_f32(sum0_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f0,sum0_f),u_1)),0.5));
                sum0_1_f = vaddq_f32(sum0_1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f0,sum0_1_f),u_1)),0.5));
                sum0_f = vaddq_f32(sum0_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f0,sum0_f),u_1)),-0.5));
                sum0_1_f = vaddq_f32(sum0_1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f0,sum0_1_f),u_1)),-0.5));
                //sum0_f = vaddq_f32(sum0_f, f_0_5);
                //sum0_1_f = vaddq_f32(sum0_1_f, f_0_5);

                float32x4_t sum2_f = vcvtq_f32_s32(sum2);
                float32x4_t sum2_1_f = vcvtq_f32_s32(sum2_1);

                sum0 = vcvtq_s32_f32(sum0_f);
                sum0_1 = vcvtq_s32_f32(sum0_1_f);
                int16x4_t out0_16 = vqmovn_s32(sum0);
                int16x4_t out0_1_16 = vqmovn_s32(sum0_1);
                int8x8_t out = vqmovn_s16(vcombine_s16(out0_16, out0_1_16));
                out = vmax_s8(out, d_127);
                out = vmin_s8(out, d127);
                vst1_s8(outptr0, out);

                sum2_f = vmulq_f32(sum2_f, qdescale);
                sum2_1_f = vmulq_f32(sum2_1_f, qdescale);
                if (activation >= 0)
                {
                    if (activation != 1)
                    {
                        sum2_f = vmaxq_f32(sum2_f, f0);
                        sum2_1_f = vmaxq_f32(sum2_1_f, f0);
                    }
                    if (activation == 1)
                    {
                        sum2_f = vminq_f32(sum2_f, f1);
                        sum2_1_f = vminq_f32(sum2_1_f, f1);
                        sum2_f = vmaxq_f32(sum2_f, f_1);
                        sum2_1_f = vmaxq_f32(sum2_1_f, f_1);
                    }
                    if (activation == 6)
                    {
                        sum2_f = vminq_f32(sum2_f, f6);
                        sum2_1_f = vminq_f32(sum2_1_f, f6);
                    }
                }
                sum2_f = vmulq_f32(sum2_f, outscale);
                sum2_1_f = vmulq_f32(sum2_1_f, outscale);
                /* round */
                sum2_f = vaddq_f32(sum2_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f0,sum2_f),u_1)),0.5));
                sum2_1_f = vaddq_f32(sum2_1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f0,sum2_1_f),u_1)),0.5));
                sum2_f = vaddq_f32(sum2_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f0,sum2_f),u_1)),-0.5));
                sum2_1_f = vaddq_f32(sum2_1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f0,sum2_1_f),u_1)),-0.5));

                sum2 = vcvtq_s32_f32(sum2_f);
                sum2_1 = vcvtq_s32_f32(sum2_1_f);
                out0_16 = vqmovn_s32(sum2);
                out0_1_16 = vqmovn_s32(sum2_1);
                out = vqmovn_s16(vcombine_s16(out0_16, out0_1_16));
                out = vmax_s8(out, d_127);
                out = vmin_s8(out, d127);
                vst1_s8(outptr0n, out);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                outptr0 += 8;
                outptr0n += 8;
            }

            for (; remain > 0; remain--)
            {
                int sum = 0;
                int sumn = 0;

                sum += ( int )r0[0] * kernel_ptr[0];
                sum += ( int )r0[1] * kernel_ptr[1];
                sum += ( int )r0[2] * kernel_ptr[2];
                sum += ( int )r1[0] * kernel_ptr[3];
                sum += ( int )r1[1] * kernel_ptr[4];
                sum += ( int )r1[2] * kernel_ptr[5];
                sum += ( int )r2[0] * kernel_ptr[6];
                sum += ( int )r2[1] * kernel_ptr[7];
                sum += ( int )r2[2] * kernel_ptr[8];

                sumn += ( int )r1[0] * kernel_ptr[0];
                sumn += ( int )r1[1] * kernel_ptr[1];
                sumn += ( int )r1[2] * kernel_ptr[2];
                sumn += ( int )r2[0] * kernel_ptr[3];
                sumn += ( int )r2[1] * kernel_ptr[4];
                sumn += ( int )r2[2] * kernel_ptr[5];
                sumn += ( int )r3[0] * kernel_ptr[6];
                sumn += ( int )r3[1] * kernel_ptr[7];
                sumn += ( int )r3[2] * kernel_ptr[8];

                *outptr0 = sum2int8(sum, descale, bias0, output_scales, activation);
                *outptr0n = sum2int8(sumn, descale, bias0, output_scales, activation);

                r0++;
                r1++;
                r2++;
                r3++;
                outptr0++;
                outptr0n++;
            }

            r0 += 2 + w;
            r1 += 2 + w;
            r2 += 2 + w;
            r3 += 2 + w;

            outptr0 += outw;
            outptr0n += outw;
        }

        for (; i < outh; ++i)
        {
            int nn = outw >> 3;
            int remain = outw & 7;
            for (int j = 0; j < nn; ++j)
            {
                // line0  outline0
                input_0_7 = vld1_s8(r0);
                input_8_15 = vld1_s8(r0 + 8);
                input_1_8 = vext_s8(input_0_7, input_8_15, 1);
                input_2_9 = vext_s8(input_0_7, input_8_15, 2);
                input_0_7_16 = vmovl_s8(input_0_7);
                input_1_8_16 = vmovl_s8(input_1_8);
                input_2_9_16 = vmovl_s8(input_2_9);
                sum0 = vmull_lane_s16(vget_low_s16(input_0_7_16), _k0123, 0);
                sum0_1 = vmull_lane_s16(vget_high_s16(input_0_7_16), _k0123, 0);
                sum1 = vmull_lane_s16(vget_low_s16(input_1_8_16), _k0123, 1);
                sum1_1 = vmull_lane_s16(vget_high_s16(input_1_8_16), _k0123, 1);
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_2_9_16), vget_lane_s16(_k0123, 2));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_2_9_16), vget_lane_s16(_k0123, 2));

                // line1 outline0
                input_0_7 = vld1_s8(r1);
                input_8_15 = vld1_s8(r1 + 8);
                input_1_8 = vext_s8(input_0_7, input_8_15, 1);
                input_2_9 = vext_s8(input_0_7, input_8_15, 2);
                input_0_7_16 = vmovl_s8(input_0_7);
                input_1_8_16 = vmovl_s8(input_1_8);
                input_2_9_16 = vmovl_s8(input_2_9);
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_0_7_16), vget_lane_s16(_k0123, 3));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_0_7_16), vget_lane_s16(_k0123, 3));
                sum1 = vmlal_n_s16(sum1, vget_low_s16(input_1_8_16), vget_lane_s16(_k4567, 0));
                sum1_1 = vmlal_n_s16(sum1_1, vget_high_s16(input_1_8_16), vget_lane_s16(_k4567, 0));
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_2_9_16), vget_lane_s16(_k4567, 1));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_2_9_16), vget_lane_s16(_k4567, 1));

                // line2 outline0
                input_0_7 = vld1_s8(r2);
                input_8_15 = vld1_s8(r2 + 8);
                input_1_8 = vext_s8(input_0_7, input_8_15, 1);
                input_2_9 = vext_s8(input_0_7, input_8_15, 2);
                input_0_7_16 = vmovl_s8(input_0_7);
                input_1_8_16 = vmovl_s8(input_1_8);
                input_2_9_16 = vmovl_s8(input_2_9);
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_0_7_16), vget_lane_s16(_k4567, 2));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_0_7_16), vget_lane_s16(_k4567, 2));
                sum1 = vmlal_n_s16(sum1, vget_low_s16(input_1_8_16), vget_lane_s16(_k4567, 3));
                sum1_1 = vmlal_n_s16(sum1_1, vget_high_s16(input_1_8_16), vget_lane_s16(_k4567, 3));
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input_2_9_16), vget_lane_s16(_k8xxx, 0));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input_2_9_16), vget_lane_s16(_k8xxx, 0));

                sum0 = vaddq_s32(sum0, sum1);
                sum0_1 = vaddq_s32(sum0_1, sum1_1);

                sum0 = vaddq_s32(sum0, qbias);
                sum0_1 = vaddq_s32(sum0_1, qbias);

                float32x4_t sum0_f = vcvtq_f32_s32(sum0);
                float32x4_t sum0_1_f = vcvtq_f32_s32(sum0_1);

                sum0_f = vmulq_f32(sum0_f, qdescale);
                sum0_1_f = vmulq_f32(sum0_1_f, qdescale);
                if (activation >= 0)
                {
                    if (activation != 1)
                    {
                        sum0_f = vmaxq_f32(sum0_f, f0);
                        sum0_1_f = vmaxq_f32(sum0_1_f, f0);
                    }
                    if (activation == 1)
                    {
                        sum0_f = vminq_f32(sum0_f, f1);
                        sum0_1_f = vminq_f32(sum0_1_f, f1);
                        sum0_f = vmaxq_f32(sum0_f, f_1);
                        sum0_1_f = vmaxq_f32(sum0_1_f, f_1);
                    }
                    if (activation == 6)
                    {
                        sum0_f = vminq_f32(sum0_f, f6);
                        sum0_1_f = vminq_f32(sum0_1_f, f6);
                    }
                }
                /* round */
                sum0_f = vmulq_f32(sum0_f, outscale);
                sum0_1_f = vmulq_f32(sum0_1_f, outscale);
                sum0_f = vaddq_f32(sum0_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f0,sum0_f),u_1)),0.5));
                sum0_1_f = vaddq_f32(sum0_1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f0,sum0_1_f),u_1)),0.5));
                sum0_f = vaddq_f32(sum0_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f0,sum0_f),u_1)),-0.5));
                sum0_1_f = vaddq_f32(sum0_1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f0,sum0_1_f),u_1)),-0.5));

                sum0 = vcvtq_s32_f32(sum0_f);
                sum0_1 = vcvtq_s32_f32(sum0_1_f);
                int16x4_t out0_16 = vqmovn_s32(sum0);
                int16x4_t out0_1_16 = vqmovn_s32(sum0_1);
                int8x8_t out = vqmovn_s16(vcombine_s16(out0_16, out0_1_16));
                out = vmax_s8(out, d_127);
                out = vmin_s8(out, d127);
                vst1_s8(outptr0, out);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr0 += 8;
            }
            for (; remain > 0; remain--)
            {
                int sum = 0;
                sum += ( int )r0[0] * kernel_ptr[0];
                sum += ( int )r0[1] * kernel_ptr[1];
                sum += ( int )r0[2] * kernel_ptr[2];
                sum += ( int )r1[0] * kernel_ptr[3];
                sum += ( int )r1[1] * kernel_ptr[4];
                sum += ( int )r1[2] * kernel_ptr[5];
                sum += ( int )r2[0] * kernel_ptr[6];
                sum += ( int )r2[1] * kernel_ptr[7];
                sum += ( int )r2[2] * kernel_ptr[8];
                *outptr0 = sum2int8(sum, descale, bias0, output_scales, activation);
                r0++;
                r1++;
                r2++;
                outptr0++;
            }
            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
}

static inline void conv_dw_int8_3x3s2(const int8_t* input, const int8_t* kernel, const int32_t* bias, int8_t* output,
                                      const float* dequant_scales, float output_scales, int c, int h, int w, int outc,
                                      int outh, int outw, int activation)
{
    const int tailstep = w - 2 * outw + w;
    for (int p = 0; p < outc; ++p)
    {
        const int32_t bias0 = bias ? bias[p] : 0.f;
        const float descale = dequant_scales[p];
        int8_t* outptr0 = output + p * outh * outw;
        const int8_t* kernel_ptr = kernel + 9 * p;
        const int8_t* r0 = input + p * h * w;
        const int8_t* r1 = input + p * h * w + w;
        const int8_t* r2 = input + p * h * w + w * 2;

        int i = 0;
        int8x16_t _k0123456789x = vld1q_s8(kernel_ptr);
        int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
        int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

        int16x4_t _k0123 = vget_low_s16(_k_s16);
        int16x4_t _k4567 = vget_high_s16(_k_s16);
        int16x4_t _k8xxx = vget_low_s16(_kn_s16);

        int8x8x2_t input0_15, input16_31;
        int8x8_t input0_2_14, input1_2_15, input2_2_16;
        int16x8_t input0_2_14_16, input1_2_15_16, input2_2_16_16;
        int32x4_t sum0, sum0_1, sum1, sum1_1;
        int32x4_t qbias = vdupq_n_s32(bias0);
        float32x4_t qdescale = vdupq_n_f32(descale);
        float32x4_t outscale = vdupq_n_f32(output_scales);
        float32x4_t f_0_5 = vdupq_n_f32(0.5f);
        int8x8_t d127 = vdup_n_s8(127);
        int8x8_t d_127 = vdup_n_s8(-127);
        float32x4_t f0 = vdupq_n_f32(0);
        float32x4_t f1 = vdupq_n_f32(1);
        float32x4_t f6 = vdupq_n_f32(6);
        float32x4_t f_1 = vdupq_n_f32(-1);
        uint32x4_t u_1 = vmovq_n_u32(1);

        for (; i < outh; ++i)
        {
            int nn = outw >> 3;
            int remain = outw & 7;
            for (int j = 0; j < nn; ++j)
            {
                input0_15 = vld2_s8(r0);
                input16_31 = vld2_s8(r0 + 16);
                input0_2_14 = input0_15.val[0];
                input1_2_15 = input0_15.val[1];
                input2_2_16 = vext_s8(input0_2_14, input16_31.val[0], 1);
                input0_2_14_16 = vmovl_s8(input0_2_14);
                input1_2_15_16 = vmovl_s8(input1_2_15);
                input2_2_16_16 = vmovl_s8(input2_2_16);
                sum0 = vmull_lane_s16(vget_low_s16(input0_2_14_16), _k0123, 0);
                sum0_1 = vmull_lane_s16(vget_high_s16(input0_2_14_16), _k0123, 0);
                sum1 = vmull_lane_s16(vget_low_s16(input1_2_15_16), _k0123, 1);
                sum1_1 = vmull_lane_s16(vget_high_s16(input1_2_15_16), _k0123, 1);
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input2_2_16_16), vget_lane_s16(_k0123, 2));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input2_2_16_16), vget_lane_s16(_k0123, 2));

                input0_15 = vld2_s8(r1);
                input16_31 = vld2_s8(r1 + 16);
                input0_2_14 = input0_15.val[0];
                input1_2_15 = input0_15.val[1];
                input2_2_16 = vext_s8(input0_2_14, input16_31.val[0], 1);
                input0_2_14_16 = vmovl_s8(input0_2_14);
                input1_2_15_16 = vmovl_s8(input1_2_15);
                input2_2_16_16 = vmovl_s8(input2_2_16);
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input0_2_14_16), vget_lane_s16(_k0123, 3));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input0_2_14_16), vget_lane_s16(_k0123, 3));
                sum1 = vmlal_n_s16(sum1, vget_low_s16(input1_2_15_16), vget_lane_s16(_k4567, 0));
                sum1_1 = vmlal_n_s16(sum1_1, vget_high_s16(input1_2_15_16), vget_lane_s16(_k4567, 0));
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input2_2_16_16), vget_lane_s16(_k4567, 1));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input2_2_16_16), vget_lane_s16(_k4567, 1));

                input0_15 = vld2_s8(r2);
                input16_31 = vld2_s8(r2 + 16);
                input0_2_14 = input0_15.val[0];
                input1_2_15 = input0_15.val[1];
                input2_2_16 = vext_s8(input0_2_14, input16_31.val[0], 1);
                input0_2_14_16 = vmovl_s8(input0_2_14);
                input1_2_15_16 = vmovl_s8(input1_2_15);
                input2_2_16_16 = vmovl_s8(input2_2_16);
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input0_2_14_16), vget_lane_s16(_k4567, 2));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input0_2_14_16), vget_lane_s16(_k4567, 2));
                sum1 = vmlal_n_s16(sum1, vget_low_s16(input1_2_15_16), vget_lane_s16(_k4567, 3));
                sum1_1 = vmlal_n_s16(sum1_1, vget_high_s16(input1_2_15_16), vget_lane_s16(_k4567, 3));
                sum0 = vmlal_n_s16(sum0, vget_low_s16(input2_2_16_16), vget_lane_s16(_k8xxx, 0));
                sum0_1 = vmlal_n_s16(sum0_1, vget_high_s16(input2_2_16_16), vget_lane_s16(_k8xxx, 0));

                sum0 = vaddq_s32(sum0, sum1);
                sum0_1 = vaddq_s32(sum0_1, sum1_1);
                sum0 = vaddq_s32(sum0, qbias);
                sum0_1 = vaddq_s32(sum0_1, qbias);

                float32x4_t sum0_f = vcvtq_f32_s32(sum0);
                float32x4_t sum0_1_f = vcvtq_f32_s32(sum0_1);

                sum0_f = vmulq_f32(sum0_f, qdescale);
                sum0_1_f = vmulq_f32(sum0_1_f, qdescale);

                if (activation >= 0)
                {
                    if (activation != 1)
                    {
                        sum0_f = vmaxq_f32(sum0_f, f0);
                        sum0_1_f = vmaxq_f32(sum0_1_f, f0);
                    }
                    if (activation == 1)
                    {
                        sum0_f = vminq_f32(sum0_f, f1);
                        sum0_1_f = vminq_f32(sum0_1_f, f1);
                        sum0_f = vmaxq_f32(sum0_f, f_1);
                        sum0_1_f = vmaxq_f32(sum0_1_f, f_1);
                    }
                    if (activation == 6)
                    {
                        sum0_f = vminq_f32(sum0_f, f6);
                        sum0_1_f = vminq_f32(sum0_1_f, f6);
                    }
                }

                sum0_f = vmulq_f32(sum0_f, outscale);
                sum0_1_f = vmulq_f32(sum0_1_f, outscale);
                /* round */
                sum0_f = vaddq_f32(sum0_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f0,sum0_f),u_1)),0.5));
                sum0_1_f = vaddq_f32(sum0_1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcgtq_f32(f0,sum0_1_f),u_1)),0.5));
                sum0_f = vaddq_f32(sum0_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f0,sum0_f),u_1)),-0.5));
                sum0_1_f = vaddq_f32(sum0_1_f, vmulq_n_f32(vcvtq_f32_u32(vaddq_u32(vcltq_f32(f0,sum0_1_f),u_1)),-0.5));

                sum0 = vcvtq_s32_f32(sum0_f);
                sum0_1 = vcvtq_s32_f32(sum0_1_f);
                int16x4_t out0_16 = vqmovn_s32(sum0);
                int16x4_t out0_1_16 = vqmovn_s32(sum0_1);
                int8x8_t out = vqmovn_s16(vcombine_s16(out0_16, out0_1_16));
                out = vmax_s8(out, d_127);
                out = vmin_s8(out, d127);
                vst1_s8(outptr0, out);
                r0 += 16;
                r1 += 16;
                r2 += 16;
                outptr0 += 8;
            }

            for (; remain > 0; remain--)
            {
                int sum = 0;

                sum += ( int )r0[0] * kernel_ptr[0];
                sum += ( int )r0[1] * kernel_ptr[1];
                sum += ( int )r0[2] * kernel_ptr[2];
                sum += ( int )r1[0] * kernel_ptr[3];
                sum += ( int )r1[1] * kernel_ptr[4];
                sum += ( int )r1[2] * kernel_ptr[5];
                sum += ( int )r2[0] * kernel_ptr[6];
                sum += ( int )r2[1] * kernel_ptr[7];
                sum += ( int )r2[2] * kernel_ptr[8];

                *outptr0 = sum2int8(sum, descale, bias0, output_scales, activation);
                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr0++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

int conv_dw_int8_run(struct ir_tensor* input_tensor, struct ir_tensor* filter_tensor, struct ir_tensor* bias_tensor,
                 struct ir_tensor* output_tensor, struct conv_priv_info* priv_info, struct conv_param* param,
                 int num_thread, int cpu_affinity)
{
    int pads[4];
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    pads[0] = param->pad_h0;
    pads[1] = param->pad_w0;
    pads[2] = param->pad_h1;
    pads[3] = param->pad_w1;

    if (stride_h != stride_w)
        return -1;

    if (pads[0] != pads[1])
        return -1;

    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1] / group;
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int input_size = in_c * in_h * in_w;

    int out_c = output_tensor->dims[1] / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int output_size = out_c * out_h * out_w;

    int8_t* input_buf = ( int8_t* )input_tensor->data;
    int8_t* kernel_buf = ( int8_t* )filter_tensor->data;
    int8_t* output_buf = ( int8_t* )output_tensor->data;
    int32_t* biases_buf = NULL;
    if (bias_tensor)
        biases_buf = ( int32_t* )bias_tensor->data;

    float input_scale = input_tensor->scale;
    float* kernel_scales = filter_tensor->scale_list;
    float output_scale = 1 / output_tensor->scale;
    int activation = param->activation;

    int dequant_scales_size = group * out_c;
    float* dequant_scales = ( float* )malloc(sizeof(float) * dequant_scales_size);

    for (int i = 0; i < dequant_scales_size; i++)
    {
        dequant_scales[i] = (input_scale * kernel_scales[i]);
    }

    // pad
    int padded_h = in_h + pads[0] + pads[2];
    int padded_w = in_w + pads[1] + pads[3];
    int8_t* input_padd_buf = ( int8_t* )sys_malloc(padded_h * padded_w * in_c * group);

    pad_input(input_buf, input_padd_buf, in_c * group, in_h, in_w, padded_h, padded_w, pads[0], pads[1]);

    if (stride_h == 1)
    {
        conv_dw_int8_3x3s1(input_padd_buf, kernel_buf, biases_buf, output_buf, dequant_scales, output_scale,
                           in_c * group, padded_h, padded_w, group, out_h, out_w, activation);
    }
    else if (stride_h == 2)
    {
        conv_dw_int8_3x3s2(input_padd_buf, kernel_buf, biases_buf, output_buf, dequant_scales, output_scale,
                           in_c * group, padded_h, padded_w, group, out_h, out_w, activation);
    }
    sys_free(input_padd_buf);
    return 0;
}
