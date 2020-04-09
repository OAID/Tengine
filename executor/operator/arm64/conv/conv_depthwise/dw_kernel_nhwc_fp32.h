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
 * Copyright (c) 2017, Open AI Lab
 * Author: rzhuang@openailab.com
 */


#include "arm_neon.h"


//for kernel k3s1p1 start
static void k3s1p1_nhwc_fp32_hstc(float* input, float* kernel, float* out, float* bias, long act, long in_w, long in_h, long in_c, long out_w, long out_h)
{
    float* k4_ptr = kernel + 4 * in_c;
    float* k5_ptr = kernel + 5 * in_c;
    float* k7_ptr = k4_ptr + 3 * in_c;
    float* k8_ptr = k4_ptr + 4 * in_c;

    float* l01_ptr = input;
    float* l02_ptr = input + in_c;
    float* l11_ptr = input + in_w * in_c;
    float* l12_ptr = input + in_w * in_c + in_c;
    
    float* b_ptr = bias;

    int lc = 0;

    float32x4_t k0_c0, k0_c1, k0_c2, k0_c3;
    float32x4_t l0_c0, l0_c1, l0_c2, l0_c3;
    float32x4_t b0_c0, b0_c1, b0_c2, b0_c3;
    float32x4_t r0_c0, r0_c1, r0_c2, r0_c3;

    float32x4_t v_actz = vdupq_n_f32(0);
    float32x4_t v_actx = vdupq_n_f32(act);

    r0_c0 = vdupq_n_f32(0);
    r0_c1 = vdupq_n_f32(0);
    r0_c2 = vdupq_n_f32(0);
    r0_c3 = vdupq_n_f32(0);

//top left point
    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k5_ptr);
        k0_c1 = vld1q_f32(k5_ptr+4);
        k0_c2 = vld1q_f32(k5_ptr+8);
        k0_c3 = vld1q_f32(k5_ptr+12);
        k5_ptr += 16;
        
        l0_c0 = vld1q_f32(l02_ptr);
        l0_c1 = vld1q_f32(l02_ptr+4);
        l0_c2 = vld1q_f32(l02_ptr+8);
        l0_c3 = vld1q_f32(l02_ptr+12);
        l02_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k7_ptr);
        k0_c1 = vld1q_f32(k7_ptr+4);
        k0_c2 = vld1q_f32(k7_ptr+8);
        k0_c3 = vld1q_f32(k7_ptr+12);
        k7_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k8_ptr);
        k0_c1 = vld1q_f32(k8_ptr+4);
        k0_c2 = vld1q_f32(k8_ptr+8);
        k0_c3 = vld1q_f32(k8_ptr+12);
        k8_ptr += 16;

        l0_c0 = vld1q_f32(l12_ptr);
        l0_c1 = vld1q_f32(l12_ptr+4);
        l0_c2 = vld1q_f32(l12_ptr+8);
        l0_c3 = vld1q_f32(l12_ptr+12);
        l12_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k4_p = *k4_ptr++;
        float l01_p = *l01_ptr++;
        float k5_p = *k5_ptr++;
        float l02_p = *l02_ptr++;
        float k7_p = *k7_ptr++;
        float l11_p = *l11_ptr++;
        float k8_p = *k8_ptr++;
        float l12_p = *l12_ptr++;
        float r0_p = (k4_p * l01_p) + (k5_p * l02_p) + (k7_p * l11_p) + (k8_p * l12_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }

//top middle points
    float* k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    k5_ptr = kernel + 5 * in_c;
    float* k6_ptr = k4_ptr + 2 * in_c;
    k7_ptr = k4_ptr + 3 * in_c;
    k8_ptr = k4_ptr + 4 * in_c;

    float* l00_ptr = input;
    l01_ptr = input + in_c;
    l02_ptr = input + 2 * in_c;
    float* l10_ptr = input + in_w * in_c;
    l11_ptr = input + in_w * in_c + in_c;
    l12_ptr = input + in_w * in_c + 2 * in_c;
   
    b_ptr = bias;

    for(int lw=0; lw<(out_w-2); lw++)
    {
        lc = 0;

        for(lc=0; (lc+16)<=in_c; lc+=16)
        {
            k0_c0 = vld1q_f32(k3_ptr);
            k0_c1 = vld1q_f32(k3_ptr+4);
            k0_c2 = vld1q_f32(k3_ptr+8);
            k0_c3 = vld1q_f32(k3_ptr+12);
            k3_ptr += 16;
            
            l0_c0 = vld1q_f32(l00_ptr);
            l0_c1 = vld1q_f32(l00_ptr+4);
            l0_c2 = vld1q_f32(l00_ptr+8);
            l0_c3 = vld1q_f32(l00_ptr+12);
            l00_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k4_ptr);
            k0_c1 = vld1q_f32(k4_ptr+4);
            k0_c2 = vld1q_f32(k4_ptr+8);
            k0_c3 = vld1q_f32(k4_ptr+12);
            k4_ptr += 16;
            
            l0_c0 = vld1q_f32(l01_ptr);
            l0_c1 = vld1q_f32(l01_ptr+4);
            l0_c2 = vld1q_f32(l01_ptr+8);
            l0_c3 = vld1q_f32(l01_ptr+12);
            l01_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k5_ptr);
            k0_c1 = vld1q_f32(k5_ptr+4);
            k0_c2 = vld1q_f32(k5_ptr+8);
            k0_c3 = vld1q_f32(k5_ptr+12);
            k5_ptr += 16;
            
            l0_c0 = vld1q_f32(l02_ptr);
            l0_c1 = vld1q_f32(l02_ptr+4);
            l0_c2 = vld1q_f32(l02_ptr+8);
            l0_c3 = vld1q_f32(l02_ptr+12);
            l02_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k6_ptr);
            k0_c1 = vld1q_f32(k6_ptr+4);
            k0_c2 = vld1q_f32(k6_ptr+8);
            k0_c3 = vld1q_f32(k6_ptr+12);
            k6_ptr += 16;

            l0_c0 = vld1q_f32(l10_ptr);
            l0_c1 = vld1q_f32(l10_ptr+4);
            l0_c2 = vld1q_f32(l10_ptr+8);
            l0_c3 = vld1q_f32(l10_ptr+12);
            l10_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k7_ptr);
            k0_c1 = vld1q_f32(k7_ptr+4);
            k0_c2 = vld1q_f32(k7_ptr+8);
            k0_c3 = vld1q_f32(k7_ptr+12);
            k7_ptr += 16;

            l0_c0 = vld1q_f32(l11_ptr);
            l0_c1 = vld1q_f32(l11_ptr+4);
            l0_c2 = vld1q_f32(l11_ptr+8);
            l0_c3 = vld1q_f32(l11_ptr+12);
            l11_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k8_ptr);
            k0_c1 = vld1q_f32(k8_ptr+4);
            k0_c2 = vld1q_f32(k8_ptr+8);
            k0_c3 = vld1q_f32(k8_ptr+12);
            k8_ptr += 16;

            l0_c0 = vld1q_f32(l12_ptr);
            l0_c1 = vld1q_f32(l12_ptr+4);
            l0_c2 = vld1q_f32(l12_ptr+8);
            l0_c3 = vld1q_f32(l12_ptr+12);
            l12_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            if(b_ptr!=nullptr)
            {
                b0_c0 = vld1q_f32(b_ptr);
                b0_c1 = vld1q_f32(b_ptr+4);
                b0_c2 = vld1q_f32(b_ptr+8);
                b0_c3 = vld1q_f32(b_ptr+12);
                b_ptr += 16;

                r0_c0 = vaddq_f32(r0_c0, b0_c0);
                r0_c1 = vaddq_f32(r0_c1, b0_c1);
                r0_c2 = vaddq_f32(r0_c2, b0_c2);
                r0_c3 = vaddq_f32(r0_c3, b0_c3);
            }
            //
            if(act>=0)
            {
                r0_c0 = vmaxq_f32(r0_c0, v_actz);
                r0_c1 = vmaxq_f32(r0_c1, v_actz);
                r0_c2 = vmaxq_f32(r0_c2, v_actz);
                r0_c3 = vmaxq_f32(r0_c3, v_actz);
                if(act>0)
                {
                    r0_c0 = vminq_f32(r0_c0, v_actx);
                    r0_c1 = vminq_f32(r0_c1, v_actx);
                    r0_c2 = vminq_f32(r0_c2, v_actx);
                    r0_c3 = vminq_f32(r0_c3, v_actx);
                }
            }
            //
            vst1q_f32(out, r0_c0);
            vst1q_f32(out+4, r0_c1);
            vst1q_f32(out+8, r0_c2);
            vst1q_f32(out+12, r0_c3);
            out += 16;
            //
            r0_c0 = vdupq_n_f32(0);
            r0_c1 = vdupq_n_f32(0);
            r0_c2 = vdupq_n_f32(0);
            r0_c3 = vdupq_n_f32(0);
        }
        for( ; lc<in_c; lc++)
        {
            float k3_p = *k3_ptr++;
            float l00_p = *l00_ptr++;
            float k4_p = *k4_ptr++;
            float l01_p = *l01_ptr++;
            float k5_p = *k5_ptr++;
            float l02_p = *l02_ptr++;
            float k6_p = *k6_ptr++;
            float l10_p = *l10_ptr++;
            float k7_p = *k7_ptr++;
            float l11_p = *l11_ptr++;
            float k8_p = *k8_ptr++;
            float l12_p = *l12_ptr++;
            float r0_p = (k3_p * l00_p) + (k4_p * l01_p) + (k5_p * l02_p) + (k6_p * l10_p) + (k7_p * l11_p) + (k8_p * l12_p);
            if(b_ptr!=nullptr)
            {
                r0_p = r0_p + *b_ptr++;
            }
            if(act>=0)
            {
                r0_p = std::max(r0_p, 0.f);
                if(act>0)
                    r0_p = std::min(r0_p, (float)act);
            }
            *out++ = r0_p;
        }

        k3_ptr = kernel + 3 * in_c;
        k4_ptr = kernel + 4 * in_c;
        k5_ptr = kernel + 5 * in_c;
        k6_ptr = k4_ptr + 2 * in_c;
        k7_ptr = k4_ptr + 3 * in_c;
        k8_ptr = k4_ptr + 4 * in_c;
   
        b_ptr = bias;
    }

//top right point
    k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    k6_ptr = k4_ptr + 2 * in_c;
    k7_ptr = k4_ptr + 3 * in_c;
    
    b_ptr = bias;

    lc = 0;

    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k3_ptr);
        k0_c1 = vld1q_f32(k3_ptr+4);
        k0_c2 = vld1q_f32(k3_ptr+8);
        k0_c3 = vld1q_f32(k3_ptr+12);
        k3_ptr += 16;
        
        l0_c0 = vld1q_f32(l00_ptr);
        l0_c1 = vld1q_f32(l00_ptr+4);
        l0_c2 = vld1q_f32(l00_ptr+8);
        l0_c3 = vld1q_f32(l00_ptr+12);
        l00_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k6_ptr);
        k0_c1 = vld1q_f32(k6_ptr+4);
        k0_c2 = vld1q_f32(k6_ptr+8);
        k0_c3 = vld1q_f32(k6_ptr+12);
        k6_ptr += 16;

        l0_c0 = vld1q_f32(l10_ptr);
        l0_c1 = vld1q_f32(l10_ptr+4);
        l0_c2 = vld1q_f32(l10_ptr+8);
        l0_c3 = vld1q_f32(l10_ptr+12);
        l10_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k7_ptr);
        k0_c1 = vld1q_f32(k7_ptr+4);
        k0_c2 = vld1q_f32(k7_ptr+8);
        k0_c3 = vld1q_f32(k7_ptr+12);
        k7_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k3_p = *k3_ptr++;
        float l00_p = *l00_ptr++;
        float k4_p = *k4_ptr++;
        float l01_p = *l01_ptr++;
        float k6_p = *k6_ptr++;
        float l10_p = *l10_ptr++;
        float k7_p = *k7_ptr++;
        float l11_p = *l11_ptr++;
        float r0_p = (k3_p * l00_p) + (k4_p * l01_p) + (k6_p * l10_p) + (k7_p * l11_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }
}






static void k3s1p1_nhwc_fp32_hetc(float* input, float* kernel, float* out, float* bias, long act, long in_w, long in_h, long in_c, long out_w, long out_h)
{
    float* k1_ptr = kernel + in_c;
    float* k2_ptr = kernel + 2 * in_c;
    float* k4_ptr = k2_ptr + 2 * in_c;
    float* k5_ptr = k2_ptr + 3 * in_c;

    float* l01_ptr = input;
    float* l02_ptr = input + in_c;
    float* l11_ptr = input + in_w * in_c;
    float* l12_ptr = input + in_w * in_c + in_c;
    
    float* b_ptr = bias;

    int lc = 0;

    float32x4_t k0_c0, k0_c1, k0_c2, k0_c3;
    float32x4_t l0_c0, l0_c1, l0_c2, l0_c3;
    float32x4_t b0_c0, b0_c1, b0_c2, b0_c3;
    float32x4_t r0_c0, r0_c1, r0_c2, r0_c3;

    float32x4_t v_actz = vdupq_n_f32(0);
    float32x4_t v_actx = vdupq_n_f32(act);

    r0_c0 = vdupq_n_f32(0);
    r0_c1 = vdupq_n_f32(0);
    r0_c2 = vdupq_n_f32(0);
    r0_c3 = vdupq_n_f32(0);

//bottom left point
    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k2_ptr);
        k0_c1 = vld1q_f32(k2_ptr+4);
        k0_c2 = vld1q_f32(k2_ptr+8);
        k0_c3 = vld1q_f32(k2_ptr+12);
        k2_ptr += 16;
        
        l0_c0 = vld1q_f32(l02_ptr);
        l0_c1 = vld1q_f32(l02_ptr+4);
        l0_c2 = vld1q_f32(l02_ptr+8);
        l0_c3 = vld1q_f32(l02_ptr+12);
        l02_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k5_ptr);
        k0_c1 = vld1q_f32(k5_ptr+4);
        k0_c2 = vld1q_f32(k5_ptr+8);
        k0_c3 = vld1q_f32(k5_ptr+12);
        k5_ptr += 16;

        l0_c0 = vld1q_f32(l12_ptr);
        l0_c1 = vld1q_f32(l12_ptr+4);
        l0_c2 = vld1q_f32(l12_ptr+8);
        l0_c3 = vld1q_f32(l12_ptr+12);
        l12_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k2_p = *k2_ptr++;
        float l02_p = *l02_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float k5_p = *k5_ptr++;
        float l12_p = *l12_ptr++;
        float r0_p = (k1_p * l01_p) + (k2_p * l02_p) + (k4_p * l11_p) + (k5_p * l12_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }

//bottom middle points
    float* k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k2_ptr = kernel + 2 * in_c;
    float* k3_ptr = k2_ptr + in_c;
    k4_ptr = k2_ptr + 2 * in_c;
    k5_ptr = k2_ptr + 3 * in_c;

    float* l00_ptr = input;
    l01_ptr = input + in_c;
    l02_ptr = input + 2 * in_c;
    float* l10_ptr = input + in_w * in_c;
    l11_ptr = input + in_w * in_c + in_c;
    l12_ptr = input + in_w * in_c + 2 * in_c;
   
    b_ptr = bias;

    for(int lw=0; lw<(out_w-2); lw++)
    {
        lc = 0;

        for(lc=0; (lc+16)<=in_c; lc+=16)
        {
            k0_c0 = vld1q_f32(k0_ptr);
            k0_c1 = vld1q_f32(k0_ptr+4);
            k0_c2 = vld1q_f32(k0_ptr+8);
            k0_c3 = vld1q_f32(k0_ptr+12);
            k0_ptr += 16;
            
            l0_c0 = vld1q_f32(l00_ptr);
            l0_c1 = vld1q_f32(l00_ptr+4);
            l0_c2 = vld1q_f32(l00_ptr+8);
            l0_c3 = vld1q_f32(l00_ptr+12);
            l00_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k1_ptr);
            k0_c1 = vld1q_f32(k1_ptr+4);
            k0_c2 = vld1q_f32(k1_ptr+8);
            k0_c3 = vld1q_f32(k1_ptr+12);
            k1_ptr += 16;
            
            l0_c0 = vld1q_f32(l01_ptr);
            l0_c1 = vld1q_f32(l01_ptr+4);
            l0_c2 = vld1q_f32(l01_ptr+8);
            l0_c3 = vld1q_f32(l01_ptr+12);
            l01_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k2_ptr);
            k0_c1 = vld1q_f32(k2_ptr+4);
            k0_c2 = vld1q_f32(k2_ptr+8);
            k0_c3 = vld1q_f32(k2_ptr+12);
            k2_ptr += 16;
            
            l0_c0 = vld1q_f32(l02_ptr);
            l0_c1 = vld1q_f32(l02_ptr+4);
            l0_c2 = vld1q_f32(l02_ptr+8);
            l0_c3 = vld1q_f32(l02_ptr+12);
            l02_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k3_ptr);
            k0_c1 = vld1q_f32(k3_ptr+4);
            k0_c2 = vld1q_f32(k3_ptr+8);
            k0_c3 = vld1q_f32(k3_ptr+12);
            k3_ptr += 16;

            l0_c0 = vld1q_f32(l10_ptr);
            l0_c1 = vld1q_f32(l10_ptr+4);
            l0_c2 = vld1q_f32(l10_ptr+8);
            l0_c3 = vld1q_f32(l10_ptr+12);
            l10_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k4_ptr);
            k0_c1 = vld1q_f32(k4_ptr+4);
            k0_c2 = vld1q_f32(k4_ptr+8);
            k0_c3 = vld1q_f32(k4_ptr+12);
            k4_ptr += 16;

            l0_c0 = vld1q_f32(l11_ptr);
            l0_c1 = vld1q_f32(l11_ptr+4);
            l0_c2 = vld1q_f32(l11_ptr+8);
            l0_c3 = vld1q_f32(l11_ptr+12);
            l11_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k5_ptr);
            k0_c1 = vld1q_f32(k5_ptr+4);
            k0_c2 = vld1q_f32(k5_ptr+8);
            k0_c3 = vld1q_f32(k5_ptr+12);
            k5_ptr += 16;

            l0_c0 = vld1q_f32(l12_ptr);
            l0_c1 = vld1q_f32(l12_ptr+4);
            l0_c2 = vld1q_f32(l12_ptr+8);
            l0_c3 = vld1q_f32(l12_ptr+12);
            l12_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            if(b_ptr!=nullptr)
            {
                b0_c0 = vld1q_f32(b_ptr);
                b0_c1 = vld1q_f32(b_ptr+4);
                b0_c2 = vld1q_f32(b_ptr+8);
                b0_c3 = vld1q_f32(b_ptr+12);
                b_ptr += 16;

                r0_c0 = vaddq_f32(r0_c0, b0_c0);
                r0_c1 = vaddq_f32(r0_c1, b0_c1);
                r0_c2 = vaddq_f32(r0_c2, b0_c2);
                r0_c3 = vaddq_f32(r0_c3, b0_c3);
            }
            //
            if(act>=0)
            {
                r0_c0 = vmaxq_f32(r0_c0, v_actz);
                r0_c1 = vmaxq_f32(r0_c1, v_actz);
                r0_c2 = vmaxq_f32(r0_c2, v_actz);
                r0_c3 = vmaxq_f32(r0_c3, v_actz);
                if(act>0)
                {
                    r0_c0 = vminq_f32(r0_c0, v_actx);
                    r0_c1 = vminq_f32(r0_c1, v_actx);
                    r0_c2 = vminq_f32(r0_c2, v_actx);
                    r0_c3 = vminq_f32(r0_c3, v_actx);
                }
            }
            //
            vst1q_f32(out, r0_c0);
            vst1q_f32(out+4, r0_c1);
            vst1q_f32(out+8, r0_c2);
            vst1q_f32(out+12, r0_c3);
            out += 16;
            //
            r0_c0 = vdupq_n_f32(0);
            r0_c1 = vdupq_n_f32(0);
            r0_c2 = vdupq_n_f32(0);
            r0_c3 = vdupq_n_f32(0);
        }
        for( ; lc<in_c; lc++)
        {
            float k0_p = *k0_ptr++;
            float l00_p = *l00_ptr++;
            float k1_p = *k1_ptr++;
            float l01_p = *l01_ptr++;
            float k2_p = *k2_ptr++;
            float l02_p = *l02_ptr++;
            float k3_p = *k3_ptr++;
            float l10_p = *l10_ptr++;
            float k4_p = *k4_ptr++;
            float l11_p = *l11_ptr++;
            float k5_p = *k5_ptr++;
            float l12_p = *l12_ptr++;
            float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k2_p * l02_p) + (k3_p * l10_p) + (k4_p * l11_p) + (k5_p * l12_p);
            if(b_ptr!=nullptr)
            {
                r0_p = r0_p + *b_ptr++;
            }
            if(act>=0)
            {
                r0_p = std::max(r0_p, 0.f);
                if(act>0)
                    r0_p = std::min(r0_p, (float)act);
            }
            *out++ = r0_p;
        }

        k0_ptr = kernel;
        k1_ptr = kernel + in_c;
        k2_ptr = kernel + 2 * in_c;
        k3_ptr = k2_ptr + 1 * in_c;
        k4_ptr = k2_ptr + 2 * in_c;
        k5_ptr = k2_ptr + 3 * in_c;
   
        b_ptr = bias;
    }

//bottom right point
    k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    
    b_ptr = bias;

    lc = 0;

    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k0_ptr);
        k0_c1 = vld1q_f32(k0_ptr+4);
        k0_c2 = vld1q_f32(k0_ptr+8);
        k0_c3 = vld1q_f32(k0_ptr+12);
        k0_ptr += 16;
        
        l0_c0 = vld1q_f32(l00_ptr);
        l0_c1 = vld1q_f32(l00_ptr+4);
        l0_c2 = vld1q_f32(l00_ptr+8);
        l0_c3 = vld1q_f32(l00_ptr+12);
        l00_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k3_ptr);
        k0_c1 = vld1q_f32(k3_ptr+4);
        k0_c2 = vld1q_f32(k3_ptr+8);
        k0_c3 = vld1q_f32(k3_ptr+12);
        k3_ptr += 16;

        l0_c0 = vld1q_f32(l10_ptr);
        l0_c1 = vld1q_f32(l10_ptr+4);
        l0_c2 = vld1q_f32(l10_ptr+8);
        l0_c3 = vld1q_f32(l10_ptr+12);
        l10_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k0_p = *k0_ptr++;
        float l00_p = *l00_ptr++;
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k3_p = *k3_ptr++;
        float l10_p = *l10_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k3_p * l10_p) + (k4_p * l11_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }
}







static void k3s1p1_nhwc_fp32_hmtc(float* input, float* kernel, float* out, float* bias, long act, long in_w, long in_h, long in_c, long out_w, long out_h)
{
    float* k1_ptr = kernel + in_c;
    float* k2_ptr = kernel + 2 * in_c;
    float* k4_ptr = k2_ptr + 2 * in_c;
    float* k5_ptr = k2_ptr + 3 * in_c;
    float* k7_ptr = k5_ptr + 2 * in_c;
    float* k8_ptr = k5_ptr + 3 * in_c;

    float* l01_ptr = input;
    float* l02_ptr = input + in_c;
    float* l11_ptr = input + in_w * in_c;
    float* l12_ptr = input + in_w * in_c + in_c;
    float* l21_ptr = input + 2 * in_w * in_c;
    float* l22_ptr = input + 2 * in_w * in_c + in_c;
    
    float* b_ptr = bias;

    int lc = 0;

    float32x4_t k0_c0, k0_c1, k0_c2, k0_c3;
    float32x4_t l0_c0, l0_c1, l0_c2, l0_c3;
    float32x4_t b0_c0, b0_c1, b0_c2, b0_c3;
    float32x4_t r0_c0, r0_c1, r0_c2, r0_c3;

    float32x4_t v_actz = vdupq_n_f32(0);
    float32x4_t v_actx = vdupq_n_f32(act);

    r0_c0 = vdupq_n_f32(0);
    r0_c1 = vdupq_n_f32(0);
    r0_c2 = vdupq_n_f32(0);
    r0_c3 = vdupq_n_f32(0);

//middle left point
    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k2_ptr);
        k0_c1 = vld1q_f32(k2_ptr+4);
        k0_c2 = vld1q_f32(k2_ptr+8);
        k0_c3 = vld1q_f32(k2_ptr+12);
        k2_ptr += 16;
        
        l0_c0 = vld1q_f32(l02_ptr);
        l0_c1 = vld1q_f32(l02_ptr+4);
        l0_c2 = vld1q_f32(l02_ptr+8);
        l0_c3 = vld1q_f32(l02_ptr+12);
        l02_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k5_ptr);
        k0_c1 = vld1q_f32(k5_ptr+4);
        k0_c2 = vld1q_f32(k5_ptr+8);
        k0_c3 = vld1q_f32(k5_ptr+12);
        k5_ptr += 16;

        l0_c0 = vld1q_f32(l12_ptr);
        l0_c1 = vld1q_f32(l12_ptr+4);
        l0_c2 = vld1q_f32(l12_ptr+8);
        l0_c3 = vld1q_f32(l12_ptr+12);
        l12_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k7_ptr);
        k0_c1 = vld1q_f32(k7_ptr+4);
        k0_c2 = vld1q_f32(k7_ptr+8);
        k0_c3 = vld1q_f32(k7_ptr+12);
        k7_ptr += 16;

        l0_c0 = vld1q_f32(l21_ptr);
        l0_c1 = vld1q_f32(l21_ptr+4);
        l0_c2 = vld1q_f32(l21_ptr+8);
        l0_c3 = vld1q_f32(l21_ptr+12);
        l21_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k8_ptr);
        k0_c1 = vld1q_f32(k8_ptr+4);
        k0_c2 = vld1q_f32(k8_ptr+8);
        k0_c3 = vld1q_f32(k8_ptr+12);
        k8_ptr += 16;

        l0_c0 = vld1q_f32(l22_ptr);
        l0_c1 = vld1q_f32(l22_ptr+4);
        l0_c2 = vld1q_f32(l22_ptr+8);
        l0_c3 = vld1q_f32(l22_ptr+12);
        l22_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        { 
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k2_p = *k2_ptr++;
        float l02_p = *l02_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float k5_p = *k5_ptr++;
        float l12_p = *l12_ptr++;
        float k7_p = *k7_ptr++;
        float l21_p = *l21_ptr++;
        float k8_p = *k8_ptr++;
        float l22_p = *l22_ptr++;
        float r0_p = (k1_p * l01_p) + (k2_p * l02_p) + (k4_p * l11_p) + (k5_p * l12_p) + (k7_p * l21_p) + (k8_p * l22_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }

//middle middle points
    float* k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k2_ptr = kernel + 2 * in_c;
    float* k3_ptr = k2_ptr + in_c;
    k4_ptr = k2_ptr + 2 * in_c;
    k5_ptr = k2_ptr + 3 * in_c;
    float* k6_ptr = k5_ptr + in_c;
    k7_ptr = k5_ptr + 2 * in_c;
    k8_ptr = k5_ptr + 3 * in_c;

    float* l00_ptr = input;
    l01_ptr = input + in_c;
    l02_ptr = input + 2 * in_c;
    float* l10_ptr = input + in_w * in_c;
    l11_ptr = input + in_w * in_c + in_c;
    l12_ptr = input + in_w * in_c + 2 * in_c;
    float* l20_ptr = input + 2 * in_w * in_c;
    l21_ptr = input + 2 * in_w * in_c + in_c;
    l22_ptr = input + 2 * in_w * in_c + 2 * in_c;
   
    b_ptr = bias;

    for(int lw=0; lw<(out_w-2); lw++)
    {
        lc = 0;

        for(lc=0; (lc+16)<=in_c; lc+=16)
        {
            k0_c0 = vld1q_f32(k0_ptr);
            k0_c1 = vld1q_f32(k0_ptr+4);
            k0_c2 = vld1q_f32(k0_ptr+8);
            k0_c3 = vld1q_f32(k0_ptr+12);
            k0_ptr += 16;
            
            l0_c0 = vld1q_f32(l00_ptr);
            l0_c1 = vld1q_f32(l00_ptr+4);
            l0_c2 = vld1q_f32(l00_ptr+8);
            l0_c3 = vld1q_f32(l00_ptr+12);
            l00_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k1_ptr);
            k0_c1 = vld1q_f32(k1_ptr+4);
            k0_c2 = vld1q_f32(k1_ptr+8);
            k0_c3 = vld1q_f32(k1_ptr+12);
            k1_ptr += 16;
            
            l0_c0 = vld1q_f32(l01_ptr);
            l0_c1 = vld1q_f32(l01_ptr+4);
            l0_c2 = vld1q_f32(l01_ptr+8);
            l0_c3 = vld1q_f32(l01_ptr+12);
            l01_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k2_ptr);
            k0_c1 = vld1q_f32(k2_ptr+4);
            k0_c2 = vld1q_f32(k2_ptr+8);
            k0_c3 = vld1q_f32(k2_ptr+12);
            k2_ptr += 16;
            
            l0_c0 = vld1q_f32(l02_ptr);
            l0_c1 = vld1q_f32(l02_ptr+4);
            l0_c2 = vld1q_f32(l02_ptr+8);
            l0_c3 = vld1q_f32(l02_ptr+12);
            l02_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k3_ptr);
            k0_c1 = vld1q_f32(k3_ptr+4);
            k0_c2 = vld1q_f32(k3_ptr+8);
            k0_c3 = vld1q_f32(k3_ptr+12);
            k3_ptr += 16;

            l0_c0 = vld1q_f32(l10_ptr);
            l0_c1 = vld1q_f32(l10_ptr+4);
            l0_c2 = vld1q_f32(l10_ptr+8);
            l0_c3 = vld1q_f32(l10_ptr+12);
            l10_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k4_ptr);
            k0_c1 = vld1q_f32(k4_ptr+4);
            k0_c2 = vld1q_f32(k4_ptr+8);
            k0_c3 = vld1q_f32(k4_ptr+12);
            k4_ptr += 16;

            l0_c0 = vld1q_f32(l11_ptr);
            l0_c1 = vld1q_f32(l11_ptr+4);
            l0_c2 = vld1q_f32(l11_ptr+8);
            l0_c3 = vld1q_f32(l11_ptr+12);
            l11_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k5_ptr);
            k0_c1 = vld1q_f32(k5_ptr+4);
            k0_c2 = vld1q_f32(k5_ptr+8);
            k0_c3 = vld1q_f32(k5_ptr+12);
            k5_ptr += 16;

            l0_c0 = vld1q_f32(l12_ptr);
            l0_c1 = vld1q_f32(l12_ptr+4);
            l0_c2 = vld1q_f32(l12_ptr+8);
            l0_c3 = vld1q_f32(l12_ptr+12);
            l12_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k6_ptr);
            k0_c1 = vld1q_f32(k6_ptr+4);
            k0_c2 = vld1q_f32(k6_ptr+8);
            k0_c3 = vld1q_f32(k6_ptr+12);
            k6_ptr += 16;

            l0_c0 = vld1q_f32(l20_ptr);
            l0_c1 = vld1q_f32(l20_ptr+4);
            l0_c2 = vld1q_f32(l20_ptr+8);
            l0_c3 = vld1q_f32(l20_ptr+12);
            l20_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k7_ptr);
            k0_c1 = vld1q_f32(k7_ptr+4);
            k0_c2 = vld1q_f32(k7_ptr+8);
            k0_c3 = vld1q_f32(k7_ptr+12);
            k7_ptr += 16;

            l0_c0 = vld1q_f32(l21_ptr);
            l0_c1 = vld1q_f32(l21_ptr+4);
            l0_c2 = vld1q_f32(l21_ptr+8);
            l0_c3 = vld1q_f32(l21_ptr+12);
            l21_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k8_ptr);
            k0_c1 = vld1q_f32(k8_ptr+4);
            k0_c2 = vld1q_f32(k8_ptr+8);
            k0_c3 = vld1q_f32(k8_ptr+12);
            k8_ptr += 16;

            l0_c0 = vld1q_f32(l22_ptr);
            l0_c1 = vld1q_f32(l22_ptr+4);
            l0_c2 = vld1q_f32(l22_ptr+8);
            l0_c3 = vld1q_f32(l22_ptr+12);
            l22_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            if(b_ptr!=nullptr)
            {
                b0_c0 = vld1q_f32(b_ptr);
                b0_c1 = vld1q_f32(b_ptr+4);
                b0_c2 = vld1q_f32(b_ptr+8);
                b0_c3 = vld1q_f32(b_ptr+12);
                b_ptr += 16;

                r0_c0 = vaddq_f32(r0_c0, b0_c0);
                r0_c1 = vaddq_f32(r0_c1, b0_c1);
                r0_c2 = vaddq_f32(r0_c2, b0_c2);
                r0_c3 = vaddq_f32(r0_c3, b0_c3);
            }
            //
            if(act>=0)
            {
                r0_c0 = vmaxq_f32(r0_c0, v_actz);
                r0_c1 = vmaxq_f32(r0_c1, v_actz);
                r0_c2 = vmaxq_f32(r0_c2, v_actz);
                r0_c3 = vmaxq_f32(r0_c3, v_actz);
                if(act>0)
                {
                    r0_c0 = vminq_f32(r0_c0, v_actx);
                    r0_c1 = vminq_f32(r0_c1, v_actx);
                    r0_c2 = vminq_f32(r0_c2, v_actx);
                    r0_c3 = vminq_f32(r0_c3, v_actx);
                }
            }
            //
            vst1q_f32(out, r0_c0);
            vst1q_f32(out+4, r0_c1);
            vst1q_f32(out+8, r0_c2);
            vst1q_f32(out+12, r0_c3);
            out += 16;
            //
            r0_c0 = vdupq_n_f32(0);
            r0_c1 = vdupq_n_f32(0);
            r0_c2 = vdupq_n_f32(0);
            r0_c3 = vdupq_n_f32(0);
        }
        for( ; lc<in_c; lc++)
        {
            float k0_p = *k0_ptr++;
            float l00_p = *l00_ptr++;
            float k1_p = *k1_ptr++;
            float l01_p = *l01_ptr++;
            float k2_p = *k2_ptr++;
            float l02_p = *l02_ptr++;
            float k3_p = *k3_ptr++;
            float l10_p = *l10_ptr++;
            float k4_p = *k4_ptr++;
            float l11_p = *l11_ptr++;
            float k5_p = *k5_ptr++;
            float l12_p = *l12_ptr++;
            float k6_p = *k6_ptr++;
            float l20_p = *l20_ptr++;
            float k7_p = *k7_ptr++;
            float l21_p = *l21_ptr++;
            float k8_p = *k8_ptr++;
            float l22_p = *l22_ptr++;
            float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k2_p * l02_p) + (k3_p * l10_p) + (k4_p * l11_p) + (k5_p * l12_p) + (k6_p * l20_p) + (k7_p * l21_p) + (k8_p * l22_p);
            if(b_ptr!=nullptr)
            {
                r0_p = r0_p + *b_ptr++;
            }
            if(act>=0)
            {
                r0_p = std::max(r0_p, 0.f);
                if(act>0)
                    r0_p = std::min(r0_p, (float)act);
            }
            *out++ = r0_p;
        }

        k0_ptr = kernel;
        k1_ptr = kernel + in_c;
        k2_ptr = kernel + 2 * in_c;
        k3_ptr = k2_ptr + 1 * in_c;
        k4_ptr = k2_ptr + 2 * in_c;
        k5_ptr = k2_ptr + 3 * in_c;
        k6_ptr = k5_ptr + 1 * in_c;
        k7_ptr = k5_ptr + 2 * in_c;
        k8_ptr = k5_ptr + 3 * in_c;
   
        b_ptr = bias;
    }

//middle right point
    k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    k6_ptr = kernel + 6 * in_c;
    k7_ptr = kernel + 7 * in_c;
    
    b_ptr = bias;

    lc = 0;

    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k0_ptr);
        k0_c1 = vld1q_f32(k0_ptr+4);
        k0_c2 = vld1q_f32(k0_ptr+8);
        k0_c3 = vld1q_f32(k0_ptr+12);
        k0_ptr += 16;
        
        l0_c0 = vld1q_f32(l00_ptr);
        l0_c1 = vld1q_f32(l00_ptr+4);
        l0_c2 = vld1q_f32(l00_ptr+8);
        l0_c3 = vld1q_f32(l00_ptr+12);
        l00_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k3_ptr);
        k0_c1 = vld1q_f32(k3_ptr+4);
        k0_c2 = vld1q_f32(k3_ptr+8);
        k0_c3 = vld1q_f32(k3_ptr+12);
        k3_ptr += 16;

        l0_c0 = vld1q_f32(l10_ptr);
        l0_c1 = vld1q_f32(l10_ptr+4);
        l0_c2 = vld1q_f32(l10_ptr+8);
        l0_c3 = vld1q_f32(l10_ptr+12);
        l10_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k6_ptr);
        k0_c1 = vld1q_f32(k6_ptr+4);
        k0_c2 = vld1q_f32(k6_ptr+8);
        k0_c3 = vld1q_f32(k6_ptr+12);
        k6_ptr += 16;

        l0_c0 = vld1q_f32(l20_ptr);
        l0_c1 = vld1q_f32(l20_ptr+4);
        l0_c2 = vld1q_f32(l20_ptr+8);
        l0_c3 = vld1q_f32(l20_ptr+12);
        l20_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k7_ptr);
        k0_c1 = vld1q_f32(k7_ptr+4);
        k0_c2 = vld1q_f32(k7_ptr+8);
        k0_c3 = vld1q_f32(k7_ptr+12);
        k7_ptr += 16;

        l0_c0 = vld1q_f32(l21_ptr);
        l0_c1 = vld1q_f32(l21_ptr+4);
        l0_c2 = vld1q_f32(l21_ptr+8);
        l0_c3 = vld1q_f32(l21_ptr+12);
        l21_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k0_p = *k0_ptr++;
        float l00_p = *l00_ptr++;
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k3_p = *k3_ptr++;
        float l10_p = *l10_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float k6_p = *k6_ptr++;
        float l20_p = *l20_ptr++;
        float k7_p = *k7_ptr++;
        float l21_p = *l21_ptr++;
        float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k3_p * l10_p) + (k4_p * l11_p) + (k6_p * l20_p) + (k7_p * l21_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }
}
//for kernel k3s1p1 end



//for kernel k3s2p0p1 start
static void k3s2p0p1_nhwc_fp32_hetc(float* input, float* kernel, float* out, float* bias, long act, long in_w, long in_h, long in_c, long out_w, long out_h)
{
    float* k0_ptr = kernel;
    float* k1_ptr = kernel + in_c;
    float* k2_ptr = kernel + 2 * in_c;
    float* k3_ptr = k2_ptr + in_c;
    float* k4_ptr = k2_ptr + 2 * in_c;
    float* k5_ptr = k2_ptr + 3 * in_c;

    float* l00_ptr = input;
    float* l01_ptr = input + in_c;
    float* l02_ptr = input + 2 * in_c;
    float* l10_ptr = input + in_w * in_c;
    float* l11_ptr = input + in_w * in_c + in_c;
    float* l12_ptr = input + in_w * in_c + 2 * in_c;
    
    float* b_ptr = bias;

    int lc = 0;

    float32x4_t k0_c0, k0_c1, k0_c2, k0_c3;
    float32x4_t l0_c0, l0_c1, l0_c2, l0_c3;
    float32x4_t b0_c0, b0_c1, b0_c2, b0_c3;
    float32x4_t r0_c0, r0_c1, r0_c2, r0_c3;

    float32x4_t v_actz = vdupq_n_f32(0);
    float32x4_t v_actx = vdupq_n_f32(act);

    r0_c0 = vdupq_n_f32(0);
    r0_c1 = vdupq_n_f32(0);
    r0_c2 = vdupq_n_f32(0);
    r0_c3 = vdupq_n_f32(0);

//bottom left points
    for(int lw=0; lw<(out_w-1); lw++)
    {
        lc = 0;

        for(lc=0; (lc+16)<=in_c; lc+=16)
        {
            k0_c0 = vld1q_f32(k0_ptr);
            k0_c1 = vld1q_f32(k0_ptr+4);
            k0_c2 = vld1q_f32(k0_ptr+8);
            k0_c3 = vld1q_f32(k0_ptr+12);
            k0_ptr += 16;
            
            l0_c0 = vld1q_f32(l00_ptr);
            l0_c1 = vld1q_f32(l00_ptr+4);
            l0_c2 = vld1q_f32(l00_ptr+8);
            l0_c3 = vld1q_f32(l00_ptr+12);
            l00_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k1_ptr);
            k0_c1 = vld1q_f32(k1_ptr+4);
            k0_c2 = vld1q_f32(k1_ptr+8);
            k0_c3 = vld1q_f32(k1_ptr+12);
            k1_ptr += 16;
            
            l0_c0 = vld1q_f32(l01_ptr);
            l0_c1 = vld1q_f32(l01_ptr+4);
            l0_c2 = vld1q_f32(l01_ptr+8);
            l0_c3 = vld1q_f32(l01_ptr+12);
            l01_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k2_ptr);
            k0_c1 = vld1q_f32(k2_ptr+4);
            k0_c2 = vld1q_f32(k2_ptr+8);
            k0_c3 = vld1q_f32(k2_ptr+12);
            k2_ptr += 16;
            
            l0_c0 = vld1q_f32(l02_ptr);
            l0_c1 = vld1q_f32(l02_ptr+4);
            l0_c2 = vld1q_f32(l02_ptr+8);
            l0_c3 = vld1q_f32(l02_ptr+12);
            l02_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k3_ptr);
            k0_c1 = vld1q_f32(k3_ptr+4);
            k0_c2 = vld1q_f32(k3_ptr+8);
            k0_c3 = vld1q_f32(k3_ptr+12);
            k3_ptr += 16;

            l0_c0 = vld1q_f32(l10_ptr);
            l0_c1 = vld1q_f32(l10_ptr+4);
            l0_c2 = vld1q_f32(l10_ptr+8);
            l0_c3 = vld1q_f32(l10_ptr+12);
            l10_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k4_ptr);
            k0_c1 = vld1q_f32(k4_ptr+4);
            k0_c2 = vld1q_f32(k4_ptr+8);
            k0_c3 = vld1q_f32(k4_ptr+12);
            k4_ptr += 16;

            l0_c0 = vld1q_f32(l11_ptr);
            l0_c1 = vld1q_f32(l11_ptr+4);
            l0_c2 = vld1q_f32(l11_ptr+8);
            l0_c3 = vld1q_f32(l11_ptr+12);
            l11_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k5_ptr);
            k0_c1 = vld1q_f32(k5_ptr+4);
            k0_c2 = vld1q_f32(k5_ptr+8);
            k0_c3 = vld1q_f32(k5_ptr+12);
            k5_ptr += 16;

            l0_c0 = vld1q_f32(l12_ptr);
            l0_c1 = vld1q_f32(l12_ptr+4);
            l0_c2 = vld1q_f32(l12_ptr+8);
            l0_c3 = vld1q_f32(l12_ptr+12);
            l12_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            if(b_ptr!=nullptr)
            {
                b0_c0 = vld1q_f32(b_ptr);
                b0_c1 = vld1q_f32(b_ptr+4);
                b0_c2 = vld1q_f32(b_ptr+8);
                b0_c3 = vld1q_f32(b_ptr+12);
                b_ptr += 16;

                r0_c0 = vaddq_f32(r0_c0, b0_c0);
                r0_c1 = vaddq_f32(r0_c1, b0_c1);
                r0_c2 = vaddq_f32(r0_c2, b0_c2);
                r0_c3 = vaddq_f32(r0_c3, b0_c3);
            }
            //
            if(act>=0)
            {
                r0_c0 = vmaxq_f32(r0_c0, v_actz);
                r0_c1 = vmaxq_f32(r0_c1, v_actz);
                r0_c2 = vmaxq_f32(r0_c2, v_actz);
                r0_c3 = vmaxq_f32(r0_c3, v_actz);
                if(act>0)
                {
                    r0_c0 = vminq_f32(r0_c0, v_actx);
                    r0_c1 = vminq_f32(r0_c1, v_actx);
                    r0_c2 = vminq_f32(r0_c2, v_actx);
                    r0_c3 = vminq_f32(r0_c3, v_actx);
                }
            }
            //
            vst1q_f32(out, r0_c0);
            vst1q_f32(out+4, r0_c1);
            vst1q_f32(out+8, r0_c2);
            vst1q_f32(out+12, r0_c3);
            out += 16;
            //
            r0_c0 = vdupq_n_f32(0);
            r0_c1 = vdupq_n_f32(0);
            r0_c2 = vdupq_n_f32(0);
            r0_c3 = vdupq_n_f32(0);
        }
        for( ; lc<in_c; lc++)
        {
            float k0_p = *k0_ptr++;
            float l00_p = *l00_ptr++;
            float k1_p = *k1_ptr++;
            float l01_p = *l01_ptr++;
            float k2_p = *k2_ptr++;
            float l02_p = *l02_ptr++;
            float k3_p = *k3_ptr++;
            float l10_p = *l10_ptr++;
            float k4_p = *k4_ptr++;
            float l11_p = *l11_ptr++;
            float k5_p = *k5_ptr++;
            float l12_p = *l12_ptr++;
            float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k2_p * l02_p) + (k3_p * l10_p) + (k4_p * l11_p) + (k5_p * l12_p);
            if(b_ptr!=nullptr)
            {
                r0_p = r0_p + *b_ptr++;
            }
            if(act>=0)
            {
                r0_p = std::max(r0_p, 0.f);
                if(act>0)
                    r0_p = std::min(r0_p, (float)act);
            }
            *out++ = r0_p;
        }

        k0_ptr = kernel;
        k1_ptr = kernel + in_c;
        k2_ptr = kernel + 2 * in_c;
        k3_ptr = k2_ptr + 1 * in_c;
        k4_ptr = k2_ptr + 2 * in_c;
        k5_ptr = k2_ptr + 3 * in_c;
   
        l00_ptr = l01_ptr;
        l01_ptr = l02_ptr;
        l02_ptr += in_c;
        l10_ptr = l11_ptr;
        l11_ptr = l12_ptr;
        l12_ptr += in_c;

        b_ptr = bias;
    }

//bottom right point
    k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    
    lc = 0;

    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k0_ptr);
        k0_c1 = vld1q_f32(k0_ptr+4);
        k0_c2 = vld1q_f32(k0_ptr+8);
        k0_c3 = vld1q_f32(k0_ptr+12);
        k0_ptr += 16;
        
        l0_c0 = vld1q_f32(l00_ptr);
        l0_c1 = vld1q_f32(l00_ptr+4);
        l0_c2 = vld1q_f32(l00_ptr+8);
        l0_c3 = vld1q_f32(l00_ptr+12);
        l00_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k3_ptr);
        k0_c1 = vld1q_f32(k3_ptr+4);
        k0_c2 = vld1q_f32(k3_ptr+8);
        k0_c3 = vld1q_f32(k3_ptr+12);
        k3_ptr += 16;

        l0_c0 = vld1q_f32(l10_ptr);
        l0_c1 = vld1q_f32(l10_ptr+4);
        l0_c2 = vld1q_f32(l10_ptr+8);
        l0_c3 = vld1q_f32(l10_ptr+12);
        l10_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k0_p = *k0_ptr++;
        float l00_p = *l00_ptr++;
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k3_p = *k3_ptr++;
        float l10_p = *l10_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k3_p * l10_p) + (k4_p * l11_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }
}







static void k3s2p0p1_nhwc_fp32_hstc(float* input, float* kernel, float* out, float* bias, long act, long in_w, long in_h, long in_c, long out_w, long out_h)
{
    float* k0_ptr = kernel;
    float* k1_ptr = kernel + in_c;
    float* k2_ptr = kernel + 2 * in_c;
    float* k3_ptr = k2_ptr + in_c;
    float* k4_ptr = k2_ptr + 2 * in_c;
    float* k5_ptr = k2_ptr + 3 * in_c;
    float* k6_ptr = k5_ptr + in_c;
    float* k7_ptr = k5_ptr + 2 * in_c;
    float* k8_ptr = k5_ptr + 3 * in_c;

    float* l00_ptr = input;
    float* l01_ptr = input + in_c;
    float* l02_ptr = input + 2 * in_c;
    float* l10_ptr = input + in_w * in_c;
    float* l11_ptr = input + in_w * in_c + in_c;
    float* l12_ptr = input + in_w * in_c + 2 * in_c;
    float* l20_ptr = input + 2 * in_w * in_c;
    float* l21_ptr = input + 2 * in_w * in_c + in_c;
    float* l22_ptr = input + 2 * in_w * in_c + 2 * in_c;

    float* b_ptr = bias;

    int lc = 0;

    float32x4_t k0_c0, k0_c1, k0_c2, k0_c3;
    float32x4_t l0_c0, l0_c1, l0_c2, l0_c3;
    float32x4_t b0_c0, b0_c1, b0_c2, b0_c3;
    float32x4_t r0_c0, r0_c1, r0_c2, r0_c3;

    float32x4_t v_actz = vdupq_n_f32(0);
    float32x4_t v_actx = vdupq_n_f32(act);

    r0_c0 = vdupq_n_f32(0);
    r0_c1 = vdupq_n_f32(0);
    r0_c2 = vdupq_n_f32(0);
    r0_c3 = vdupq_n_f32(0);

//top left points
    for(int lw=0; lw<(out_w-1); lw++)
    {
        lc = 0;

        for(lc=0; (lc+16)<=in_c; lc+=16)
        {
            k0_c0 = vld1q_f32(k0_ptr);
            k0_c1 = vld1q_f32(k0_ptr+4);
            k0_c2 = vld1q_f32(k0_ptr+8);
            k0_c3 = vld1q_f32(k0_ptr+12);
            k0_ptr += 16;
            
            l0_c0 = vld1q_f32(l00_ptr);
            l0_c1 = vld1q_f32(l00_ptr+4);
            l0_c2 = vld1q_f32(l00_ptr+8);
            l0_c3 = vld1q_f32(l00_ptr+12);
            l00_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k1_ptr);
            k0_c1 = vld1q_f32(k1_ptr+4);
            k0_c2 = vld1q_f32(k1_ptr+8);
            k0_c3 = vld1q_f32(k1_ptr+12);
            k1_ptr += 16;
            
            l0_c0 = vld1q_f32(l01_ptr);
            l0_c1 = vld1q_f32(l01_ptr+4);
            l0_c2 = vld1q_f32(l01_ptr+8);
            l0_c3 = vld1q_f32(l01_ptr+12);
            l01_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k2_ptr);
            k0_c1 = vld1q_f32(k2_ptr+4);
            k0_c2 = vld1q_f32(k2_ptr+8);
            k0_c3 = vld1q_f32(k2_ptr+12);
            k2_ptr += 16;
            
            l0_c0 = vld1q_f32(l02_ptr);
            l0_c1 = vld1q_f32(l02_ptr+4);
            l0_c2 = vld1q_f32(l02_ptr+8);
            l0_c3 = vld1q_f32(l02_ptr+12);
            l02_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k3_ptr);
            k0_c1 = vld1q_f32(k3_ptr+4);
            k0_c2 = vld1q_f32(k3_ptr+8);
            k0_c3 = vld1q_f32(k3_ptr+12);
            k3_ptr += 16;

            l0_c0 = vld1q_f32(l10_ptr);
            l0_c1 = vld1q_f32(l10_ptr+4);
            l0_c2 = vld1q_f32(l10_ptr+8);
            l0_c3 = vld1q_f32(l10_ptr+12);
            l10_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k4_ptr);
            k0_c1 = vld1q_f32(k4_ptr+4);
            k0_c2 = vld1q_f32(k4_ptr+8);
            k0_c3 = vld1q_f32(k4_ptr+12);
            k4_ptr += 16;

            l0_c0 = vld1q_f32(l11_ptr);
            l0_c1 = vld1q_f32(l11_ptr+4);
            l0_c2 = vld1q_f32(l11_ptr+8);
            l0_c3 = vld1q_f32(l11_ptr+12);
            l11_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k5_ptr);
            k0_c1 = vld1q_f32(k5_ptr+4);
            k0_c2 = vld1q_f32(k5_ptr+8);
            k0_c3 = vld1q_f32(k5_ptr+12);
            k5_ptr += 16;

            l0_c0 = vld1q_f32(l12_ptr);
            l0_c1 = vld1q_f32(l12_ptr+4);
            l0_c2 = vld1q_f32(l12_ptr+8);
            l0_c3 = vld1q_f32(l12_ptr+12);
            l12_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k6_ptr);
            k0_c1 = vld1q_f32(k6_ptr+4);
            k0_c2 = vld1q_f32(k6_ptr+8);
            k0_c3 = vld1q_f32(k6_ptr+12);
            k6_ptr += 16;

            l0_c0 = vld1q_f32(l20_ptr);
            l0_c1 = vld1q_f32(l20_ptr+4);
            l0_c2 = vld1q_f32(l20_ptr+8);
            l0_c3 = vld1q_f32(l20_ptr+12);
            l20_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k7_ptr);
            k0_c1 = vld1q_f32(k7_ptr+4);
            k0_c2 = vld1q_f32(k7_ptr+8);
            k0_c3 = vld1q_f32(k7_ptr+12);
            k7_ptr += 16;

            l0_c0 = vld1q_f32(l21_ptr);
            l0_c1 = vld1q_f32(l21_ptr+4);
            l0_c2 = vld1q_f32(l21_ptr+8);
            l0_c3 = vld1q_f32(l21_ptr+12);
            l21_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k8_ptr);
            k0_c1 = vld1q_f32(k8_ptr+4);
            k0_c2 = vld1q_f32(k8_ptr+8);
            k0_c3 = vld1q_f32(k8_ptr+12);
            k8_ptr += 16;

            l0_c0 = vld1q_f32(l22_ptr);
            l0_c1 = vld1q_f32(l22_ptr+4);
            l0_c2 = vld1q_f32(l22_ptr+8);
            l0_c3 = vld1q_f32(l22_ptr+12);
            l22_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            if(b_ptr!=nullptr)
            {
                b0_c0 = vld1q_f32(b_ptr);
                b0_c1 = vld1q_f32(b_ptr+4);
                b0_c2 = vld1q_f32(b_ptr+8);
                b0_c3 = vld1q_f32(b_ptr+12);
                b_ptr += 16;

                r0_c0 = vaddq_f32(r0_c0, b0_c0);
                r0_c1 = vaddq_f32(r0_c1, b0_c1);
                r0_c2 = vaddq_f32(r0_c2, b0_c2);
                r0_c3 = vaddq_f32(r0_c3, b0_c3);
            }
            //
            if(act>=0)
            {
                r0_c0 = vmaxq_f32(r0_c0, v_actz);
                r0_c1 = vmaxq_f32(r0_c1, v_actz);
                r0_c2 = vmaxq_f32(r0_c2, v_actz);
                r0_c3 = vmaxq_f32(r0_c3, v_actz);
                if(act>0)
                {
                    r0_c0 = vminq_f32(r0_c0, v_actx);
                    r0_c1 = vminq_f32(r0_c1, v_actx);
                    r0_c2 = vminq_f32(r0_c2, v_actx);
                    r0_c3 = vminq_f32(r0_c3, v_actx);
                }
            }
            //
            vst1q_f32(out, r0_c0);
            vst1q_f32(out+4, r0_c1);
            vst1q_f32(out+8, r0_c2);
            vst1q_f32(out+12, r0_c3);
            out += 16;
            //
            r0_c0 = vdupq_n_f32(0);
            r0_c1 = vdupq_n_f32(0);
            r0_c2 = vdupq_n_f32(0);
            r0_c3 = vdupq_n_f32(0);
        }
        for( ; lc<in_c; lc++)
        {
            float k0_p = *k0_ptr++;
            float l00_p = *l00_ptr++;
            float k1_p = *k1_ptr++;
            float l01_p = *l01_ptr++;
            float k2_p = *k2_ptr++;
            float l02_p = *l02_ptr++;
            float k3_p = *k3_ptr++;
            float l10_p = *l10_ptr++;
            float k4_p = *k4_ptr++;
            float l11_p = *l11_ptr++;
            float k5_p = *k5_ptr++;
            float l12_p = *l12_ptr++;
            float k6_p = *k6_ptr++;
            float l20_p = *l20_ptr++;
            float k7_p = *k7_ptr++;
            float l21_p = *l21_ptr++;
            float k8_p = *k8_ptr++;
            float l22_p = *l22_ptr++;
            float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k2_p * l02_p) + (k3_p * l10_p) + (k4_p * l11_p) + (k5_p * l12_p) + (k6_p * l20_p) + (k7_p * l21_p) + (k8_p * l22_p);
            if(b_ptr!=nullptr)
            {
                r0_p = r0_p + *b_ptr++;
            }
            if(act>=0)
            {
                r0_p = std::max(r0_p, 0.f);
                if(act>0)
                    r0_p = std::min(r0_p, (float)act);
            }
            *out++ = r0_p;
        }

        k0_ptr = kernel;
        k1_ptr = kernel + in_c;
        k2_ptr = kernel + 2 * in_c;
        k3_ptr = k2_ptr + 1 * in_c;
        k4_ptr = k2_ptr + 2 * in_c;
        k5_ptr = k2_ptr + 3 * in_c;
        k6_ptr = k5_ptr + 1 * in_c;
        k7_ptr = k5_ptr + 2 * in_c;
        k8_ptr = k5_ptr + 3 * in_c;
   
        l00_ptr = l01_ptr; 
        l01_ptr = l02_ptr;
        l02_ptr += in_c; 
        l10_ptr = l11_ptr;
        l11_ptr = l12_ptr;
        l12_ptr += in_c;
        l20_ptr = l21_ptr;
        l21_ptr = l22_ptr;
        l22_ptr += in_c;

        b_ptr = bias;
    }

//top right point
    k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    k6_ptr = kernel + 6 * in_c;
    k7_ptr = kernel + 7 * in_c;
    
    lc = 0;

    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k0_ptr);
        k0_c1 = vld1q_f32(k0_ptr+4);
        k0_c2 = vld1q_f32(k0_ptr+8);
        k0_c3 = vld1q_f32(k0_ptr+12);
        k0_ptr += 16;
        
        l0_c0 = vld1q_f32(l00_ptr);
        l0_c1 = vld1q_f32(l00_ptr+4);
        l0_c2 = vld1q_f32(l00_ptr+8);
        l0_c3 = vld1q_f32(l00_ptr+12);
        l00_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k3_ptr);
        k0_c1 = vld1q_f32(k3_ptr+4);
        k0_c2 = vld1q_f32(k3_ptr+8);
        k0_c3 = vld1q_f32(k3_ptr+12);
        k3_ptr += 16;

        l0_c0 = vld1q_f32(l10_ptr);
        l0_c1 = vld1q_f32(l10_ptr+4);
        l0_c2 = vld1q_f32(l10_ptr+8);
        l0_c3 = vld1q_f32(l10_ptr+12);
        l10_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k6_ptr);
        k0_c1 = vld1q_f32(k6_ptr+4);
        k0_c2 = vld1q_f32(k6_ptr+8);
        k0_c3 = vld1q_f32(k6_ptr+12);
        k6_ptr += 16;

        l0_c0 = vld1q_f32(l20_ptr);
        l0_c1 = vld1q_f32(l20_ptr+4);
        l0_c2 = vld1q_f32(l20_ptr+8);
        l0_c3 = vld1q_f32(l20_ptr+12);
        l20_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k7_ptr);
        k0_c1 = vld1q_f32(k7_ptr+4);
        k0_c2 = vld1q_f32(k7_ptr+8);
        k0_c3 = vld1q_f32(k7_ptr+12);
        k7_ptr += 16;

        l0_c0 = vld1q_f32(l21_ptr);
        l0_c1 = vld1q_f32(l21_ptr+4);
        l0_c2 = vld1q_f32(l21_ptr+8);
        l0_c3 = vld1q_f32(l21_ptr+12);
        l21_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k0_p = *k0_ptr++;
        float l00_p = *l00_ptr++;
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k3_p = *k3_ptr++;
        float l10_p = *l10_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float k6_p = *k6_ptr++;
        float l20_p = *l20_ptr++;
        float k7_p = *k7_ptr++;
        float l21_p = *l21_ptr++;
        float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k3_p * l10_p) + (k4_p * l11_p) + (k6_p * l20_p) + (k7_p * l21_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }
}
//for kernel k3s2p0p1 end



//for kernel k3s2p1 start
static void k3s2p1_nhwc_fp32_hstc(float* input, float* kernel, float* out, float* bias, long act, long in_w, long in_h, long in_c, long out_w, long out_h)
{
    float* k4_ptr = kernel + 4 * in_c;
    float* k5_ptr = kernel + 5 * in_c;
    float* k7_ptr = k4_ptr + 3 * in_c;
    float* k8_ptr = k4_ptr + 4 * in_c;

    float* l01_ptr = input;
    float* l02_ptr = input + in_c;
    float* l11_ptr = input + in_w * in_c;
    float* l12_ptr = input + in_w * in_c + in_c;
    
    float* b_ptr = bias;

    int lc = 0;

    float32x4_t k0_c0, k0_c1, k0_c2, k0_c3;
    float32x4_t l0_c0, l0_c1, l0_c2, l0_c3;
    float32x4_t b0_c0, b0_c1, b0_c2, b0_c3;
    float32x4_t r0_c0, r0_c1, r0_c2, r0_c3;

    float32x4_t v_actz = vdupq_n_f32(0);
    float32x4_t v_actx = vdupq_n_f32(act);

    r0_c0 = vdupq_n_f32(0);
    r0_c1 = vdupq_n_f32(0);
    r0_c2 = vdupq_n_f32(0);
    r0_c3 = vdupq_n_f32(0);

//top left point
    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k5_ptr);
        k0_c1 = vld1q_f32(k5_ptr+4);
        k0_c2 = vld1q_f32(k5_ptr+8);
        k0_c3 = vld1q_f32(k5_ptr+12);
        k5_ptr += 16;
        
        l0_c0 = vld1q_f32(l02_ptr);
        l0_c1 = vld1q_f32(l02_ptr+4);
        l0_c2 = vld1q_f32(l02_ptr+8);
        l0_c3 = vld1q_f32(l02_ptr+12);
        l02_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k7_ptr);
        k0_c1 = vld1q_f32(k7_ptr+4);
        k0_c2 = vld1q_f32(k7_ptr+8);
        k0_c3 = vld1q_f32(k7_ptr+12);
        k7_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k8_ptr);
        k0_c1 = vld1q_f32(k8_ptr+4);
        k0_c2 = vld1q_f32(k8_ptr+8);
        k0_c3 = vld1q_f32(k8_ptr+12);
        k8_ptr += 16;

        l0_c0 = vld1q_f32(l12_ptr);
        l0_c1 = vld1q_f32(l12_ptr+4);
        l0_c2 = vld1q_f32(l12_ptr+8);
        l0_c3 = vld1q_f32(l12_ptr+12);
        l12_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k4_p = *k4_ptr++;
        float l01_p = *l01_ptr++;
        float k5_p = *k5_ptr++;
        float l02_p = *l02_ptr++;
        float k7_p = *k7_ptr++;
        float l11_p = *l11_ptr++;
        float k8_p = *k8_ptr++;
        float l12_p = *l12_ptr++;
        float r0_p = (k4_p * l01_p) + (k5_p * l02_p) + (k7_p * l11_p) + (k8_p * l12_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }

//top middle points
    float* k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    k5_ptr = kernel + 5 * in_c;
    float* k6_ptr = k4_ptr + 2 * in_c;
    k7_ptr = k4_ptr + 3 * in_c;
    k8_ptr = k4_ptr + 4 * in_c;

    float* l00_ptr = input + in_c;
    l01_ptr = input + 2 * in_c;
    l02_ptr = input + 3 * in_c;
    float* l10_ptr = input + in_w * in_c + in_c;
    l11_ptr = input + in_w * in_c + 2 * in_c;
    l12_ptr = input + in_w * in_c + 3 * in_c;
   
    b_ptr = bias;

    for(int lw=0; lw<(out_w-2); lw++)
    {
        lc = 0;

        for(lc=0; (lc+16)<=in_c; lc+=16)
        {
            k0_c0 = vld1q_f32(k3_ptr);
            k0_c1 = vld1q_f32(k3_ptr+4);
            k0_c2 = vld1q_f32(k3_ptr+8);
            k0_c3 = vld1q_f32(k3_ptr+12);
            k3_ptr += 16;
            
            l0_c0 = vld1q_f32(l00_ptr);
            l0_c1 = vld1q_f32(l00_ptr+4);
            l0_c2 = vld1q_f32(l00_ptr+8);
            l0_c3 = vld1q_f32(l00_ptr+12);
            l00_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k4_ptr);
            k0_c1 = vld1q_f32(k4_ptr+4);
            k0_c2 = vld1q_f32(k4_ptr+8);
            k0_c3 = vld1q_f32(k4_ptr+12);
            k4_ptr += 16;
            
            l0_c0 = vld1q_f32(l01_ptr);
            l0_c1 = vld1q_f32(l01_ptr+4);
            l0_c2 = vld1q_f32(l01_ptr+8);
            l0_c3 = vld1q_f32(l01_ptr+12);
            l01_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k5_ptr);
            k0_c1 = vld1q_f32(k5_ptr+4);
            k0_c2 = vld1q_f32(k5_ptr+8);
            k0_c3 = vld1q_f32(k5_ptr+12);
            k5_ptr += 16;
            
            l0_c0 = vld1q_f32(l02_ptr);
            l0_c1 = vld1q_f32(l02_ptr+4);
            l0_c2 = vld1q_f32(l02_ptr+8);
            l0_c3 = vld1q_f32(l02_ptr+12);
            l02_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k6_ptr);
            k0_c1 = vld1q_f32(k6_ptr+4);
            k0_c2 = vld1q_f32(k6_ptr+8);
            k0_c3 = vld1q_f32(k6_ptr+12);
            k6_ptr += 16;

            l0_c0 = vld1q_f32(l10_ptr);
            l0_c1 = vld1q_f32(l10_ptr+4);
            l0_c2 = vld1q_f32(l10_ptr+8);
            l0_c3 = vld1q_f32(l10_ptr+12);
            l10_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k7_ptr);
            k0_c1 = vld1q_f32(k7_ptr+4);
            k0_c2 = vld1q_f32(k7_ptr+8);
            k0_c3 = vld1q_f32(k7_ptr+12);
            k7_ptr += 16;

            l0_c0 = vld1q_f32(l11_ptr);
            l0_c1 = vld1q_f32(l11_ptr+4);
            l0_c2 = vld1q_f32(l11_ptr+8);
            l0_c3 = vld1q_f32(l11_ptr+12);
            l11_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k8_ptr);
            k0_c1 = vld1q_f32(k8_ptr+4);
            k0_c2 = vld1q_f32(k8_ptr+8);
            k0_c3 = vld1q_f32(k8_ptr+12);
            k8_ptr += 16;

            l0_c0 = vld1q_f32(l12_ptr);
            l0_c1 = vld1q_f32(l12_ptr+4);
            l0_c2 = vld1q_f32(l12_ptr+8);
            l0_c3 = vld1q_f32(l12_ptr+12);
            l12_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            if(b_ptr!=nullptr)
            {
                b0_c0 = vld1q_f32(b_ptr);
                b0_c1 = vld1q_f32(b_ptr+4);
                b0_c2 = vld1q_f32(b_ptr+8);
                b0_c3 = vld1q_f32(b_ptr+12);
                b_ptr += 16;

                r0_c0 = vaddq_f32(r0_c0, b0_c0);
                r0_c1 = vaddq_f32(r0_c1, b0_c1);
                r0_c2 = vaddq_f32(r0_c2, b0_c2);
                r0_c3 = vaddq_f32(r0_c3, b0_c3);
            }
            //
            if(act>=0)
            {
                r0_c0 = vmaxq_f32(r0_c0, v_actz);
                r0_c1 = vmaxq_f32(r0_c1, v_actz);
                r0_c2 = vmaxq_f32(r0_c2, v_actz);
                r0_c3 = vmaxq_f32(r0_c3, v_actz);
                if(act>0)
                {
                    r0_c0 = vminq_f32(r0_c0, v_actx);
                    r0_c1 = vminq_f32(r0_c1, v_actx);
                    r0_c2 = vminq_f32(r0_c2, v_actx);
                    r0_c3 = vminq_f32(r0_c3, v_actx);
                }
            }
            //
            vst1q_f32(out, r0_c0);
            vst1q_f32(out+4, r0_c1);
            vst1q_f32(out+8, r0_c2);
            vst1q_f32(out+12, r0_c3);
            out += 16;
            //
            r0_c0 = vdupq_n_f32(0);
            r0_c1 = vdupq_n_f32(0);
            r0_c2 = vdupq_n_f32(0);
            r0_c3 = vdupq_n_f32(0);
        }
        for( ; lc<in_c; lc++)
        {
            float k3_p = *k3_ptr++;
            float l00_p = *l00_ptr++;
            float k4_p = *k4_ptr++;
            float l01_p = *l01_ptr++;
            float k5_p = *k5_ptr++;
            float l02_p = *l02_ptr++;
            float k6_p = *k6_ptr++;
            float l10_p = *l10_ptr++;
            float k7_p = *k7_ptr++;
            float l11_p = *l11_ptr++;
            float k8_p = *k8_ptr++;
            float l12_p = *l12_ptr++;
            float r0_p = (k3_p * l00_p) + (k4_p * l01_p) + (k5_p * l02_p) + (k6_p * l10_p) + (k7_p * l11_p) + (k8_p * l12_p);
            if(b_ptr!=nullptr)
            {
                r0_p = r0_p + *b_ptr++;
            }
            if(act>=0)
            {
                r0_p = std::max(r0_p, 0.f);
                if(act>0)
                    r0_p = std::min(r0_p, (float)act);
            }
            *out++ = r0_p;
        }

        k3_ptr = kernel + 3 * in_c;
        k4_ptr = kernel + 4 * in_c;
        k5_ptr = kernel + 5 * in_c;
        k6_ptr = k4_ptr + 2 * in_c;
        k7_ptr = k4_ptr + 3 * in_c;
        k8_ptr = k4_ptr + 4 * in_c;
   
        l00_ptr = l01_ptr;
        l01_ptr = l02_ptr;
        l02_ptr += in_c;
        l10_ptr = l11_ptr;
        l11_ptr = l12_ptr;
        l12_ptr += in_c;

        b_ptr = bias;
    }

//top right point
    k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    k6_ptr = k4_ptr + 2 * in_c;
    k7_ptr = k4_ptr + 3 * in_c;
    
    lc = 0;

    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k3_ptr);
        k0_c1 = vld1q_f32(k3_ptr+4);
        k0_c2 = vld1q_f32(k3_ptr+8);
        k0_c3 = vld1q_f32(k3_ptr+12);
        k3_ptr += 16;
        
        l0_c0 = vld1q_f32(l00_ptr);
        l0_c1 = vld1q_f32(l00_ptr+4);
        l0_c2 = vld1q_f32(l00_ptr+8);
        l0_c3 = vld1q_f32(l00_ptr+12);
        l00_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k6_ptr);
        k0_c1 = vld1q_f32(k6_ptr+4);
        k0_c2 = vld1q_f32(k6_ptr+8);
        k0_c3 = vld1q_f32(k6_ptr+12);
        k6_ptr += 16;

        l0_c0 = vld1q_f32(l10_ptr);
        l0_c1 = vld1q_f32(l10_ptr+4);
        l0_c2 = vld1q_f32(l10_ptr+8);
        l0_c3 = vld1q_f32(l10_ptr+12);
        l10_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k7_ptr);
        k0_c1 = vld1q_f32(k7_ptr+4);
        k0_c2 = vld1q_f32(k7_ptr+8);
        k0_c3 = vld1q_f32(k7_ptr+12);
        k7_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k3_p = *k3_ptr++;
        float l00_p = *l00_ptr++;
        float k4_p = *k4_ptr++;
        float l01_p = *l01_ptr++;
        float k6_p = *k6_ptr++;
        float l10_p = *l10_ptr++;
        float k7_p = *k7_ptr++;
        float l11_p = *l11_ptr++;
        float r0_p = (k3_p * l00_p) + (k4_p * l01_p) + (k6_p * l10_p) + (k7_p * l11_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }
}






static void k3s2p1_nhwc_fp32_hetc(float* input, float* kernel, float* out, float* bias, long act, long in_w, long in_h, long in_c, long out_w, long out_h)
{
    float* k1_ptr = kernel + in_c;
    float* k2_ptr = kernel + 2 * in_c;
    float* k4_ptr = k2_ptr + 2 * in_c;
    float* k5_ptr = k2_ptr + 3 * in_c;

    float* l01_ptr = input;
    float* l02_ptr = input + in_c;
    float* l11_ptr = input + in_w * in_c;
    float* l12_ptr = input + in_w * in_c + in_c;
    
    float* b_ptr = bias;

    int lc = 0;

    float32x4_t k0_c0, k0_c1, k0_c2, k0_c3;
    float32x4_t l0_c0, l0_c1, l0_c2, l0_c3;
    float32x4_t b0_c0, b0_c1, b0_c2, b0_c3;
    float32x4_t r0_c0, r0_c1, r0_c2, r0_c3;

    float32x4_t v_actz = vdupq_n_f32(0);
    float32x4_t v_actx = vdupq_n_f32(act);

    r0_c0 = vdupq_n_f32(0);
    r0_c1 = vdupq_n_f32(0);
    r0_c2 = vdupq_n_f32(0);
    r0_c3 = vdupq_n_f32(0);

//bottom left point
    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k2_ptr);
        k0_c1 = vld1q_f32(k2_ptr+4);
        k0_c2 = vld1q_f32(k2_ptr+8);
        k0_c3 = vld1q_f32(k2_ptr+12);
        k2_ptr += 16;
        
        l0_c0 = vld1q_f32(l02_ptr);
        l0_c1 = vld1q_f32(l02_ptr+4);
        l0_c2 = vld1q_f32(l02_ptr+8);
        l0_c3 = vld1q_f32(l02_ptr+12);
        l02_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k5_ptr);
        k0_c1 = vld1q_f32(k5_ptr+4);
        k0_c2 = vld1q_f32(k5_ptr+8);
        k0_c3 = vld1q_f32(k5_ptr+12);
        k5_ptr += 16;

        l0_c0 = vld1q_f32(l12_ptr);
        l0_c1 = vld1q_f32(l12_ptr+4);
        l0_c2 = vld1q_f32(l12_ptr+8);
        l0_c3 = vld1q_f32(l12_ptr+12);
        l12_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k2_p = *k2_ptr++;
        float l02_p = *l02_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float k5_p = *k5_ptr++;
        float l12_p = *l12_ptr++;
        float r0_p = (k1_p * l01_p) + (k2_p * l02_p) + (k4_p * l11_p) + (k5_p * l12_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }

//bottom middle points
    float* k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k2_ptr = kernel + 2 * in_c;
    float* k3_ptr = k2_ptr + in_c;
    k4_ptr = k2_ptr + 2 * in_c;
    k5_ptr = k2_ptr + 3 * in_c;

    float* l00_ptr = input + in_c;
    l01_ptr = input + 2 * in_c;
    l02_ptr = input + 3 * in_c;
    float* l10_ptr = input + in_w * in_c + in_c;
    l11_ptr = input + in_w * in_c + 2 * in_c;
    l12_ptr = input + in_w * in_c + 3 * in_c;
   
    b_ptr = bias;

    for(int lw=0; lw<(out_w-2); lw++)
    {
        lc = 0;

        for(lc=0; (lc+16)<=in_c; lc+=16)
        {
            k0_c0 = vld1q_f32(k0_ptr);
            k0_c1 = vld1q_f32(k0_ptr+4);
            k0_c2 = vld1q_f32(k0_ptr+8);
            k0_c3 = vld1q_f32(k0_ptr+12);
            k0_ptr += 16;
            
            l0_c0 = vld1q_f32(l00_ptr);
            l0_c1 = vld1q_f32(l00_ptr+4);
            l0_c2 = vld1q_f32(l00_ptr+8);
            l0_c3 = vld1q_f32(l00_ptr+12);
            l00_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k1_ptr);
            k0_c1 = vld1q_f32(k1_ptr+4);
            k0_c2 = vld1q_f32(k1_ptr+8);
            k0_c3 = vld1q_f32(k1_ptr+12);
            k1_ptr += 16;
            
            l0_c0 = vld1q_f32(l01_ptr);
            l0_c1 = vld1q_f32(l01_ptr+4);
            l0_c2 = vld1q_f32(l01_ptr+8);
            l0_c3 = vld1q_f32(l01_ptr+12);
            l01_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k2_ptr);
            k0_c1 = vld1q_f32(k2_ptr+4);
            k0_c2 = vld1q_f32(k2_ptr+8);
            k0_c3 = vld1q_f32(k2_ptr+12);
            k2_ptr += 16;
            
            l0_c0 = vld1q_f32(l02_ptr);
            l0_c1 = vld1q_f32(l02_ptr+4);
            l0_c2 = vld1q_f32(l02_ptr+8);
            l0_c3 = vld1q_f32(l02_ptr+12);
            l02_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k3_ptr);
            k0_c1 = vld1q_f32(k3_ptr+4);
            k0_c2 = vld1q_f32(k3_ptr+8);
            k0_c3 = vld1q_f32(k3_ptr+12);
            k3_ptr += 16;

            l0_c0 = vld1q_f32(l10_ptr);
            l0_c1 = vld1q_f32(l10_ptr+4);
            l0_c2 = vld1q_f32(l10_ptr+8);
            l0_c3 = vld1q_f32(l10_ptr+12);
            l10_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k4_ptr);
            k0_c1 = vld1q_f32(k4_ptr+4);
            k0_c2 = vld1q_f32(k4_ptr+8);
            k0_c3 = vld1q_f32(k4_ptr+12);
            k4_ptr += 16;

            l0_c0 = vld1q_f32(l11_ptr);
            l0_c1 = vld1q_f32(l11_ptr+4);
            l0_c2 = vld1q_f32(l11_ptr+8);
            l0_c3 = vld1q_f32(l11_ptr+12);
            l11_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k5_ptr);
            k0_c1 = vld1q_f32(k5_ptr+4);
            k0_c2 = vld1q_f32(k5_ptr+8);
            k0_c3 = vld1q_f32(k5_ptr+12);
            k5_ptr += 16;

            l0_c0 = vld1q_f32(l12_ptr);
            l0_c1 = vld1q_f32(l12_ptr+4);
            l0_c2 = vld1q_f32(l12_ptr+8);
            l0_c3 = vld1q_f32(l12_ptr+12);
            l12_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            if(b_ptr!=nullptr)
            {
                b0_c0 = vld1q_f32(b_ptr);
                b0_c1 = vld1q_f32(b_ptr+4);
                b0_c2 = vld1q_f32(b_ptr+8);
                b0_c3 = vld1q_f32(b_ptr+12);
                b_ptr += 16;

                r0_c0 = vaddq_f32(r0_c0, b0_c0);
                r0_c1 = vaddq_f32(r0_c1, b0_c1);
                r0_c2 = vaddq_f32(r0_c2, b0_c2);
                r0_c3 = vaddq_f32(r0_c3, b0_c3);
            }
            //
            if(act>=0)
            {
                r0_c0 = vmaxq_f32(r0_c0, v_actz);
                r0_c1 = vmaxq_f32(r0_c1, v_actz);
                r0_c2 = vmaxq_f32(r0_c2, v_actz);
                r0_c3 = vmaxq_f32(r0_c3, v_actz);
                if(act>0)
                {
                    r0_c0 = vminq_f32(r0_c0, v_actx);
                    r0_c1 = vminq_f32(r0_c1, v_actx);
                    r0_c2 = vminq_f32(r0_c2, v_actx);
                    r0_c3 = vminq_f32(r0_c3, v_actx);
                }
            }
            //
            vst1q_f32(out, r0_c0);
            vst1q_f32(out+4, r0_c1);
            vst1q_f32(out+8, r0_c2);
            vst1q_f32(out+12, r0_c3);
            out += 16;
            //
            r0_c0 = vdupq_n_f32(0);
            r0_c1 = vdupq_n_f32(0);
            r0_c2 = vdupq_n_f32(0);
            r0_c3 = vdupq_n_f32(0);
        }
        for( ; lc<in_c; lc++)
        {
            float k0_p = *k0_ptr++;
            float l00_p = *l00_ptr++;
            float k1_p = *k1_ptr++;
            float l01_p = *l01_ptr++;
            float k2_p = *k2_ptr++;
            float l02_p = *l02_ptr++;
            float k3_p = *k3_ptr++;
            float l10_p = *l10_ptr++;
            float k4_p = *k4_ptr++;
            float l11_p = *l11_ptr++;
            float k5_p = *k5_ptr++;
            float l12_p = *l12_ptr++;
            float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k2_p * l02_p) + (k3_p * l10_p) + (k4_p * l11_p) + (k5_p * l12_p);
            if(b_ptr!=nullptr)
            {
                r0_p = r0_p + *b_ptr++;
            }
            if(act>=0)
            {
                r0_p = std::max(r0_p, 0.f);
                if(act>0)
                    r0_p = std::min(r0_p, (float)act);
            }
            *out++ = r0_p;
        }

        k0_ptr = kernel;
        k1_ptr = kernel + in_c;
        k2_ptr = kernel + 2 * in_c;
        k3_ptr = k2_ptr + 1 * in_c;
        k4_ptr = k2_ptr + 2 * in_c;
        k5_ptr = k2_ptr + 3 * in_c;
   
        l00_ptr = l01_ptr;
        l01_ptr = l02_ptr;
        l02_ptr += in_c;
        l10_ptr = l11_ptr;
        l11_ptr = l12_ptr;
        l12_ptr += in_c;

        b_ptr = bias;
    }

//bottom right point
    k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    
    lc = 0;

    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k0_ptr);
        k0_c1 = vld1q_f32(k0_ptr+4);
        k0_c2 = vld1q_f32(k0_ptr+8);
        k0_c3 = vld1q_f32(k0_ptr+12);
        k0_ptr += 16;
        
        l0_c0 = vld1q_f32(l00_ptr);
        l0_c1 = vld1q_f32(l00_ptr+4);
        l0_c2 = vld1q_f32(l00_ptr+8);
        l0_c3 = vld1q_f32(l00_ptr+12);
        l00_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k3_ptr);
        k0_c1 = vld1q_f32(k3_ptr+4);
        k0_c2 = vld1q_f32(k3_ptr+8);
        k0_c3 = vld1q_f32(k3_ptr+12);
        k3_ptr += 16;

        l0_c0 = vld1q_f32(l10_ptr);
        l0_c1 = vld1q_f32(l10_ptr+4);
        l0_c2 = vld1q_f32(l10_ptr+8);
        l0_c3 = vld1q_f32(l10_ptr+12);
        l10_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k0_p = *k0_ptr++;
        float l00_p = *l00_ptr++;
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k3_p = *k3_ptr++;
        float l10_p = *l10_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k3_p * l10_p) + (k4_p * l11_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }
}







static void k3s2p1_nhwc_fp32_hmtc(float* input, float* kernel, float* out, float* bias, long act, long in_w, long in_h, long in_c, long out_w, long out_h)
{
    float* k1_ptr = kernel + in_c;
    float* k2_ptr = kernel + 2 * in_c;
    float* k4_ptr = k2_ptr + 2 * in_c;
    float* k5_ptr = k2_ptr + 3 * in_c;
    float* k7_ptr = k5_ptr + 2 * in_c;
    float* k8_ptr = k5_ptr + 3 * in_c;

    float* l01_ptr = input;
    float* l02_ptr = input + in_c;
    float* l11_ptr = input + in_w * in_c;
    float* l12_ptr = input + in_w * in_c + in_c;
    float* l21_ptr = input + 2 * in_w * in_c;
    float* l22_ptr = input + 2 * in_w * in_c + in_c;
    
    float* b_ptr = bias;

    int lc = 0;

    float32x4_t k0_c0, k0_c1, k0_c2, k0_c3;
    float32x4_t l0_c0, l0_c1, l0_c2, l0_c3;
    float32x4_t b0_c0, b0_c1, b0_c2, b0_c3;
    float32x4_t r0_c0, r0_c1, r0_c2, r0_c3;

    float32x4_t v_actz = vdupq_n_f32(0);
    float32x4_t v_actx = vdupq_n_f32(act);

    r0_c0 = vdupq_n_f32(0);
    r0_c1 = vdupq_n_f32(0);
    r0_c2 = vdupq_n_f32(0);
    r0_c3 = vdupq_n_f32(0);

//middle left point
    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k2_ptr);
        k0_c1 = vld1q_f32(k2_ptr+4);
        k0_c2 = vld1q_f32(k2_ptr+8);
        k0_c3 = vld1q_f32(k2_ptr+12);
        k2_ptr += 16;
        
        l0_c0 = vld1q_f32(l02_ptr);
        l0_c1 = vld1q_f32(l02_ptr+4);
        l0_c2 = vld1q_f32(l02_ptr+8);
        l0_c3 = vld1q_f32(l02_ptr+12);
        l02_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k5_ptr);
        k0_c1 = vld1q_f32(k5_ptr+4);
        k0_c2 = vld1q_f32(k5_ptr+8);
        k0_c3 = vld1q_f32(k5_ptr+12);
        k5_ptr += 16;

        l0_c0 = vld1q_f32(l12_ptr);
        l0_c1 = vld1q_f32(l12_ptr+4);
        l0_c2 = vld1q_f32(l12_ptr+8);
        l0_c3 = vld1q_f32(l12_ptr+12);
        l12_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k7_ptr);
        k0_c1 = vld1q_f32(k7_ptr+4);
        k0_c2 = vld1q_f32(k7_ptr+8);
        k0_c3 = vld1q_f32(k7_ptr+12);
        k7_ptr += 16;

        l0_c0 = vld1q_f32(l21_ptr);
        l0_c1 = vld1q_f32(l21_ptr+4);
        l0_c2 = vld1q_f32(l21_ptr+8);
        l0_c3 = vld1q_f32(l21_ptr+12);
        l21_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k8_ptr);
        k0_c1 = vld1q_f32(k8_ptr+4);
        k0_c2 = vld1q_f32(k8_ptr+8);
        k0_c3 = vld1q_f32(k8_ptr+12);
        k8_ptr += 16;

        l0_c0 = vld1q_f32(l22_ptr);
        l0_c1 = vld1q_f32(l22_ptr+4);
        l0_c2 = vld1q_f32(l22_ptr+8);
        l0_c3 = vld1q_f32(l22_ptr+12);
        l22_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        { 
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k2_p = *k2_ptr++;
        float l02_p = *l02_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float k5_p = *k5_ptr++;
        float l12_p = *l12_ptr++;
        float k7_p = *k7_ptr++;
        float l21_p = *l21_ptr++;
        float k8_p = *k8_ptr++;
        float l22_p = *l22_ptr++;
        float r0_p = (k1_p * l01_p) + (k2_p * l02_p) + (k4_p * l11_p) + (k5_p * l12_p) + (k7_p * l21_p) + (k8_p * l22_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }

//middle middle points
    float* k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k2_ptr = kernel + 2 * in_c;
    float* k3_ptr = k2_ptr + in_c;
    k4_ptr = k2_ptr + 2 * in_c;
    k5_ptr = k2_ptr + 3 * in_c;
    float* k6_ptr = k5_ptr + in_c;
    k7_ptr = k5_ptr + 2 * in_c;
    k8_ptr = k5_ptr + 3 * in_c;

    float* l00_ptr = input + in_c;
    l01_ptr = input + 2 * in_c;
    l02_ptr = input + 3 * in_c;
    float* l10_ptr = input + in_w * in_c + in_c;
    l11_ptr = input + in_w * in_c + 2 * in_c;
    l12_ptr = input + in_w * in_c + 3 * in_c;
    float* l20_ptr = input + 2 * in_w * in_c + in_c;
    l21_ptr = input + 2 * in_w * in_c + 2 * in_c;
    l22_ptr = input + 2 * in_w * in_c + 3 * in_c;
   
    b_ptr = bias;

    for(int lw=0; lw<(out_w-2); lw++)
    {
        lc = 0;

        for(lc=0; (lc+16)<=in_c; lc+=16)
        {
            k0_c0 = vld1q_f32(k0_ptr);
            k0_c1 = vld1q_f32(k0_ptr+4);
            k0_c2 = vld1q_f32(k0_ptr+8);
            k0_c3 = vld1q_f32(k0_ptr+12);
            k0_ptr += 16;
            
            l0_c0 = vld1q_f32(l00_ptr);
            l0_c1 = vld1q_f32(l00_ptr+4);
            l0_c2 = vld1q_f32(l00_ptr+8);
            l0_c3 = vld1q_f32(l00_ptr+12);
            l00_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k1_ptr);
            k0_c1 = vld1q_f32(k1_ptr+4);
            k0_c2 = vld1q_f32(k1_ptr+8);
            k0_c3 = vld1q_f32(k1_ptr+12);
            k1_ptr += 16;
            
            l0_c0 = vld1q_f32(l01_ptr);
            l0_c1 = vld1q_f32(l01_ptr+4);
            l0_c2 = vld1q_f32(l01_ptr+8);
            l0_c3 = vld1q_f32(l01_ptr+12);
            l01_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k2_ptr);
            k0_c1 = vld1q_f32(k2_ptr+4);
            k0_c2 = vld1q_f32(k2_ptr+8);
            k0_c3 = vld1q_f32(k2_ptr+12);
            k2_ptr += 16;
            
            l0_c0 = vld1q_f32(l02_ptr);
            l0_c1 = vld1q_f32(l02_ptr+4);
            l0_c2 = vld1q_f32(l02_ptr+8);
            l0_c3 = vld1q_f32(l02_ptr+12);
            l02_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k3_ptr);
            k0_c1 = vld1q_f32(k3_ptr+4);
            k0_c2 = vld1q_f32(k3_ptr+8);
            k0_c3 = vld1q_f32(k3_ptr+12);
            k3_ptr += 16;

            l0_c0 = vld1q_f32(l10_ptr);
            l0_c1 = vld1q_f32(l10_ptr+4);
            l0_c2 = vld1q_f32(l10_ptr+8);
            l0_c3 = vld1q_f32(l10_ptr+12);
            l10_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k4_ptr);
            k0_c1 = vld1q_f32(k4_ptr+4);
            k0_c2 = vld1q_f32(k4_ptr+8);
            k0_c3 = vld1q_f32(k4_ptr+12);
            k4_ptr += 16;

            l0_c0 = vld1q_f32(l11_ptr);
            l0_c1 = vld1q_f32(l11_ptr+4);
            l0_c2 = vld1q_f32(l11_ptr+8);
            l0_c3 = vld1q_f32(l11_ptr+12);
            l11_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k5_ptr);
            k0_c1 = vld1q_f32(k5_ptr+4);
            k0_c2 = vld1q_f32(k5_ptr+8);
            k0_c3 = vld1q_f32(k5_ptr+12);
            k5_ptr += 16;

            l0_c0 = vld1q_f32(l12_ptr);
            l0_c1 = vld1q_f32(l12_ptr+4);
            l0_c2 = vld1q_f32(l12_ptr+8);
            l0_c3 = vld1q_f32(l12_ptr+12);
            l12_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k6_ptr);
            k0_c1 = vld1q_f32(k6_ptr+4);
            k0_c2 = vld1q_f32(k6_ptr+8);
            k0_c3 = vld1q_f32(k6_ptr+12);
            k6_ptr += 16;

            l0_c0 = vld1q_f32(l20_ptr);
            l0_c1 = vld1q_f32(l20_ptr+4);
            l0_c2 = vld1q_f32(l20_ptr+8);
            l0_c3 = vld1q_f32(l20_ptr+12);
            l20_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k7_ptr);
            k0_c1 = vld1q_f32(k7_ptr+4);
            k0_c2 = vld1q_f32(k7_ptr+8);
            k0_c3 = vld1q_f32(k7_ptr+12);
            k7_ptr += 16;

            l0_c0 = vld1q_f32(l21_ptr);
            l0_c1 = vld1q_f32(l21_ptr+4);
            l0_c2 = vld1q_f32(l21_ptr+8);
            l0_c3 = vld1q_f32(l21_ptr+12);
            l21_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            k0_c0 = vld1q_f32(k8_ptr);
            k0_c1 = vld1q_f32(k8_ptr+4);
            k0_c2 = vld1q_f32(k8_ptr+8);
            k0_c3 = vld1q_f32(k8_ptr+12);
            k8_ptr += 16;

            l0_c0 = vld1q_f32(l22_ptr);
            l0_c1 = vld1q_f32(l22_ptr+4);
            l0_c2 = vld1q_f32(l22_ptr+8);
            l0_c3 = vld1q_f32(l22_ptr+12);
            l22_ptr += 16;

            r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
            r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
            r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
            r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
            //
            if(b_ptr!=nullptr)
            {
                b0_c0 = vld1q_f32(b_ptr);
                b0_c1 = vld1q_f32(b_ptr+4);
                b0_c2 = vld1q_f32(b_ptr+8);
                b0_c3 = vld1q_f32(b_ptr+12);
                b_ptr += 16;

                r0_c0 = vaddq_f32(r0_c0, b0_c0);
                r0_c1 = vaddq_f32(r0_c1, b0_c1);
                r0_c2 = vaddq_f32(r0_c2, b0_c2);
                r0_c3 = vaddq_f32(r0_c3, b0_c3);
            }
            //
            if(act>=0)
            {
                r0_c0 = vmaxq_f32(r0_c0, v_actz);
                r0_c1 = vmaxq_f32(r0_c1, v_actz);
                r0_c2 = vmaxq_f32(r0_c2, v_actz);
                r0_c3 = vmaxq_f32(r0_c3, v_actz);
                if(act>0)
                {
                    r0_c0 = vminq_f32(r0_c0, v_actx);
                    r0_c1 = vminq_f32(r0_c1, v_actx);
                    r0_c2 = vminq_f32(r0_c2, v_actx);
                    r0_c3 = vminq_f32(r0_c3, v_actx);
                }
            }
            //
            vst1q_f32(out, r0_c0);
            vst1q_f32(out+4, r0_c1);
            vst1q_f32(out+8, r0_c2);
            vst1q_f32(out+12, r0_c3);
            out += 16;
            //
            r0_c0 = vdupq_n_f32(0);
            r0_c1 = vdupq_n_f32(0);
            r0_c2 = vdupq_n_f32(0);
            r0_c3 = vdupq_n_f32(0);
        }
        for( ; lc<in_c; lc++)
        {
            float k0_p = *k0_ptr++;
            float l00_p = *l00_ptr++;
            float k1_p = *k1_ptr++;
            float l01_p = *l01_ptr++;
            float k2_p = *k2_ptr++;
            float l02_p = *l02_ptr++;
            float k3_p = *k3_ptr++;
            float l10_p = *l10_ptr++;
            float k4_p = *k4_ptr++;
            float l11_p = *l11_ptr++;
            float k5_p = *k5_ptr++;
            float l12_p = *l12_ptr++;
            float k6_p = *k6_ptr++;
            float l20_p = *l20_ptr++;
            float k7_p = *k7_ptr++;
            float l21_p = *l21_ptr++;
            float k8_p = *k8_ptr++;
            float l22_p = *l22_ptr++;
            float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k2_p * l02_p) + (k3_p * l10_p) + (k4_p * l11_p) + (k5_p * l12_p) + (k6_p * l20_p) + (k7_p * l21_p) + (k8_p * l22_p);
            if(b_ptr!=nullptr)
            {
                r0_p = r0_p + *b_ptr++;
            }
            if(act>=0)
            {
                r0_p = std::max(r0_p, 0.f);
                if(act>0)
                    r0_p = std::min(r0_p, (float)act);
            }
            *out++ = r0_p;
        }

        k0_ptr = kernel;
        k1_ptr = kernel + in_c;
        k2_ptr = kernel + 2 * in_c;
        k3_ptr = k2_ptr + 1 * in_c;
        k4_ptr = k2_ptr + 2 * in_c;
        k5_ptr = k2_ptr + 3 * in_c;
        k6_ptr = k5_ptr + 1 * in_c;
        k7_ptr = k5_ptr + 2 * in_c;
        k8_ptr = k5_ptr + 3 * in_c;
   
        l00_ptr = l01_ptr; 
        l01_ptr = l02_ptr;
        l02_ptr += in_c; 
        l10_ptr = l11_ptr;
        l11_ptr = l12_ptr;
        l12_ptr += in_c;
        l20_ptr = l21_ptr;
        l21_ptr = l22_ptr;
        l22_ptr += in_c;

        b_ptr = bias;
    }

//middle right point
    k0_ptr = kernel;
    k1_ptr = kernel + in_c;
    k3_ptr = kernel + 3 * in_c;
    k4_ptr = kernel + 4 * in_c;
    k6_ptr = kernel + 6 * in_c;
    k7_ptr = kernel + 7 * in_c;
    
    lc = 0;

    for(lc=0; (lc+16)<=in_c; lc+=16)
    {
        k0_c0 = vld1q_f32(k0_ptr);
        k0_c1 = vld1q_f32(k0_ptr+4);
        k0_c2 = vld1q_f32(k0_ptr+8);
        k0_c3 = vld1q_f32(k0_ptr+12);
        k0_ptr += 16;
        
        l0_c0 = vld1q_f32(l00_ptr);
        l0_c1 = vld1q_f32(l00_ptr+4);
        l0_c2 = vld1q_f32(l00_ptr+8);
        l0_c3 = vld1q_f32(l00_ptr+12);
        l00_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k1_ptr);
        k0_c1 = vld1q_f32(k1_ptr+4);
        k0_c2 = vld1q_f32(k1_ptr+8);
        k0_c3 = vld1q_f32(k1_ptr+12);
        k1_ptr += 16;
        
        l0_c0 = vld1q_f32(l01_ptr);
        l0_c1 = vld1q_f32(l01_ptr+4);
        l0_c2 = vld1q_f32(l01_ptr+8);
        l0_c3 = vld1q_f32(l01_ptr+12);
        l01_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k3_ptr);
        k0_c1 = vld1q_f32(k3_ptr+4);
        k0_c2 = vld1q_f32(k3_ptr+8);
        k0_c3 = vld1q_f32(k3_ptr+12);
        k3_ptr += 16;

        l0_c0 = vld1q_f32(l10_ptr);
        l0_c1 = vld1q_f32(l10_ptr+4);
        l0_c2 = vld1q_f32(l10_ptr+8);
        l0_c3 = vld1q_f32(l10_ptr+12);
        l10_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k4_ptr);
        k0_c1 = vld1q_f32(k4_ptr+4);
        k0_c2 = vld1q_f32(k4_ptr+8);
        k0_c3 = vld1q_f32(k4_ptr+12);
        k4_ptr += 16;

        l0_c0 = vld1q_f32(l11_ptr);
        l0_c1 = vld1q_f32(l11_ptr+4);
        l0_c2 = vld1q_f32(l11_ptr+8);
        l0_c3 = vld1q_f32(l11_ptr+12);
        l11_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k6_ptr);
        k0_c1 = vld1q_f32(k6_ptr+4);
        k0_c2 = vld1q_f32(k6_ptr+8);
        k0_c3 = vld1q_f32(k6_ptr+12);
        k6_ptr += 16;

        l0_c0 = vld1q_f32(l20_ptr);
        l0_c1 = vld1q_f32(l20_ptr+4);
        l0_c2 = vld1q_f32(l20_ptr+8);
        l0_c3 = vld1q_f32(l20_ptr+12);
        l20_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        k0_c0 = vld1q_f32(k7_ptr);
        k0_c1 = vld1q_f32(k7_ptr+4);
        k0_c2 = vld1q_f32(k7_ptr+8);
        k0_c3 = vld1q_f32(k7_ptr+12);
        k7_ptr += 16;

        l0_c0 = vld1q_f32(l21_ptr);
        l0_c1 = vld1q_f32(l21_ptr+4);
        l0_c2 = vld1q_f32(l21_ptr+8);
        l0_c3 = vld1q_f32(l21_ptr+12);
        l21_ptr += 16;

        r0_c0 = vmlaq_f32(r0_c0, k0_c0, l0_c0);
        r0_c1 = vmlaq_f32(r0_c1, k0_c1, l0_c1);
        r0_c2 = vmlaq_f32(r0_c2, k0_c2, l0_c2);
        r0_c3 = vmlaq_f32(r0_c3, k0_c3, l0_c3);
        //
        if(b_ptr!=nullptr)
        {
            b0_c0 = vld1q_f32(b_ptr);
            b0_c1 = vld1q_f32(b_ptr+4);
            b0_c2 = vld1q_f32(b_ptr+8);
            b0_c3 = vld1q_f32(b_ptr+12);
            b_ptr += 16;

            r0_c0 = vaddq_f32(r0_c0, b0_c0);
            r0_c1 = vaddq_f32(r0_c1, b0_c1);
            r0_c2 = vaddq_f32(r0_c2, b0_c2);
            r0_c3 = vaddq_f32(r0_c3, b0_c3);
        }
        //
        if(act>=0)
        {
            r0_c0 = vmaxq_f32(r0_c0, v_actz);
            r0_c1 = vmaxq_f32(r0_c1, v_actz);
            r0_c2 = vmaxq_f32(r0_c2, v_actz);
            r0_c3 = vmaxq_f32(r0_c3, v_actz);
            if(act>0)
            {
                r0_c0 = vminq_f32(r0_c0, v_actx);
                r0_c1 = vminq_f32(r0_c1, v_actx);
                r0_c2 = vminq_f32(r0_c2, v_actx);
                r0_c3 = vminq_f32(r0_c3, v_actx);
            }
        }
        //
        vst1q_f32(out, r0_c0);
        vst1q_f32(out+4, r0_c1);
        vst1q_f32(out+8, r0_c2);
        vst1q_f32(out+12, r0_c3);
        out += 16;
        //
        r0_c0 = vdupq_n_f32(0);
        r0_c1 = vdupq_n_f32(0);
        r0_c2 = vdupq_n_f32(0);
        r0_c3 = vdupq_n_f32(0);
    }
    for( ; lc<in_c; lc++)
    {
        float k0_p = *k0_ptr++;
        float l00_p = *l00_ptr++;
        float k1_p = *k1_ptr++;
        float l01_p = *l01_ptr++;
        float k3_p = *k3_ptr++;
        float l10_p = *l10_ptr++;
        float k4_p = *k4_ptr++;
        float l11_p = *l11_ptr++;
        float k6_p = *k6_ptr++;
        float l20_p = *l20_ptr++;
        float k7_p = *k7_ptr++;
        float l21_p = *l21_ptr++;
        float r0_p = (k0_p * l00_p) + (k1_p * l01_p) + (k3_p * l10_p) + (k4_p * l11_p) + (k6_p * l20_p) + (k7_p * l21_p);
        if(b_ptr!=nullptr)
        {
            r0_p = r0_p + *b_ptr++;
        }
        if(act>=0)
        {
            r0_p = std::max(r0_p, 0.f);
            if(act>0)
                r0_p = std::min(r0_p, (float)act);
        }
        *out++ = r0_p;
    }
}
//for kernel k3s2p1 end






