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


#include <iostream>

#include "logger.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "node_ops.hpp"
#include "operator/deconvolution.hpp"
#include <arm_neon.h>


#ifdef CONFIG_AUTH_DEVICE
#include "auth_nodeops.hpp"
#endif

#define DECONV_DW_MAX_(a, b) ((a) > (b) ? (a) : (b))
#define DECONV_DW_MIN_(a, b) ((a) < (b) ? (a) : (b))


#include <math.h>
namespace TEngine {

namespace deconv_2d_dw_4x4 {

inline float do_activation(float input, int activation)
{
    if(activation == 0)
    {
        input = DECONV_DW_MAX_(input, 0);
        if(activation == 6)
            input =DECONV_DW_MIN_(input, 6);
    }
    return input;
}
const char* conv_name = "DECONV_DW";
const int default_prio = 98;

void deconv_4x4s2(float* input, 
                float* kernel,
                float* output, 
                int group_start,
                int group_end,int cg,
                int inh, int inw, 
                int outh,int outw,
                int activation,int pad)
{
    int in_hw = inw * inh;
    int out_hw = outh * outw;
    if(pad==1)
    {
        for(int g= group_start; g < group_end; g++)
        {
            float* out_ptr = output + g * out_hw;
            for(int c=0;c< cg; c++)
            {
                int group_idx = (g*cg + c);
                float* ker_ptr = kernel + group_idx*16;
                float32x4_t w0 = vld1q_f32(ker_ptr);
                float32x4_t w1 = vld1q_f32(ker_ptr+4);
                float32x4_t w2 = vld1q_f32(ker_ptr+8);
                float32x4_t w3 = vld1q_f32(ker_ptr+12);

                float* inp_ptr = input + group_idx* in_hw;
                //h-begin
                int h=0;
                {
                    float* out1 = out_ptr;
                    float* out2 = out1 + outw;
                    float* out3 = out2 + outw;
                    
                    int w=0;
                    //w-begin [0,1,2,3] =========================
                    {
                        float32x4_t inp = vld1q_f32(inp_ptr);
                        // printf("h begin | w begin\n");
                        out1[0]+= inp_ptr[0]*ker_ptr[5];
                        out1[1]+= inp_ptr[0]*ker_ptr[6];
                        out1[2]+= inp_ptr[0]*ker_ptr[7];
                        out2[0]+= inp_ptr[0]*ker_ptr[9];
                        out2[1]+= inp_ptr[0]*ker_ptr[10];
                        out2[2]+= inp_ptr[0]*ker_ptr[11];
                        out3[0]+= inp_ptr[0]*ker_ptr[13];
                        out3[1]+= inp_ptr[0]*ker_ptr[14];
                        out3[2]+= inp_ptr[0]*ker_ptr[15];
                        float32x4_t v11 = vld1q_f32(out1+1);
                        float32x4_t v21 = vld1q_f32(out2+1);
                        float32x4_t v31 = vld1q_f32(out3+1);
                        v11 = vmlaq_lane_f32(v11, w1, vget_low_f32(inp), 1);
                        v21 = vmlaq_lane_f32(v21, w2, vget_low_f32(inp), 1);
                        v31 = vmlaq_lane_f32(v31, w3, vget_low_f32(inp), 1);
                        vst1q_f32(out1+1, v11);
                        vst1q_f32(out2+1, v21);
                        vst1q_f32(out3+1, v31);

                        float32x4_t v12 = vld1q_f32(out1+3);
                        float32x4_t v22 = vld1q_f32(out2+3);
                        float32x4_t v32 = vld1q_f32(out3+3);
                        v12 = vmlaq_lane_f32(v12, w1, vget_high_f32(inp), 0);
                        v22 = vmlaq_lane_f32(v22, w2, vget_high_f32(inp), 0);
                        v32 = vmlaq_lane_f32(v32, w3, vget_high_f32(inp), 0);
                        vst1q_f32(out1+3, v12);
                        vst1q_f32(out2+3, v22);
                        vst1q_f32(out3+3, v32);
                        if(inw>4)
                        {
                            float32x4_t v13 = vld1q_f32(out1+5);
                            float32x4_t v23 = vld1q_f32(out2+5);
                            float32x4_t v33 = vld1q_f32(out3+5);
                            v13 = vmlaq_lane_f32(v13, w1, vget_high_f32(inp), 1);
                            v23 = vmlaq_lane_f32(v23, w2, vget_high_f32(inp), 1);
                            v33 = vmlaq_lane_f32(v33, w3, vget_high_f32(inp), 1);
                            vst1q_f32(out1+5, v13);
                            vst1q_f32(out2+5, v23);
                            vst1q_f32(out3+5, v33);
                        }
                        else
                        {
                            out1[5]+= inp_ptr[3]*ker_ptr[4];
                            out1[6]+= inp_ptr[3]*ker_ptr[5];
                            out1[7]+= inp_ptr[3]*ker_ptr[6];

                            out2[5]+= inp_ptr[3]*ker_ptr[8];
                            out2[6]+= inp_ptr[3]*ker_ptr[9];
                            out2[7]+= inp_ptr[3]*ker_ptr[10];

                            out3[5]+= inp_ptr[3]*ker_ptr[12];
                            out3[6]+= inp_ptr[3]*ker_ptr[13];
                            out3[7]+= inp_ptr[3]*ker_ptr[14];
                        }
                        inp_ptr+=4;
                        out1 += 7;
                        out2 += 7;
                        out3 += 7;
                    }
                    //w-mid =====================================
                    for(w=4;w+3<inw-4;w+=4)
                    {
                        //  printf("h begin | w mid\n");
                        //load inp[4]
                        float32x4_t inp = vld1q_f32(inp_ptr);
                        //v1
                        float32x4x2_t v1 = vld2q_f32(out1);
                        v1.val[0] = vmlaq_lane_f32(v1.val[0], inp, vget_low_f32(w1), 0);
                        v1.val[1] = vmlaq_lane_f32(v1.val[1], inp, vget_low_f32(w1), 1);
                        vst2q_f32(out1, v1);
                        v1 = vld2q_f32(out1 + 2);
                        v1.val[0] = vmlaq_lane_f32(v1.val[0], inp, vget_high_f32(w1), 0);
                        v1.val[1] = vmlaq_lane_f32(v1.val[1], inp, vget_high_f32(w1), 1);
                        vst2q_f32(out1 + 2, v1);

                        //v2
                        float32x4x2_t v2 = vld2q_f32(out2);
                        v2.val[0] = vmlaq_lane_f32(v2.val[0], inp, vget_low_f32(w2), 0);
                        v2.val[1] = vmlaq_lane_f32(v2.val[1], inp, vget_low_f32(w2), 1);
                        vst2q_f32(out2, v2);
                        v2 = vld2q_f32(out2 + 2);
                        v2.val[0] = vmlaq_lane_f32(v2.val[0], inp, vget_high_f32(w2), 0);
                        v2.val[1] = vmlaq_lane_f32(v2.val[1], inp, vget_high_f32(w2), 1);
                        vst2q_f32(out2 + 2, v2);

                        //v3
                        float32x4x2_t v3 = vld2q_f32(out3);
                        v3.val[0] = vmlaq_lane_f32(v3.val[0], inp, vget_low_f32(w3), 0);
                        v3.val[1] = vmlaq_lane_f32(v3.val[1], inp, vget_low_f32(w3), 1);
                        vst2q_f32(out3, v3);
                        v3 = vld2q_f32(out3 + 2);
                        v3.val[0] = vmlaq_lane_f32(v3.val[0], inp, vget_high_f32(w3), 0);
                        v3.val[1] = vmlaq_lane_f32(v3.val[1], inp, vget_high_f32(w3), 1);
                        vst2q_f32(out3 + 2, v3);

                        inp_ptr+=4;
                        out1 += 8;
                        out2 += 8;
                        out3 += 8;
                    }
                    //w-end [, , ,] ==============================
                    if(w<inw)
                    {
                        if(inw%4==0)
                        {
                            // printf("h begin | w end\n");
                            float32x4_t inp = vld1q_f32(inp_ptr);
                
                            float32x4_t v11 = vld1q_f32(out1);
                            float32x4_t v21 = vld1q_f32(out2);
                            float32x4_t v31 = vld1q_f32(out3);
                            v11 = vmlaq_lane_f32(v11, w1, vget_low_f32(inp), 0);
                            v21 = vmlaq_lane_f32(v21, w2, vget_low_f32(inp), 0);
                            v31 = vmlaq_lane_f32(v31, w3, vget_low_f32(inp), 0);
                            vst1q_f32(out1, v11);
                            vst1q_f32(out2, v21);
                            vst1q_f32(out3, v31);

                            float32x4_t v12 = vld1q_f32(out1+2);
                            float32x4_t v22 = vld1q_f32(out2+2);
                            float32x4_t v32 = vld1q_f32(out3+2);
                            v12 = vmlaq_lane_f32(v12, w1, vget_low_f32(inp), 1);
                            v22 = vmlaq_lane_f32(v22, w2, vget_low_f32(inp), 1);
                            v32 = vmlaq_lane_f32(v32, w3, vget_low_f32(inp), 1);
                            vst1q_f32(out1+2, v12);
                            vst1q_f32(out2+2, v22);
                            vst1q_f32(out3+2, v32);

                            float32x4_t v13 = vld1q_f32(out1+4);
                            float32x4_t v23 = vld1q_f32(out2+4);
                            float32x4_t v33 = vld1q_f32(out3+4);
                            v13 = vmlaq_lane_f32(v13, w1, vget_high_f32(inp), 0);
                            v23 = vmlaq_lane_f32(v23, w2, vget_high_f32(inp), 0);
                            v33 = vmlaq_lane_f32(v33, w3, vget_high_f32(inp), 0);
                            vst1q_f32(out1+4, v13);
                            vst1q_f32(out2+4, v23);
                            vst1q_f32(out3+4, v33);
                            out1[6]+= inp_ptr[3]*ker_ptr[4];
                            out1[7]+= inp_ptr[3]*ker_ptr[5];
                            out1[8]+= inp_ptr[3]*ker_ptr[6];

                            out2[6]+= inp_ptr[3]*ker_ptr[8];
                            out2[7]+= inp_ptr[3]*ker_ptr[9];
                            out2[8]+= inp_ptr[3]*ker_ptr[10];

                            out3[6]+= inp_ptr[3]*ker_ptr[12];
                            out3[7]+= inp_ptr[3]*ker_ptr[13];
                            out3[8]+= inp_ptr[3]*ker_ptr[14];
                            inp_ptr+=4;
                        }
                        else
                        {
                            for(;w<inw-1;w++)
                            {
                                float32x4_t inp = vld1q_f32(inp_ptr);
                    
                                float32x4_t v11 = vld1q_f32(out1);
                                float32x4_t v21 = vld1q_f32(out2);
                                float32x4_t v31 = vld1q_f32(out3);
                                v11 = vmlaq_lane_f32(v11, w1, vget_low_f32(inp), 0);
                                v21 = vmlaq_lane_f32(v21, w2, vget_low_f32(inp), 0);
                                v31 = vmlaq_lane_f32(v31, w3, vget_low_f32(inp), 0);
                                vst1q_f32(out1, v11);
                                vst1q_f32(out2, v21);
                                vst1q_f32(out3, v31);
                                out1+=2;
                                out2+=2;
                                out3+=2;
                                inp_ptr++;
                            }
                            {
                                out1[0]+= inp_ptr[0]*ker_ptr[4];
                                out1[1]+= inp_ptr[0]*ker_ptr[5];
                                out1[2]+= inp_ptr[0]*ker_ptr[6];

                                out2[0]+= inp_ptr[0]*ker_ptr[8];
                                out2[1]+= inp_ptr[0]*ker_ptr[9];
                                out2[2]+= inp_ptr[0]*ker_ptr[10];

                                out3[0]+= inp_ptr[0]*ker_ptr[12];
                                out3[1]+= inp_ptr[0]*ker_ptr[13];
                                out3[2]+= inp_ptr[0]*ker_ptr[14];
                                inp_ptr++;
                            }
                        }
                    }

                    // printM(output,8,8);
                }
                //h-mid
                for(h=1;h<inh-1;h++)
                {
                    float* out_ptr1 = out_ptr + (h*2 -1)*outw;
                    float* out0 = out_ptr1;
                    float* out1 = out0 + outw;
                    float* out2 = out1 + outw;
                    float* out3 = out2 + outw;
                    int w=0;
                    //w-begin [0,1,2,3] =========================
                    {
                        // printf("h mid | w begin\n");

                        float32x4_t inp = vld1q_f32(inp_ptr);
                        out0[0]+= inp_ptr[0]*ker_ptr[1];
                        out0[1]+= inp_ptr[0]*ker_ptr[2];
                        out0[2]+= inp_ptr[0]*ker_ptr[3];
                        out1[0]+= inp_ptr[0]*ker_ptr[5];
                        out1[1]+= inp_ptr[0]*ker_ptr[6];
                        out1[2]+= inp_ptr[0]*ker_ptr[7];
                        out2[0]+= inp_ptr[0]*ker_ptr[9];
                        out2[1]+= inp_ptr[0]*ker_ptr[10];
                        out2[2]+= inp_ptr[0]*ker_ptr[11];
                        out3[0]+= inp_ptr[0]*ker_ptr[13];
                        out3[1]+= inp_ptr[0]*ker_ptr[14];
                        out3[2]+= inp_ptr[0]*ker_ptr[15];
                        float32x4_t v1 = vld1q_f32(out0+1);
                        float32x4_t v11 = vld1q_f32(out1+1);
                        float32x4_t v21 = vld1q_f32(out2+1);
                        float32x4_t v31 = vld1q_f32(out3+1);
                        v1  = vmlaq_lane_f32(v1,  w0, vget_low_f32(inp), 1);
                        v11 = vmlaq_lane_f32(v11, w1, vget_low_f32(inp), 1);
                        v21 = vmlaq_lane_f32(v21, w2, vget_low_f32(inp), 1);
                        v31 = vmlaq_lane_f32(v31, w3, vget_low_f32(inp), 1);
                        vst1q_f32(out0+1, v1);
                        vst1q_f32(out1+1, v11);
                        vst1q_f32(out2+1, v21);
                        vst1q_f32(out3+1, v31);

                        float32x4_t v2 = vld1q_f32(out0+3);
                        float32x4_t v12 = vld1q_f32(out1+3);
                        float32x4_t v22 = vld1q_f32(out2+3);
                        float32x4_t v32 = vld1q_f32(out3+3);
                        v2  = vmlaq_lane_f32(v2,  w0, vget_high_f32(inp), 0);
                        v12 = vmlaq_lane_f32(v12, w1, vget_high_f32(inp), 0);
                        v22 = vmlaq_lane_f32(v22, w2, vget_high_f32(inp), 0);
                        v32 = vmlaq_lane_f32(v32, w3, vget_high_f32(inp), 0);
                        vst1q_f32(out0+3, v2);
                        vst1q_f32(out1+3, v12);
                        vst1q_f32(out2+3, v22);
                        vst1q_f32(out3+3, v32);

                        if(inw>4)
                        {
                            float32x4_t v3 = vld1q_f32(out0+5);
                            float32x4_t v13 = vld1q_f32(out1+5);
                            float32x4_t v23 = vld1q_f32(out2+5);
                            float32x4_t v33 = vld1q_f32(out3+5);
                            v3 = vmlaq_lane_f32(v3, w0, vget_high_f32(inp), 1);
                            v13 = vmlaq_lane_f32(v13, w1, vget_high_f32(inp), 1);
                            v23 = vmlaq_lane_f32(v23, w2, vget_high_f32(inp), 1);
                            v33 = vmlaq_lane_f32(v33, w3, vget_high_f32(inp), 1);
                            vst1q_f32(out0+5, v3);
                            vst1q_f32(out1+5, v13);
                            vst1q_f32(out2+5, v23);
                            vst1q_f32(out3+5, v33);
                        }
                        else
                        {
                            out0[5]+= inp_ptr[3]*ker_ptr[0];
                            out0[6]+= inp_ptr[3]*ker_ptr[1];
                            out0[7]+= inp_ptr[3]*ker_ptr[2];

                            out1[5]+= inp_ptr[3]*ker_ptr[4];
                            out1[6]+= inp_ptr[3]*ker_ptr[5];
                            out1[7]+= inp_ptr[3]*ker_ptr[6];

                            out2[5]+= inp_ptr[3]*ker_ptr[8];
                            out2[6]+= inp_ptr[3]*ker_ptr[9];
                            out2[7]+= inp_ptr[3]*ker_ptr[10];

                            out3[5]+= inp_ptr[3]*ker_ptr[12];
                            out3[6]+= inp_ptr[3]*ker_ptr[13];
                            out3[7]+= inp_ptr[3]*ker_ptr[14];
                        }
                        
                        inp_ptr+=4;
                        out0 += 7;
                        out1 += 7;
                        out2 += 7;
                        out3 += 7;
                    }
                    //w-mid =====================================
                    for(w=4;w+3<inw-4;w+=4)
                    {
                        //load inp[4]
                        float32x4_t inp = vld1q_f32(inp_ptr);
                        //v0
                        float32x4x2_t v0 = vld2q_f32(out0);
                        v0.val[0] = vmlaq_lane_f32(v0.val[0], inp, vget_low_f32(w0), 0);
                        v0.val[1] = vmlaq_lane_f32(v0.val[1], inp, vget_low_f32(w0), 1);
                        vst2q_f32(out0, v0);
                        v0 = vld2q_f32(out0 + 2);
                        v0.val[0] = vmlaq_lane_f32(v0.val[0], inp, vget_high_f32(w0), 0);
                        v0.val[1] = vmlaq_lane_f32(v0.val[1], inp, vget_high_f32(w0), 1);
                        vst2q_f32(out0 + 2, v0);

                        //v1
                        float32x4x2_t v1 = vld2q_f32(out1);
                        v1.val[0] = vmlaq_lane_f32(v1.val[0], inp, vget_low_f32(w1), 0);
                        v1.val[1] = vmlaq_lane_f32(v1.val[1], inp, vget_low_f32(w1), 1);
                        vst2q_f32(out1, v1);
                        v1 = vld2q_f32(out1 + 2);
                        v1.val[0] = vmlaq_lane_f32(v1.val[0], inp, vget_high_f32(w1), 0);
                        v1.val[1] = vmlaq_lane_f32(v1.val[1], inp, vget_high_f32(w1), 1);
                        vst2q_f32(out1 + 2, v1);

                        //v2
                        float32x4x2_t v2 = vld2q_f32(out2);
                        v2.val[0] = vmlaq_lane_f32(v2.val[0], inp, vget_low_f32(w2), 0);
                        v2.val[1] = vmlaq_lane_f32(v2.val[1], inp, vget_low_f32(w2), 1);
                        vst2q_f32(out2, v2);
                        v2 = vld2q_f32(out2 + 2);
                        v2.val[0] = vmlaq_lane_f32(v2.val[0], inp, vget_high_f32(w2), 0);
                        v2.val[1] = vmlaq_lane_f32(v2.val[1], inp, vget_high_f32(w2), 1);
                        vst2q_f32(out2 + 2, v2);

                        //v3
                        float32x4x2_t v3 = vld2q_f32(out3);
                        v3.val[0] = vmlaq_lane_f32(v3.val[0], inp, vget_low_f32(w3), 0);
                        v3.val[1] = vmlaq_lane_f32(v3.val[1], inp, vget_low_f32(w3), 1);
                        vst2q_f32(out3, v3);
                        v3 = vld2q_f32(out3 + 2);
                        v3.val[0] = vmlaq_lane_f32(v3.val[0], inp, vget_high_f32(w3), 0);
                        v3.val[1] = vmlaq_lane_f32(v3.val[1], inp, vget_high_f32(w3), 1);
                        vst2q_f32(out3 + 2, v3);

                        inp_ptr+=4;
                        out0 += 8;
                        out1 += 8;
                        out2 += 8;
                        out3 += 8;
                    }
                    //w-end [, , ,] ==============================
                    if(w<inw)
                    {
                        if(inw%4==0)
                        {
                            float32x4_t inp = vld1q_f32(inp_ptr);
                
                            float32x4_t v1 = vld1q_f32(out0);
                            float32x4_t v11 = vld1q_f32(out1);
                            float32x4_t v21 = vld1q_f32(out2);
                            float32x4_t v31 = vld1q_f32(out3);
                            v1  = vmlaq_lane_f32(v1,  w0, vget_low_f32(inp), 0);
                            v11 = vmlaq_lane_f32(v11, w1, vget_low_f32(inp), 0);
                            v21 = vmlaq_lane_f32(v21, w2, vget_low_f32(inp), 0);
                            v31 = vmlaq_lane_f32(v31, w3, vget_low_f32(inp), 0);
                            vst1q_f32(out0, v1);
                            vst1q_f32(out1, v11);
                            vst1q_f32(out2, v21);
                            vst1q_f32(out3, v31);

                            float32x4_t v2 = vld1q_f32(out0+2);
                            float32x4_t v12 = vld1q_f32(out1+2);
                            float32x4_t v22 = vld1q_f32(out2+2);
                            float32x4_t v32 = vld1q_f32(out3+2);
                            v2  = vmlaq_lane_f32(v2,  w0, vget_low_f32(inp), 1);
                            v12 = vmlaq_lane_f32(v12, w1, vget_low_f32(inp), 1);
                            v22 = vmlaq_lane_f32(v22, w2, vget_low_f32(inp), 1);
                            v32 = vmlaq_lane_f32(v32, w3, vget_low_f32(inp), 1);
                            vst1q_f32(out0+2, v2);
                            vst1q_f32(out1+2, v12);
                            vst1q_f32(out2+2, v22);
                            vst1q_f32(out3+2, v32);

                            float32x4_t v3  = vld1q_f32(out0+4);
                            float32x4_t v13 = vld1q_f32(out1+4);
                            float32x4_t v23 = vld1q_f32(out2+4);
                            float32x4_t v33 = vld1q_f32(out3+4);
                            v3 = vmlaq_lane_f32(v3, w0,   vget_high_f32(inp), 0);
                            v13 = vmlaq_lane_f32(v13, w1, vget_high_f32(inp), 0);
                            v23 = vmlaq_lane_f32(v23, w2, vget_high_f32(inp), 0);
                            v33 = vmlaq_lane_f32(v33, w3, vget_high_f32(inp), 0);
                            vst1q_f32(out0+4, v3);
                            vst1q_f32(out1+4, v13);
                            vst1q_f32(out2+4, v23);
                            vst1q_f32(out3+4, v33);

                            out0[6]+= inp_ptr[3]*ker_ptr[0];
                            out0[7]+= inp_ptr[3]*ker_ptr[1];
                            out0[8]+= inp_ptr[3]*ker_ptr[2];

                            out1[6]+= inp_ptr[3]*ker_ptr[4];
                            out1[7]+= inp_ptr[3]*ker_ptr[5];
                            out1[8]+= inp_ptr[3]*ker_ptr[6];

                            out2[6]+= inp_ptr[3]*ker_ptr[8];
                            out2[7]+= inp_ptr[3]*ker_ptr[9];
                            out2[8]+= inp_ptr[3]*ker_ptr[10];

                            out3[6]+= inp_ptr[3]*ker_ptr[12];
                            out3[7]+= inp_ptr[3]*ker_ptr[13];
                            out3[8]+= inp_ptr[3]*ker_ptr[14];
                            inp_ptr+=4;
                            out0 += 7;
                            out1 += 7;
                            out2 += 7;
                            out3 += 7;
                        }
                        else
                        {
                            for(;w<inw-1;w++)
                            {
                                float32x4_t inp = vld1q_f32(inp_ptr);
                                float32x4_t v1 = vld1q_f32(out0);
                                float32x4_t v11 = vld1q_f32(out1);
                                float32x4_t v21 = vld1q_f32(out2);
                                float32x4_t v31 = vld1q_f32(out3);
                                v1  = vmlaq_lane_f32(v1,  w0, vget_low_f32(inp), 0);
                                v11 = vmlaq_lane_f32(v11, w1, vget_low_f32(inp), 0);
                                v21 = vmlaq_lane_f32(v21, w2, vget_low_f32(inp), 0);
                                v31 = vmlaq_lane_f32(v31, w3, vget_low_f32(inp), 0);
                                vst1q_f32(out0, v1);
                                vst1q_f32(out1, v11);
                                vst1q_f32(out2, v21);
                                vst1q_f32(out3, v31);
                                out0+=2;
                                out1+=2;
                                out2+=2;
                                out3+=2;
                                inp_ptr+=1;
                            }
                            {
                                out0[0]+= inp_ptr[0]*ker_ptr[0];
                                out0[1]+= inp_ptr[0]*ker_ptr[1];
                                out0[2]+= inp_ptr[0]*ker_ptr[2];

                                out1[0]+= inp_ptr[0]*ker_ptr[4];
                                out1[1]+= inp_ptr[0]*ker_ptr[5];
                                out1[2]+= inp_ptr[0]*ker_ptr[6];

                                out2[0]+= inp_ptr[0]*ker_ptr[8];
                                out2[1]+= inp_ptr[0]*ker_ptr[9];
                                out2[2]+= inp_ptr[0]*ker_ptr[10];

                                out3[0]+= inp_ptr[0]*ker_ptr[12];
                                out3[1]+= inp_ptr[0]*ker_ptr[13];
                                out3[2]+= inp_ptr[0]*ker_ptr[14];
                                inp_ptr+=1;
                                out0 += 1;
                                out1 += 1;
                                out2 += 1;
                                out3 += 1;
                            }
                        }

                    }
                }
                //h-end
                if(h<inh)
                {
                    float* out0 = out_ptr+(h*2-1)*outw;
                    float* out1 = out0 + outw;
                    float* out2 = out1 + outw;
                    int w=0;
                    //w-begin [0,1,2,3] =========================
                    {
                        // printf("h end | w begin\n");

                        float32x4_t inp = vld1q_f32(inp_ptr);
                        out0[0]+= inp_ptr[0]*ker_ptr[1];
                        out0[1]+= inp_ptr[0]*ker_ptr[2];
                        out0[2]+= inp_ptr[0]*ker_ptr[3];
                        out1[0]+= inp_ptr[0]*ker_ptr[5];
                        out1[1]+= inp_ptr[0]*ker_ptr[6];
                        out1[2]+= inp_ptr[0]*ker_ptr[7];
                        out2[0]+= inp_ptr[0]*ker_ptr[9];
                        out2[1]+= inp_ptr[0]*ker_ptr[10];
                        out2[2]+= inp_ptr[0]*ker_ptr[11];

                        float32x4_t v1 = vld1q_f32(out0+1);
                        float32x4_t v11 = vld1q_f32(out1+1);
                        float32x4_t v21 = vld1q_f32(out2+1);
                        v1  = vmlaq_lane_f32(v1,  w0, vget_low_f32(inp), 1);
                        v11 = vmlaq_lane_f32(v11, w1, vget_low_f32(inp), 1);
                        v21 = vmlaq_lane_f32(v21, w2, vget_low_f32(inp), 1);
                        vst1q_f32(out0+1, v1);
                        vst1q_f32(out1+1, v11);
                        vst1q_f32(out2+1, v21);

                        float32x4_t v2 = vld1q_f32(out0+3);
                        float32x4_t v12 = vld1q_f32(out1+3);
                        float32x4_t v22 = vld1q_f32(out2+3);
                        v2  = vmlaq_lane_f32(v2,  w0, vget_high_f32(inp), 0);
                        v12 = vmlaq_lane_f32(v12, w1, vget_high_f32(inp), 0);
                        v22 = vmlaq_lane_f32(v22, w2, vget_high_f32(inp), 0);
                        vst1q_f32(out0+3, v2);
                        vst1q_f32(out1+3, v12);
                        vst1q_f32(out2+3, v22);

                        if(inw>4)
                        {
                            float32x4_t v3 = vld1q_f32(out0+5);
                            float32x4_t v13 = vld1q_f32(out1+5);
                            float32x4_t v23 = vld1q_f32(out2+5);
                            v3 = vmlaq_lane_f32(v3, w0, vget_high_f32(inp), 1);
                            v13 = vmlaq_lane_f32(v13, w1, vget_high_f32(inp), 1);
                            v23 = vmlaq_lane_f32(v23, w2, vget_high_f32(inp), 1);
                            vst1q_f32(out0+5, v3);
                            vst1q_f32(out1+5, v13);
                            vst1q_f32(out2+5, v23);
                        }
                        else
                        {
                            out0[5]+= inp_ptr[3]*ker_ptr[0];
                            out0[6]+= inp_ptr[3]*ker_ptr[1];
                            out0[7]+= inp_ptr[3]*ker_ptr[2];

                            out1[5]+= inp_ptr[3]*ker_ptr[4];
                            out1[6]+= inp_ptr[3]*ker_ptr[5];
                            out1[7]+= inp_ptr[3]*ker_ptr[6];

                            out2[5]+= inp_ptr[3]*ker_ptr[8];
                            out2[6]+= inp_ptr[3]*ker_ptr[9];
                            out2[7]+= inp_ptr[3]*ker_ptr[10];
                        }
                        inp_ptr+=4;
                        out0 += 7;
                        out1 += 7;
                        out2 += 7;
                    }
                    //w-mid =====================================
                    for(w=4;w+3<inw-4;w+=4)
                    {
                        //load inp[4]
                        float32x4_t inp = vld1q_f32(inp_ptr);
                        //v0
                        float32x4x2_t v0 = vld2q_f32(out0);
                        v0.val[0] = vmlaq_lane_f32(v0.val[0], inp, vget_low_f32(w0), 0);
                        v0.val[1] = vmlaq_lane_f32(v0.val[1], inp, vget_low_f32(w0), 1);
                        vst2q_f32(out0, v0);
                        v0 = vld2q_f32(out0 + 2);
                        v0.val[0] = vmlaq_lane_f32(v0.val[0], inp, vget_high_f32(w0), 0);
                        v0.val[1] = vmlaq_lane_f32(v0.val[1], inp, vget_high_f32(w0), 1);
                        vst2q_f32(out0 + 2, v0);

                        //v1
                        float32x4x2_t v1 = vld2q_f32(out1);
                        v1.val[0] = vmlaq_lane_f32(v1.val[0], inp, vget_low_f32(w1), 0);
                        v1.val[1] = vmlaq_lane_f32(v1.val[1], inp, vget_low_f32(w1), 1);
                        vst2q_f32(out1, v1);
                        v1 = vld2q_f32(out1 + 2);
                        v1.val[0] = vmlaq_lane_f32(v1.val[0], inp, vget_high_f32(w1), 0);
                        v1.val[1] = vmlaq_lane_f32(v1.val[1], inp, vget_high_f32(w1), 1);
                        vst2q_f32(out1 + 2, v1);

                        //v2
                        float32x4x2_t v2 = vld2q_f32(out2);
                        v2.val[0] = vmlaq_lane_f32(v2.val[0], inp, vget_low_f32(w2), 0);
                        v2.val[1] = vmlaq_lane_f32(v2.val[1], inp, vget_low_f32(w2), 1);
                        vst2q_f32(out2, v2);
                        v2 = vld2q_f32(out2 + 2);
                        v2.val[0] = vmlaq_lane_f32(v2.val[0], inp, vget_high_f32(w2), 0);
                        v2.val[1] = vmlaq_lane_f32(v2.val[1], inp, vget_high_f32(w2), 1);
                        vst2q_f32(out2 + 2, v2);

                        inp_ptr+=4;
                        out0 += 8;
                        out1 += 8;
                        out2 += 8;
                    }
                    //w-end [, , ,] ==============================
                    if(w<inw)
                    {
                        if(inw%4==0)
                        {
                            float32x4_t inp = vld1q_f32(inp_ptr);
                
                            float32x4_t v1 = vld1q_f32(out0);
                            float32x4_t v11 = vld1q_f32(out1);
                            float32x4_t v21 = vld1q_f32(out2);
                            v1  = vmlaq_lane_f32(v1,  w0, vget_low_f32(inp), 0);
                            v11 = vmlaq_lane_f32(v11, w1, vget_low_f32(inp), 0);
                            v21 = vmlaq_lane_f32(v21, w2, vget_low_f32(inp), 0);
                            vst1q_f32(out0, v1);
                            vst1q_f32(out1, v11);
                            vst1q_f32(out2, v21);

                            float32x4_t v2 = vld1q_f32(out0+2);
                            float32x4_t v12 = vld1q_f32(out1+2);
                            float32x4_t v22 = vld1q_f32(out2+2);
                            v2  = vmlaq_lane_f32(v2,  w0, vget_low_f32(inp), 1);
                            v12 = vmlaq_lane_f32(v12, w1, vget_low_f32(inp), 1);
                            v22 = vmlaq_lane_f32(v22, w2, vget_low_f32(inp), 1);
                            vst1q_f32(out0+2, v2);
                            vst1q_f32(out1+2, v12);
                            vst1q_f32(out2+2, v22);

                            float32x4_t v3  = vld1q_f32(out0+4);
                            float32x4_t v13 = vld1q_f32(out1+4);
                            float32x4_t v23 = vld1q_f32(out2+4);
                            v3 = vmlaq_lane_f32(v3, w0,   vget_high_f32(inp), 0);
                            v13 = vmlaq_lane_f32(v13, w1, vget_high_f32(inp), 0);
                            v23 = vmlaq_lane_f32(v23, w2, vget_high_f32(inp), 0);
                            vst1q_f32(out0+4, v3);
                            vst1q_f32(out1+4, v13);
                            vst1q_f32(out2+4, v23);

                            out0[6]+= inp_ptr[3]*ker_ptr[0];
                            out0[7]+= inp_ptr[3]*ker_ptr[1];
                            out0[8]+= inp_ptr[3]*ker_ptr[2];

                            out1[6]+= inp_ptr[3]*ker_ptr[4];
                            out1[7]+= inp_ptr[3]*ker_ptr[5];
                            out1[8]+= inp_ptr[3]*ker_ptr[6];

                            out2[6]+= inp_ptr[3]*ker_ptr[8];
                            out2[7]+= inp_ptr[3]*ker_ptr[9];
                            out2[8]+= inp_ptr[3]*ker_ptr[10];

                            inp_ptr+=4;
                            out0 += 7;
                            out1 += 7;
                            out2 += 7;
                        }
                        else
                        {
                            for(;w<inw-1;w++)
                            {
                                float32x4_t inp = vld1q_f32(inp_ptr);
                
                                float32x4_t v1 = vld1q_f32(out0);
                                float32x4_t v11 = vld1q_f32(out1);
                                float32x4_t v21 = vld1q_f32(out2);
                                v1  = vmlaq_lane_f32(v1,  w0, vget_low_f32(inp), 0);
                                v11 = vmlaq_lane_f32(v11, w1, vget_low_f32(inp), 0);
                                v21 = vmlaq_lane_f32(v21, w2, vget_low_f32(inp), 0);
                                vst1q_f32(out0, v1);
                                vst1q_f32(out1, v11);
                                vst1q_f32(out2, v21);
                                out0+=2;
                                out1+=2;
                                out2+=2;
                                inp_ptr++;
                            }
                            {
                                out0[0]+= inp_ptr[0]*ker_ptr[0];
                                out0[1]+= inp_ptr[0]*ker_ptr[1];
                                out0[2]+= inp_ptr[0]*ker_ptr[2];

                                out1[0]+= inp_ptr[0]*ker_ptr[4];
                                out1[1]+= inp_ptr[0]*ker_ptr[5];
                                out1[2]+= inp_ptr[0]*ker_ptr[6];

                                out2[0]+= inp_ptr[0]*ker_ptr[8];
                                out2[1]+= inp_ptr[0]*ker_ptr[9];
                                out2[2]+= inp_ptr[0]*ker_ptr[10];

                                inp_ptr+=1;
                                out0 += 1;
                                out1 += 1;
                                out2 += 1;
                            }
                        }

                    }
                }
            }
        }
    }
    else if(pad==0)
    {
        for(int g= group_start; g < group_end; g++)
        {
            float* out_ptr = output + g * out_hw;
            for(int c=0;c< cg; c++)
            {
                int group_idx = (g*cg + c);
                float* ker_ptr = kernel + group_idx*16;
                float32x4_t w0 = vld1q_f32(ker_ptr);
                float32x4_t w1 = vld1q_f32(ker_ptr+4);
                float32x4_t w2 = vld1q_f32(ker_ptr+8);
                float32x4_t w3 = vld1q_f32(ker_ptr+12);

                float* inp_ptr = input + group_idx* in_hw;
  
                for(int h=0;h<inh;h++)
                {
                    float* out_ptr1 = out_ptr + h*2*outw;
                    float* out0 = out_ptr1;
                    float* out1 = out0 + outw;
                    float* out2 = out1 + outw;
                    float* out3 = out2 + outw;
                    int w;
                    for(w=0;w+3<inw;w+=4)
                    {
                        //load inp[4]
                        float32x4_t inp = vld1q_f32(inp_ptr);
                        //v0
                        float32x4x2_t v0 = vld2q_f32(out0);
                        v0.val[0] = vmlaq_lane_f32(v0.val[0], inp, vget_low_f32(w0), 0);
                        v0.val[1] = vmlaq_lane_f32(v0.val[1], inp, vget_low_f32(w0), 1);
                        vst2q_f32(out0, v0);
                        v0 = vld2q_f32(out0 + 2);
                        v0.val[0] = vmlaq_lane_f32(v0.val[0], inp, vget_high_f32(w0), 0);
                        v0.val[1] = vmlaq_lane_f32(v0.val[1], inp, vget_high_f32(w0), 1);
                        vst2q_f32(out0 + 2, v0);

                        //v1
                        float32x4x2_t v1 = vld2q_f32(out1);
                        v1.val[0] = vmlaq_lane_f32(v1.val[0], inp, vget_low_f32(w1), 0);
                        v1.val[1] = vmlaq_lane_f32(v1.val[1], inp, vget_low_f32(w1), 1);
                        vst2q_f32(out1, v1);
                        v1 = vld2q_f32(out1 + 2);
                        v1.val[0] = vmlaq_lane_f32(v1.val[0], inp, vget_high_f32(w1), 0);
                        v1.val[1] = vmlaq_lane_f32(v1.val[1], inp, vget_high_f32(w1), 1);
                        vst2q_f32(out1 + 2, v1);

                        //v2
                        float32x4x2_t v2 = vld2q_f32(out2);
                        v2.val[0] = vmlaq_lane_f32(v2.val[0], inp, vget_low_f32(w2), 0);
                        v2.val[1] = vmlaq_lane_f32(v2.val[1], inp, vget_low_f32(w2), 1);
                        vst2q_f32(out2, v2);
                        v2 = vld2q_f32(out2 + 2);
                        v2.val[0] = vmlaq_lane_f32(v2.val[0], inp, vget_high_f32(w2), 0);
                        v2.val[1] = vmlaq_lane_f32(v2.val[1], inp, vget_high_f32(w2), 1);
                        vst2q_f32(out2 + 2, v2);

                        //v3
                        float32x4x2_t v3 = vld2q_f32(out3);
                        v3.val[0] = vmlaq_lane_f32(v3.val[0], inp, vget_low_f32(w3), 0);
                        v3.val[1] = vmlaq_lane_f32(v3.val[1], inp, vget_low_f32(w3), 1);
                        vst2q_f32(out3, v3);
                        v3 = vld2q_f32(out3 + 2);
                        v3.val[0] = vmlaq_lane_f32(v3.val[0], inp, vget_high_f32(w3), 0);
                        v3.val[1] = vmlaq_lane_f32(v3.val[1], inp, vget_high_f32(w3), 1);
                        vst2q_f32(out3 + 2, v3);

                        inp_ptr+=4;
                        out0 += 8;
                        out1 += 8;
                        out2 += 8;
                        out3 += 8;
                    }
                    for(;w<inw;w++)
                    {
                        float inp=inp_ptr[0];
                        out0[0] += inp * ker_ptr[0];
                        out0[1] += inp * ker_ptr[1];
                        out0[2] += inp * ker_ptr[2];
                        out0[3] += inp * ker_ptr[3];

                        out1[0] += inp * ker_ptr[4];
                        out1[1] += inp * ker_ptr[5];
                        out1[2] += inp * ker_ptr[6];
                        out1[3] += inp * ker_ptr[7];

                        out2[0] += inp * ker_ptr[8];
                        out2[1] += inp * ker_ptr[9];
                        out2[2] += inp * ker_ptr[10];
                        out2[3] += inp * ker_ptr[11];

                        out3[0] += inp * ker_ptr[12];
                        out3[1] += inp * ker_ptr[13];
                        out3[2] += inp * ker_ptr[14];
                        out3[3] += inp * ker_ptr[15];
                        
                        inp_ptr+=1;
                        out0 += 2;
                        out1 += 2;
                        out2 += 2;
                        out3 += 2;
                    }
                }
            }
        }
    }


    if(activation == 0 || activation == 6)
    {
        int out_offset = group_start* out_hw;
        int out_end = group_end* out_hw;
        for(int i=out_offset;i<out_end;i++)
        {
            output[i]=do_activation(output[i],activation);
        }
    }
}

void initial_output(float* output, float* bias, int output_ch, int output_wh)
{
    int i, j;
    // no bias
    if(bias == nullptr)
    {
        memset(output, 0.f, output_ch * output_wh* sizeof(float));
    }
    else
    {
        float* out_ptr= output;
        for(i = 0; i < output_ch; i++)
            for(j = 0; j < output_wh; j++)
                *out_ptr++ = bias[i];
    }
}

struct deconv_dw_param
{
    float* input_buf;
    float* weight_buf;
    float* output_buf;
    int group_start;
    int group_end;
    int cg;
    int input_h;
    int input_w;
    int output_h;
    int output_w;
    int activation;
    int pad;
};

struct DeConv2dDepth4x4 : public MTNodeOps
{
    DeConv2dDepth4x4()
    {
        name_ = "arm_dw4x4_deconv_fp32";
    }

    int activation;

    bool Run(Node* node);

    bool Aider(int cpu, int seq, void* data);
};

bool DeConv2dDepth4x4::Aider(int cpu, int seq, void* data)
{
    deconv_dw_param* param = ( deconv_dw_param* )data;

    deconv_4x4s2(param->input_buf, param->weight_buf,param->output_buf,
                param->group_start,param->group_end,param->cg,
                param->input_h, param->input_w,  param->output_h, param->output_w,
                param->activation,param->pad);

    return true;
}


bool DeConv2dDepth4x4::Run(Node* node)
{
    Tensor* input_tensor = node->GetInputTensor(0);
    Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(node->GetOp());
    DeconvParam* param_ = deconv_op->GetParam();
    int group=param_->group;
    int pad=param_->pad_h0;

    const TShape& input_shape = input_tensor->GetShape();

    int input_c = input_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();

    /* output */
    Tensor* output_tensor = node->GetOutputTensor(0);
    TShape& output_shape = output_tensor->GetShape();
    

    int output_h = output_shape.GetH();
    int output_w = output_shape.GetW();
    int output_n = output_shape.GetN();
    int output_c = output_shape.GetC();
    int output_hw = output_h * output_w;
    
    Tensor* weight_tensor = node->GetInputTensor(1);
    // TShape& wshape = weight_tensor->GetShape();
    // printf("weight shape %d %d %d %d\n",wshape.GetN(),wshape.GetC(),wshape.GetH(),wshape.GetW());
    // printf("output shape %d %d %d %d\n",output_n,output_c,output_h,output_w);

    float* weight_buf = ( float* )get_tensor_mem(weight_tensor);
    float* input_buf = ( float* )get_tensor_mem(input_tensor);
    float* output_buf = ( float* )get_tensor_mem(output_tensor);
    int input_size = input_c * input_h * input_w;
    int output_size = output_c * output_h * output_w;
    int cg=input_c/group;

    int cpu_number = cpu_info->GetCPUNumber();

    float* bias = nullptr;

    //get bias
    if(node->GetInputNum() > 2)
    {
        Tensor* bias_tensor = node->GetInputTensor(2);
        bias = ( float* )get_tensor_mem(bias_tensor);
    }
    for(int b = 0; b < output_n; b++)
    {
        float* cur_input = input_buf + b * input_size;
        float* cur_output = output_buf + b * output_size;

        initial_output(cur_output, bias, output_c, output_hw);
        if(cpu_number == 1)
        {
            deconv_4x4s2(cur_input,
                            weight_buf,
                            cur_output,
                            0,group, cg,
                            input_h, input_w,
                            output_h,output_w,activation,pad);
        }
        else
        {
            std::vector<sub_op_task> task_list;
            std::vector<deconv_dw_param> param_list;
            int step = group/cpu_number;
            int task_number = cpu_number;
            if(group <=cpu_number)
            {
                task_number=group;
                step=1;
            }
            task_list.resize(task_number);
            param_list.resize(task_number);

            auto f = std::bind(&DeConv2dDepth4x4::Aider, this, std::placeholders::_1, std::placeholders::_2,
                                std::placeholders::_3);
            for(int i = 0; i < task_number; i++)
            {
                
                deconv_dw_param* param = &param_list[i];
                sub_op_task* task = &task_list[i];
                task->exec_func = f;
                task->seq = i;
                task->data = param;

                param->input_buf = cur_input;
                param->weight_buf = weight_buf;
                param->output_buf = cur_output;
                param->group_start = i*step;
                param->group_end = param->group_start + step;
                param->cg=cg;
                param->input_h = input_h;
                param->input_w = input_w;
                param->output_h = output_h;
                param->output_w = output_w;
                param->activation = activation;
                param->pad = pad;
            }
            param_list[task_number - 1].group_end = group;
            task_dispatch(task_list, -1);
            wait_done();
        }

 
    }

    return true;
}

static bool isDepthwiseSupported(const DeconvParam* param, const TShape& input_shape,const TShape& output_shape)
{

    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;
    int output_c = output_shape.GetC();
    int input_h = input_shape.GetH();
    int input_w = input_shape.GetW();
    if(input_h<4||input_w<4)
    {
        return false;
    }
    if(group == 1 || group!=output_c || kernel_h != 4 || kernel_w != 4 || ((pad_h0 != 1)&&(pad_h0!=0)) ||pad_h0!= pad_w0 || 
       dilation_h != 1 || dilation_w != 1 ||
       stride_w != 2||stride_h!=2)
    {
        return false;
    }
    return true;
}

NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    const ExecAttr* exec_attr = any_cast<const ExecAttr*>(node->GetAttr(ATTR_EXEC_ATTR));

#ifdef CONFIG_AUTH_DEVICE
    bool float_enabled = get_auth_float_enabled();

    if(!float_enabled)
        return nullptr;
#endif

    if(data_type != TENGINE_DT_FP32 || exec_attr->graph_layout != TENGINE_LAYOUT_NCHW)

        return nullptr;

    Operator* op = node->GetOp();

    Deconvolution* deconv_op = dynamic_cast<Deconvolution*>(op);
    DeconvParam* param = deconv_op->GetParam();

    const TShape& input_shape = node->GetInputTensor(0)->GetShape();
    const TShape& output_shape = node->GetOutputTensor(0)->GetShape();

    if(!isDepthwiseSupported(param, input_shape, output_shape))
        return nullptr;

    DeConv2dDepth4x4* ops = new DeConv2dDepth4x4();
    ops->activation = param->activation;

    return ops;
}

}    // namespace deconv_2d_dw_4x4

void RegisterDeConv2dDepth4x4(void)
{
    if(!NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Deconvolution", deconv_2d_dw_4x4::SelectFunc,
                                                      deconv_2d_dw_4x4::default_prio))
        LOG_ERROR() << __FUNCTION__ << " :Regist OP failed for prio[" << deconv_2d_dw_4x4::default_prio << "]\n";
}

}    // namespace TEngine
