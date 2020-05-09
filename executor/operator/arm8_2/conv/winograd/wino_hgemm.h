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

#ifndef __WINO_SGEMM_H__
#define __WINO_SGEMM_H__

#include "wino_config.h"
#include "wino_trans_out.h"
#ifdef __cplusplus
extern "C" {
#endif    //__cplusplus

//#define D_WINO

extern void wino_hgemm_4x16_fp16(__fp16* output, const __fp16* input,const  __fp16* kernel, long cin, int stride_save);
extern void wino_hgemm_1x16_fp16(__fp16* output, const __fp16* input,const  __fp16* kernel, long cin);
extern void wino_trans_out4_fp16(__fp16* buffer,  __fp16* output,int out_w,__fp16*ker00, __fp16*bias,int activation);

// pour debug
static inline void wino_sgemm_4x16_cpu(__fp16* output,const __fp16* input,const __fp16* kernel, long cin)
{
    for(int i = 0; i < KER_COUT_UNIT; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            __fp16 sum = 0;
            for(int k = 0; k < cin; k++)
            {
                sum += input[k * 4 + j] * kernel[k * KER_COUT_UNIT + i];
            }
            output[i * 4 + j] = sum;
        }
    }
}

static inline void wino_sgemm_1x16_cpu(__fp16* output, const __fp16* input,const __fp16* kernel, long cin)
{
    for(int i = 0; i < KER_COUT_UNIT; i++)
    {
        __fp16 sum = 0;
        for(int k = 0; k < cin; k++)
        {
            sum += input[k] * kernel[k * 16 + i];
        }
        output[i] = sum;
    }
}

// pour debug trans_out_4.S
static inline void single_out( __fp16* mid, __fp16* out__, int outw, const __fp16* bias_ptr,int activation)
{
    __fp16 tmp[24] = {0};
    __fp16 r1_add_r2[6];
    __fp16 r1_minus_r2[6];
    __fp16 r3_add_r4[6];
    __fp16 r3_minus_r4_x2[6];
    __fp16 coef[2]={0.125,32};

    for(int ii=0;ii<4;ii++)
    {
        // __fp16* mid = mid0+ii;
        __fp16* out=out__+ ii*TILE;
        for(int j = 0; j < 6; j++)
        {
            r1_add_r2[j] = mid[24 * 1 + j*4 +ii] + mid[24 * 2 + j*4 +ii];
            r1_minus_r2[j] = mid[24 * 1 + j*4 +ii] - mid[24 * 2 + j*4 +ii];
            r3_add_r4[j] = mid[24 * 3 + j*4 +ii] + mid[24 * 4 + j*4 +ii];
            r3_minus_r4_x2[j] = (mid[24 * 3 + j*4 +ii] - mid[24 * 4 + j*4 +ii]) * 2;
        }
        for(int j = 0; j < 6; j++)
        {
            tmp[j]      =coef[0]*(mid[j*4 +ii] + r1_add_r2[j] + r3_add_r4[j]);
            tmp[6 + j]  =coef[0]*(r1_minus_r2[j] + r3_minus_r4_x2[j]);
            tmp[12 + j] =coef[0]*(r1_add_r2[j] + 4 * r3_add_r4[j]);
            tmp[18 + j] =coef[0]*(r1_minus_r2[j] + 4 * r3_minus_r4_x2[j] + mid[24 * 5 + j*4+ii]);
        }

        __fp16* out0 = out;
        __fp16* out1 = out0 + outw;
        __fp16* out2 = out1 + outw;
        __fp16* out3 = out2 + outw;

        __fp16 _r1_add_r2[4];
        __fp16 _r1_minus_r2[4];
        __fp16 _r3_add_r4[4];
        __fp16 _r3_minus_r4_x2[4];
        int idx;
        for(int j = 0; j < 4; j++)
        {
            idx = 6 * j;
            _r1_add_r2[j] = tmp[idx + 1] + tmp[idx + 2];
            _r1_minus_r2[j] = tmp[idx + 1] - tmp[idx + 2];
            _r3_add_r4[j] = tmp[idx + 3] + tmp[idx + 4];
            _r3_minus_r4_x2[j] = (tmp[idx + 3] - tmp[idx + 4]) * 2;
        }
        if(bias_ptr)
        {
            __fp16 bias = bias_ptr[0];
            out0[0] = do_activation(coef[1]*(tmp[0 * 6] + _r1_add_r2[0] + _r3_add_r4[0]) + bias, activation);
            out1[0] = do_activation(coef[1]*(tmp[1 * 6] + _r1_add_r2[1] + _r3_add_r4[1]) + bias, activation);
            out2[0] = do_activation(coef[1]*(tmp[2 * 6] + _r1_add_r2[2] + _r3_add_r4[2]) + bias, activation);
            out3[0] = do_activation(coef[1]*(tmp[3 * 6] + _r1_add_r2[3] + _r3_add_r4[3]) + bias, activation);

            out0[1] = do_activation(coef[1]*(_r1_minus_r2[0] + _r3_minus_r4_x2[0]) + bias, activation);
            out1[1] = do_activation(coef[1]*(_r1_minus_r2[1] + _r3_minus_r4_x2[1]) + bias, activation);
            out2[1] = do_activation(coef[1]*(_r1_minus_r2[2] + _r3_minus_r4_x2[2]) + bias, activation);
            out3[1] = do_activation(coef[1]*(_r1_minus_r2[3] + _r3_minus_r4_x2[3]) + bias, activation);

            out0[2] = do_activation(coef[1]*(_r1_add_r2[0] + 4 * _r3_add_r4[0]) + bias, activation);
            out1[2] = do_activation(coef[1]*(_r1_add_r2[1] + 4 * _r3_add_r4[1]) + bias, activation);
            out2[2] = do_activation(coef[1]*(_r1_add_r2[2] + 4 * _r3_add_r4[2]) + bias, activation);
            out3[2] = do_activation(coef[1]*(_r1_add_r2[3] + 4 * _r3_add_r4[3]) + bias, activation);

            out0[3] = do_activation(coef[1]*(_r1_minus_r2[0] + 4 * _r3_minus_r4_x2[0] + tmp[0 * 6 + 5]) + bias, activation);
            out1[3] = do_activation(coef[1]*(_r1_minus_r2[1] + 4 * _r3_minus_r4_x2[1] + tmp[1 * 6 + 5]) + bias, activation);
            out2[3] = do_activation(coef[1]*(_r1_minus_r2[2] + 4 * _r3_minus_r4_x2[2] + tmp[2 * 6 + 5]) + bias, activation);
            out3[3] = do_activation(coef[1]*(_r1_minus_r2[3] + 4 * _r3_minus_r4_x2[3] + tmp[3 * 6 + 5]) + bias, activation);
        }
        else
        {
            out0[0] = do_activation(coef[1]*(tmp[0 * 6] + _r1_add_r2[0] + _r3_add_r4[0]), activation);
            out1[0] = do_activation(coef[1]*(tmp[1 * 6] + _r1_add_r2[1] + _r3_add_r4[1]), activation);
            out2[0] = do_activation(coef[1]*(tmp[2 * 6] + _r1_add_r2[2] + _r3_add_r4[2]), activation);
            out3[0] = do_activation(coef[1]*(tmp[3 * 6] + _r1_add_r2[3] + _r3_add_r4[3]), activation);

            out0[1] = do_activation(coef[1]*(_r1_minus_r2[0] + _r3_minus_r4_x2[0]), activation);
            out1[1] = do_activation(coef[1]*(_r1_minus_r2[1] + _r3_minus_r4_x2[1]), activation);
            out2[1] = do_activation(coef[1]*(_r1_minus_r2[2] + _r3_minus_r4_x2[2]), activation);
            out3[1] = do_activation(coef[1]*(_r1_minus_r2[3] + _r3_minus_r4_x2[3]), activation);

            out0[2] = do_activation(coef[1]*(_r1_add_r2[0] + 4 * _r3_add_r4[0]), activation);
            out1[2] = do_activation(coef[1]*(_r1_add_r2[1] + 4 * _r3_add_r4[1]), activation);
            out2[2] = do_activation(coef[1]*(_r1_add_r2[2] + 4 * _r3_add_r4[2]), activation);
            out3[2] = do_activation(coef[1]*(_r1_add_r2[3] + 4 * _r3_add_r4[3]), activation);

            out0[3] = do_activation(coef[1]*(_r1_minus_r2[0] + 4 * _r3_minus_r4_x2[0] + tmp[0 * 6 + 5]), activation);
            out1[3] = do_activation(coef[1]*(_r1_minus_r2[1] + 4 * _r3_minus_r4_x2[1] + tmp[1 * 6 + 5]), activation);
            out2[3] = do_activation(coef[1]*(_r1_minus_r2[2] + 4 * _r3_minus_r4_x2[2] + tmp[2 * 6 + 5]), activation);
            out3[3] = do_activation(coef[1]*(_r1_minus_r2[3] + 4 * _r3_minus_r4_x2[3] + tmp[3 * 6 + 5]), activation);
        }
    }
}

static inline void wino_sgemm_4x16(const __fp16* ker, const __fp16* inp, __fp16* output, const __fp16* bias, int bias_term, int cin,
                            int cpu_type, int cout_start, int cout_end, int block_start, int block_end, int block_h,
                            int block_w, int out_hw, int out_w, int resi_h, int resi_w,int activation)
{
    int p, i;
    int flag_outw=1;
    if(out_w<16)flag_outw=0;
    const __fp16* ker_ptr;
    const __fp16* inp_ptr;
    #ifdef D_WINO
    printf("[SGEMM4X16]\tcout[%d-%d]\tblock[%d-%d]\tblock_h,w[%d,%d]\tresi_h,w[%d,%d]\n",
                cout_start,cout_end,block_start,block_end,block_h,block_w,resi_h,resi_w);
    #endif
    for(p = (cout_start & -KER_COUT_UNIT); p < (cout_end & -KER_COUT_UNIT); p += KER_COUT_UNIT)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;

        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            __fp16 out_buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
            int idx_h[4];
            int idx_w[4];
            idx_h[0] = (i) / block_w;
            idx_h[1] = (i + 1) / block_w;
            idx_h[2] = (i + 2) / block_w;
            idx_h[3] = (i + 3) / block_w;

            idx_w[0] = (i) % block_w;
            idx_w[1] = (i + 1) % block_w;
            idx_w[2] = (i + 2) % block_w;
            idx_w[3] = (i + 3) % block_w;
            int wino_out_4_tiles=0;
            int mulitplier = KER_COUT_UNIT;
            if(flag_outw)
                if((idx_h[0]==idx_h[3]) && (resi_h==0) && (resi_w==0))
                {
                    wino_out_4_tiles=1;
                    mulitplier=1;
                }
                    
            for(int s = 0; s < ELEM_SIZE; s++)
            {

                // wino_hgemm_4x16_fp16(out_buffer + s * BLOCK_HW_UNIT * mulitplier,
                //                     inp_ptr + s * BLOCK_HW_UNIT * cin, ker_ptr + s * KER_COUT_UNIT * cin, 
                //                     cin,
                //                     0);
                wino_hgemm_4x16_fp16(out_buffer + s * BLOCK_HW_UNIT * mulitplier,
                                    inp_ptr + s * BLOCK_HW_UNIT * cin, ker_ptr + s * KER_COUT_UNIT * cin, 
                                    cin,
                                    wino_out_4_tiles);
                // wino_sgemm_4x16_cpu(out_buffer + s * BLOCK_HW_UNIT * KER_COUT_UNIT,
                //                     inp_ptr + s * BLOCK_HW_UNIT * cin, ker_ptr + s * KER_COUT_UNIT * cin, 
                //                     cin);
            }
            if (wino_out_4_tiles==1)
            {
                // #ifdef D_WINO
                // printf("\t\t\t\t block_idx:%d,[idx_h:%d] [idx_w:%d-%d]\n",i,idx_h[0],idx_w[0],idx_w[3]);
                // #endif
                // #ifdef D_WINO
                // DumpFloat("pr_buffer",buffer,ELEM_SIZE*BLOCK_HW_UNIT*KER_COUT_UNIT,BLOCK_HW_UNIT*6,6);
                // #endif
                __fp16* bias_ptr = NULL;
                for(int pss = 0; pss < KER_COUT_UNIT; pss++)
                {
                    int cout_idx = p + pss;
                    __fp16* out_ptr = output + cout_idx * out_hw + idx_h[0] * TILE * out_w + idx_w[0] * TILE;
                    if(bias_term)
                    {
                        bias_ptr =(__fp16*)( bias + cout_idx);
                    }
                    // single_out(out_buffer+pss*ELEM_SIZE*BLOCK_HW_UNIT, 
                    //             out_ptr, 
                    //             out_w, (const __fp16*)bias_ptr,activation);
                    // __fp16 ker00[4] = {2,4,8,0};
                    __fp16 ker00[8] = {2,4,8,0.125,32,0,0,0};

                    wino_trans_out4_fp16(out_buffer+pss*ELEM_SIZE*BLOCK_HW_UNIT, 
                                out_ptr, 
                                out_w*sizeof(__fp16), ker00,bias_ptr,activation);
                }
                // #ifdef D_WINO
                // DumpFloat("pr_out_4",output,cout_end*out_hw,out_w,out_hw/out_w); 
                // # endif
            }
            else
            {
                __fp16 buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
                __fp16* buffer_ptr0 = buffer;
                // gemm_save[36][4][16] -> [4][16][36]
                for(int pp = 0; pp < KER_COUT_UNIT; pp++)
                {
                    for(int t = 0; t < BLOCK_HW_UNIT; t++)
                    {
                        for(int ss = 0; ss < ELEM_SIZE; ss++)
                        {
                            *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT + t*KER_COUT_UNIT +  pp];
                            buffer_ptr0++;
                        }
                    }
                }
                // end interleave
                // transform_output_f43_4tile((const __fp16*)buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h, resi_w,
                //                            KER_COUT_UNIT, bias, bias_term);
                {
                    __fp16 tmp_buffer[TILE * TILE];
                    const __fp16* bias_ptr = NULL;
                    for(int pss = 0; pss < KER_COUT_UNIT; pss++)
                    {
                        int cout_idx = p + pss;
                        __fp16* out_ptr = output + cout_idx * out_hw;
                        if(bias_term)
                        {
                            bias_ptr = bias + cout_idx;
                        }
                        for(int ii = 0; ii < BLOCK_HW_UNIT; ii++)
                        {
                            int i_h = idx_h[ii];
                            int j_w = idx_w[ii];
                            if((resi_h == 0 && resi_w == 0) || (resi_h == 0 && (j_w < block_w - 1)) ||
                            (resi_w == 0 && (i_h < block_h - 1)) || ((j_w < block_w - 1) && (i_h < block_h - 1)))
                            {
                                trans_output_f43(buffer+ii*ELEM_SIZE+pss*36*4, 
                                                 out_ptr + (i_h * TILE * out_w + j_w * TILE), 
                                                 out_w, (const __fp16*)bias_ptr,
                                                 activation);
                            }    // direct use_out_ptr
                            else
                            {
                                int ret_h = TILE - resi_h;
                                if(i_h < block_h - 1)
                                    ret_h = TILE;
                                int ret_w = TILE - resi_w;
                                if(j_w < block_w - 1)
                                    ret_w = TILE;
                                // tmp_buffer
                                trans_output_f43_ordinary(buffer+ii*ELEM_SIZE+pss*36*4, tmp_buffer,  (const __fp16*)bias_ptr);
                                __fp16* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                                for(int hh = 0; hh < ret_h; hh++)
                                {
                                  for(int ww = 0; ww < ret_w; ww++)
                                  {
                                    out_pointer[hh * out_w + ww] = do_activation(tmp_buffer[hh * 4 + ww],activation);
                                  }
                                }
                            }    // end else, tmp_buff
                        }
                    }
                }
                // end transform
            }
        }

        for(; i < block_end; i++)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            __fp16 out_buffer[KER_COUT_UNIT * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                // wino_sgemm_1x16(out_buffer + s * KER_COUT_UNIT, inp_ptr + s * cin,
                //                     ker_ptr + s * KER_COUT_UNIT * cin, cin);
                wino_hgemm_1x16_fp16(out_buffer + s * KER_COUT_UNIT, inp_ptr + s * cin,
                                    ker_ptr + s * KER_COUT_UNIT * cin, cin);
                                    
            }
            // interleave
            __fp16 buffer[KER_COUT_UNIT * ELEM_SIZE];
            __fp16* buffer_ptr0 = buffer;

            for(int pp = 0; pp < KER_COUT_UNIT; pp++)
            {
                for(int ss = 0; ss < ELEM_SIZE; ss++)
                {
                    *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT + pp];
                    buffer_ptr0++;
                }
            }
            // end interleave
            transform_output_f43_1tile((const __fp16*)buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h, resi_w,
                                       KER_COUT_UNIT, bias, bias_term,activation);
            // end transform
        }
    }
}


//inp [block_hw//4][36][cin][4] ->[36][block_hw//4][cin][4]
//ker [cout//16][36][cin][16] -> [36][cout//16][cin][16]
//mid [cout//16]([block//4][36][16][4] + block_i[36][16]
static inline void wino_sgemm_4x16_1(const __fp16* ker, const __fp16* inp, __fp16* trans_out,  int cin,
                            int cpu_type, int cout_start, int cout_end, int block_start, int block_end,
                            int block_hw,int outc,int s)
{
    int p, i;
    __fp16* out_ptr;
    __fp16* out_ptr1;


    for(p = (cout_start & -KER_COUT_UNIT); p < (cout_end & -KER_COUT_UNIT); p += KER_COUT_UNIT)
    {
        out_ptr = trans_out + p*ELEM_SIZE* block_hw;
        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            out_ptr1 = out_ptr + i * ELEM_SIZE * KER_COUT_UNIT;

            int offset= s * block_hw * cin +  i * cin;
            int offset_ker = s* cin*outc + p*cin; 
            if(cpu_type == TYPE_A76)
            {
                wino_hgemm_4x16_fp16(out_ptr1 + s *  BLOCK_HW_UNIT * KER_COUT_UNIT  ,
                                    inp + offset,
                                    ker + offset_ker, 
                                    cin,0);
            }
            else
            {
                wino_sgemm_4x16_cpu(out_ptr1 + s *  BLOCK_HW_UNIT * KER_COUT_UNIT  ,
                                    inp + offset, 
                                    ker + offset_ker, 
                                    cin);
            }
        }
    
        for(; i < block_end; i++)
        {
            out_ptr1 = out_ptr + i*ELEM_SIZE*KER_COUT_UNIT;

            int offset_ker = s* cin*outc + p*cin; 
            int offset= s * block_hw * cin +  i * cin;

            wino_sgemm_1x16_cpu(out_ptr1 + s * KER_COUT_UNIT, 
                            inp + offset,
                            ker + offset_ker, cin);
        }
    }
}


#ifdef __cplusplus
}
#endif    //__cplusplus

#endif    // __WINO_SGENN_H__
