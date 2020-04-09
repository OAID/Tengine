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


#include "wino_sgemm.h"

void wino_sgemm_4x4(const float* ker, const float* inp, float* output, const float* bias, int bias_term,
                                   int cin, int cpu_type, int cout_start, int cout_end, int block_start, int block_end,
                                   int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                   int activation)
{
    int p, i;
    int flag_outw = 1;
    if(out_w < 16)
        flag_outw = 0;
    const float* ker_ptr;
    const float* inp_ptr;
    // #ifdef D_WINO
    //     printf("[SGEMM4X16]\tcout[%d-%d]\tblock[%d-%d]\tblock_h,w[%d,%d]\tresi_h,w[%d,%d]\n", cout_start, cout_end,
    //         block_start, block_end, block_h, block_w, resi_h, resi_w);
    // #endif
    for(p = (cout_start & -KER_COUT_UNIT4); p < (cout_end & -KER_COUT_UNIT4); p += KER_COUT_UNIT4)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;

        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT4 * BLOCK_HW_UNIT * ELEM_SIZE];
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
            int wino_out_4_tiles = 0;
            int mulitplier = KER_COUT_UNIT4;
            if(flag_outw)
                if((idx_h[0] == idx_h[3]) && (idx_h[0] < (block_h - 1)) && (idx_w[3] < (block_w - 1)))
                {
                    wino_out_4_tiles = 1;
                    mulitplier = 1;
                }

            for(int s = 0; s < ELEM_SIZE; s++)
            {
                // if(cpu_type == TYPE_A72)
                // {
                //     wino_sgemm_4x4_A72(out_buffer + s * BLOCK_HW_UNIT * mulitplier, inp_ptr + s * BLOCK_HW_UNIT * cin,
                //                         ker_ptr + s * KER_COUT_UNIT4 * cin, cin, wino_out_4_tiles);
                // }
                // else
                {
                    wino_sgemm_4x4_A72(out_buffer + s * BLOCK_HW_UNIT * mulitplier, inp_ptr + s * BLOCK_HW_UNIT * cin,
                                        ker_ptr + s * KER_COUT_UNIT4 * cin, cin, wino_out_4_tiles);
                }
            }
            if(wino_out_4_tiles == 1)
            {
                #ifdef D_WINO
                                printf("\t\t\t\t block_idx:%d,[idx_h:%d] [idx_w:%d-%d]\n", i, idx_h[0], idx_w[0], idx_w[3]);
                #endif
                // #ifdef D_WINO
                // DumpFloat("pr_buffer",buffer,ELEM_SIZE*BLOCK_HW_UNIT*KER_COUT_UNIT,BLOCK_HW_UNIT*6,6);
                // #endif
                float* bias_ptr = NULL;
                for(int pss = 0; pss < KER_COUT_UNIT4; pss++)
                {
                    int cout_idx = p + pss;
                    float* out_ptr = output + cout_idx * out_hw + idx_h[0] * TILE * out_w + idx_w[0] * TILE;
                    if(bias_term)
                    {
                        bias_ptr = ( float* )(bias + cout_idx);
                    }
                    // single_out(out_buffer+pss*ELEM_SIZE*BLOCK_HW_UNIT,
                    //             out_ptr,
                    //             out_w, (const float*)bias_ptr);
                    float ker00[4] = {2, 4, 8, 0};

                    tran_out_4(out_buffer + pss * ELEM_SIZE * BLOCK_HW_UNIT, out_ptr, out_w * sizeof(float), ker00,
                               bias_ptr, activation);
                }
                // #ifdef D_WINO
                // DumpFloat("pr_out_4",output,cout_end*out_hw,out_w,out_hw/out_w);
                // # endif
            }
            else
            {
                float buffer[KER_COUT_UNIT4 * BLOCK_HW_UNIT * ELEM_SIZE];
                float* buffer_ptr0 = buffer;
                for(int pp = 0; pp < KER_COUT_UNIT4; pp++)
                {
                    for(int t = 0; t < BLOCK_HW_UNIT; t++)
                    {
                        for(int ss = 0; ss < ELEM_SIZE; ss++)
                        {
                            *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT4 + pp * BLOCK_HW_UNIT + t];
                            buffer_ptr0++;
                        }
                    }
                }
                // end interleave
                // transform_output_f43_4tile((const float*)buffer, output, p, i, block_h, block_w, out_hw, out_w,
                // resi_h, resi_w,
                //                            KER_COUT_UNIT, bias, bias_term);
                {
                    float tmp_buffer[TILE * TILE];
                    const float* bias_ptr = NULL;
                    for(int pss = 0; pss < KER_COUT_UNIT4; pss++)
                    {
                        int cout_idx = p + pss;
                        float* out_ptr = output + cout_idx * out_hw;
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
                                trans_output_f43(buffer + ii * ELEM_SIZE + pss * 36 * 4,
                                                 out_ptr + (i_h * TILE * out_w + j_w * TILE), out_w,
                                                 ( const float* )bias_ptr, activation);
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
                                trans_output_f43_ordinary(buffer + ii * ELEM_SIZE + pss * 36 * 4, tmp_buffer,
                                                          ( const float* )bias_ptr);
                                float* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                                for(int hh = 0; hh < ret_h; hh++)
                                {
                                    for(int ww = 0; ww < ret_w; ww++)
                                    {
                                        out_pointer[hh * out_w + ww] =
                                            do_activation(tmp_buffer[hh * 4 + ww], activation);
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

            float out_buffer[KER_COUT_UNIT4 * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                wino_sgemm_1x4(out_buffer + s * KER_COUT_UNIT4, inp_ptr + s * cin, ker_ptr + s * KER_COUT_UNIT4 * cin,
                                cin);
            }
            // interleave
            float buffer[KER_COUT_UNIT4 * ELEM_SIZE];
            float* buffer_ptr0 = buffer;

            for(int pp = 0; pp < KER_COUT_UNIT4; pp++)
            {
                for(int ss = 0; ss < ELEM_SIZE; ss++)
                {
                    *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT4 + pp];
                    buffer_ptr0++;
                }
            }
            // end interleave
            transform_output_f43_1tile(( const float* )buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h,
                                       resi_w, KER_COUT_UNIT4, bias, bias_term, activation);
            // end transform
        }
    }
    for(p = (cout_end & -KER_COUT_UNIT4); p < cout_end; p ++)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;
        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;
            float buffer[BLOCK_HW_UNIT * ELEM_SIZE];
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

            //gemm+interleave buffer[4][36]
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                float* inp_ = (float*)(inp_ptr + s * BLOCK_HW_UNIT * cin);
                float* ker_ = (float*)(ker_ptr + s * cin);

                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
                for(int k = 0; k < cin; k++)
                {
                    sum0 += inp_[k * 4    ] * ker_[k];
                    sum1 += inp_[k * 4 + 1] * ker_[k];
                    sum2 += inp_[k * 4 + 2] * ker_[k];
                    sum3 += inp_[k * 4 + 3] * ker_[k];
                }
                buffer[      s] = sum0;
                buffer[ 36 + s] = sum1;
                buffer[ 72 + s] = sum2;
                buffer[108 + s] = sum3;
            }
            //trans_out buffer[4][36]
            float tmp_buffer[TILE * TILE];
            const float* bias_ptr = NULL;     
            float* out_ptr = output + p * out_hw;
            if(bias_term)
            {
                bias_ptr = bias + p;
            }
            for(int ii = 0; ii < BLOCK_HW_UNIT; ii++)
            {
                int i_h = idx_h[ii];
                int j_w = idx_w[ii];
                if((resi_h == 0 && resi_w == 0) || (resi_h == 0 && (j_w < block_w - 1)) ||
                    (resi_w == 0 && (i_h < block_h - 1)) || ((j_w < block_w - 1) && (i_h < block_h - 1)))
                {
                    trans_output_f43(buffer + ii * ELEM_SIZE,
                                        out_ptr + (i_h * TILE * out_w + j_w * TILE), out_w,
                                        ( const float* )bias_ptr, activation);
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
                    trans_output_f43_ordinary(buffer + ii * ELEM_SIZE, tmp_buffer,
                                                ( const float* )bias_ptr);
                    float* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                    for(int hh = 0; hh < ret_h; hh++)
                    {
                        for(int ww = 0; ww < ret_w; ww++)
                        {
                            out_pointer[hh * out_w + ww] =
                                do_activation(tmp_buffer[hh * 4 + ww], activation);
                        }
                    }

                }    // end else, tmp_buff
            }// end transform
        }

        for(; i < block_end; i++)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float buffer[ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                float* inp_ = (float*)(inp_ptr + s * cin);
                float* ker_ = (float*)(ker_ptr + s * cin);

                float sum = 0;
                for(int k = 0; k < cin; k++)
                {
                    sum += inp_[k] * ker_[k];
                }
                buffer[s] = sum;
            }
            // end interleave
            transform_output_f43_1tile(( const float* )buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h,
                                       resi_w, 1, bias, bias_term, activation);
            // end transform
        }
    }
}

// inp [block_hw//4][36][cin][4]
// ker [cout//16][36][cin][16]
void wino_sgemm_4x16(const float* ker, const float* inp, float* output, const float* bias, int bias_term,
                                   int cin, int cpu_type, int cout_start, int cout_end, int block_start, int block_end,
                                   int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                   int activation)
{
    int p, i;
    int flag_outw = 1;
    if(out_w < 16)
        flag_outw = 0;
    const float* ker_ptr;
    const float* inp_ptr;
    // #ifdef D_WINO
    //     printf("[SGEMM4X16]\tcout[%d-%d]\tblock[%d-%d]\tblock_h,w[%d,%d]\tresi_h,w[%d,%d]\n", cout_start, cout_end,
    //            block_start, block_end, block_h, block_w, resi_h, resi_w);
    // #endif
    for(p = (cout_start & -KER_COUT_UNIT); p < (cout_end & -KER_COUT_UNIT); p += KER_COUT_UNIT)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;

        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
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
            int wino_out_4_tiles = 0;
            int mulitplier = KER_COUT_UNIT;
            if(flag_outw)
                if((idx_h[0] == idx_h[3]) && (idx_h[0] < (block_h - 1)) && (idx_w[3] < (block_w - 1)))
                {
                    wino_out_4_tiles = 1;
                    mulitplier = 1;
                }

            for(int s = 0; s < ELEM_SIZE; s++)
            {
                {
                    wino_sgemm_4x16_A72(out_buffer + s * BLOCK_HW_UNIT * mulitplier, inp_ptr + s * BLOCK_HW_UNIT * cin,
                                        ker_ptr + s * KER_COUT_UNIT * cin, cin, wino_out_4_tiles);
                }

            }
            if(wino_out_4_tiles == 1)
            {
                // #ifdef D_WINO
                //                 printf("\t\t\t\t block_idx:%d,[idx_h:%d] [idx_w:%d-%d]\n", i, idx_h[0], idx_w[0], idx_w[3]);
                // #endif
                // #ifdef D_WINO
                // DumpFloat("pr_buffer",buffer,ELEM_SIZE*BLOCK_HW_UNIT*KER_COUT_UNIT,BLOCK_HW_UNIT*6,6);
                // #endif
                float* bias_ptr = NULL;
                for(int pss = 0; pss < KER_COUT_UNIT; pss++)
                {
                    int cout_idx = p + pss;
                    float* out_ptr = output + cout_idx * out_hw + idx_h[0] * TILE * out_w + idx_w[0] * TILE;
                    if(bias_term)
                    {
                        bias_ptr = ( float* )(bias + cout_idx);
                    }
                    // single_out(out_buffer+pss*ELEM_SIZE*BLOCK_HW_UNIT,
                    //             out_ptr,
                    //             out_w, (const float*)bias_ptr);
                    float ker00[4] = {2, 4, 8, 0};

                    tran_out_4(out_buffer + pss * ELEM_SIZE * BLOCK_HW_UNIT, out_ptr, out_w * sizeof(float), ker00,
                               bias_ptr, activation);
                }
                // #ifdef D_WINO
                // DumpFloat("pr_out_4",output,cout_end*out_hw,out_w,out_hw/out_w);
                // # endif
            }
            else
            {
                float buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
                float* buffer_ptr0 = buffer;
                for(int pp = 0; pp < KER_COUT_UNIT; pp++)
                {
                    for(int t = 0; t < BLOCK_HW_UNIT; t++)
                    {
                        for(int ss = 0; ss < ELEM_SIZE; ss++)
                        {
                            *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT + pp * BLOCK_HW_UNIT + t];
                            buffer_ptr0++;
                        }
                    }
                }
                // end interleave
                // transform_output_f43_4tile((const float*)buffer, output, p, i, block_h, block_w, out_hw, out_w,
                // resi_h, resi_w,
                //                            KER_COUT_UNIT, bias, bias_term);
                {
                    float tmp_buffer[TILE * TILE];
                    const float* bias_ptr = NULL;
                    for(int pss = 0; pss < KER_COUT_UNIT; pss++)
                    {
                        int cout_idx = p + pss;
                        float* out_ptr = output + cout_idx * out_hw;
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
                                trans_output_f43(buffer + ii * ELEM_SIZE + pss * 36 * 4,
                                                 out_ptr + (i_h * TILE * out_w + j_w * TILE), out_w,
                                                 ( const float* )bias_ptr, activation);
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
                                trans_output_f43_ordinary(buffer + ii * ELEM_SIZE + pss * 36 * 4, tmp_buffer,
                                                          ( const float* )bias_ptr);
                                float* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                                for(int hh = 0; hh < ret_h; hh++)
                                {
                                    for(int ww = 0; ww < ret_w; ww++)
                                    {
                                        out_pointer[hh * out_w + ww] =
                                            do_activation(tmp_buffer[hh * 4 + ww], activation);
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

            float out_buffer[KER_COUT_UNIT * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                wino_sgemm_1x16(out_buffer + s * KER_COUT_UNIT, inp_ptr + s * cin, ker_ptr + s * KER_COUT_UNIT * cin,
                                cin);
            }
            // interleave
            float buffer[KER_COUT_UNIT * ELEM_SIZE];
            float* buffer_ptr0 = buffer;

            for(int pp = 0; pp < KER_COUT_UNIT; pp++)
            {
                for(int ss = 0; ss < ELEM_SIZE; ss++)
                {
                    *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT + pp];
                    buffer_ptr0++;
                }
            }
            // end interleave
            transform_output_f43_1tile(( const float* )buffer, output, p, i, block_h, block_w, out_hw, out_w, resi_h,
                                       resi_w, KER_COUT_UNIT, bias, bias_term, activation);
            // end transform
        }
    }
}

// inp [36][block_hw//4][cin][4]
// ker [36][cout//16][cin][16]
// mid [cout//16]([block//4][36][16][4] + block_i[36][16]
void wino_sgemm_4x16_1(const float* ker, const float* inp, float* trans_out, int cin, int cpu_type,
                                     int cout_start, int cout_end, int block_start, int block_end, int block_hw,
                                     int outc, int s)
{
    int p, i;
    float* out_ptr;
    float* out_ptr1;

    for(p = (cout_start & -KER_COUT_UNIT); p < (cout_end & -KER_COUT_UNIT); p += KER_COUT_UNIT)
    {
        out_ptr = trans_out + p * ELEM_SIZE * block_hw;
        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            out_ptr1 = out_ptr + i * ELEM_SIZE * KER_COUT_UNIT;

            int offset = s * block_hw * cin + i * cin;
            int offset_ker = s * cin * outc + p * cin;
            
            {
                wino_sgemm_4x16_A72(out_ptr1 + s * BLOCK_HW_UNIT, inp + offset, ker + offset_ker, cin, 1);
            }

        }

        for(; i < block_end; i++)
        {
            out_ptr1 = out_ptr + i * ELEM_SIZE * KER_COUT_UNIT;

            int offset_ker = s * cin * outc + p * cin;
            int offset = s * block_hw * cin + i * cin;

            wino_sgemm_1x16(out_ptr1 + s * KER_COUT_UNIT, inp + offset, ker + offset_ker, cin);
        }
    }
}
void wino_sgemm_1x4_cpu(float* output, const float* input, const float* kernel, long cin)
{
    for(int i = 0; i < 4; i++)
    {
        float sum = 0;
        for(int k = 0; k < cin; k++)
        {
            sum += input[k] * kernel[k * 4 + i];
        }
        output[i] = sum;
    }
}
void wino_sgemm_1x1_cpu(float* output, const float* input, const float* kernel, long cin)
{
    float sum = 0;
    for(int k = 0; k < cin; k++)
    {
        sum += input[k] * kernel[k];
    }
    output[0] = sum;
}
void wino_sgemm_4x4_cpu_save1(float* output, const float* input, const float* kernel, long cin)
{
    for(int i = 0; i < KER_COUT_UNIT4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            float sum = 0;
            for(int k = 0; k < cin; k++)
            {
                sum += input[k * 4 + j] * kernel[k * KER_COUT_UNIT4 + i];
            }
            output[i *(36*4) + j] = sum;
        }
    }
}
void wino_sgemm_4x1_cpu_save1(float* output, const float* input, const float* kernel, long cin)
{
    for(int j = 0; j < 4; j++)
    {
        float sum = 0;
        for(int k = 0; k < cin; k++)
        {
            sum += input[k * 4 + j] * kernel[k];
        }
        output[j] = sum;
    }
    
}
void wino_sgemm_4x4_1(const float* ker, const float* inp, float* trans_out, int cin, int cpu_type,
                                     int cout_start, int cout_end, int block_start, int block_end, int block_hw,
                                     int outc, int s)
{
    int p, i;
    float* out_ptr;
    float* out_ptr1;
    for(p = (cout_start & -KER_COUT_UNIT4); p < (cout_end & -KER_COUT_UNIT4); p += KER_COUT_UNIT4)
    {
        out_ptr = trans_out + p * ELEM_SIZE * block_hw;
        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            out_ptr1 = out_ptr + i * ELEM_SIZE * KER_COUT_UNIT4;

            int offset = s * block_hw * cin + i * cin;
            int offset_ker = s * cin * outc + p * cin;
            wino_sgemm_4x4_A72(out_ptr1 + s * BLOCK_HW_UNIT, inp + offset, ker + offset_ker, cin, 1);
        }
        for(; i < block_end; i++)
        {
            out_ptr1 = out_ptr + i * ELEM_SIZE * KER_COUT_UNIT4;

            int offset_ker = s * cin * outc + p * cin;
            int offset = s * block_hw * cin + i * cin;

            wino_sgemm_1x4(out_ptr1 + s * KER_COUT_UNIT4, inp + offset, ker + offset_ker, cin);
        }
    }
    for(p = (cout_end & -KER_COUT_UNIT4); p < cout_end; p ++){
        out_ptr = trans_out + p * ELEM_SIZE * block_hw;
        float* ker_ = (float*)(ker + s * cin * outc + p * cin);
        for(i = (block_start & -4); i < (block_end & -4); i += 4){
            out_ptr1 = out_ptr + i * ELEM_SIZE + s*BLOCK_HW_UNIT;
            float* inp_ = (float*)(inp + s * block_hw * cin + i*cin);
            float sum0 = 0;
            float sum1 = 0;
            float sum2 = 0;
            float sum3 = 0;
            for(int k = 0; k < cin; k++){
                sum0 += inp_[k * 4    ] * ker_[k];
                sum1 += inp_[k * 4 + 1] * ker_[k];
                sum2 += inp_[k * 4 + 2] * ker_[k];
                sum3 += inp_[k * 4 + 3] * ker_[k];
            }
            out_ptr1[0] = sum0;
            out_ptr1[1] = sum1;
            out_ptr1[2] = sum2;
            out_ptr1[3] = sum3;    
        }
        for(; i < block_end; i++){
            out_ptr1 = out_ptr + i * ELEM_SIZE + s;
            float* inp_ = (float*)(inp + s * block_hw * cin + i*cin);
            float sum0 = 0;
            for(int k = 0; k < cin; k++){
                sum0 += inp_[k] * ker_[k];
            }
            out_ptr1[0] = sum0;
        }
    }
}
void wino_sgemm_4x16_nhwc(const float* ker, const float* inp, float* output, const float* bias,
                                        int bias_term, int cin, int cpu_type, int cout_start, int cout_end,
                                        int block_start, int block_end, int block_h, int block_w, int out_c, int out_w,
                                        int resi_h, int resi_w, int activation)
{
    int p, i;

    const float* ker_ptr;
    const float* inp_ptr;

    for(p = (cout_start & -KER_COUT_UNIT); p < (cout_end & -KER_COUT_UNIT); p += KER_COUT_UNIT)
    {
        ker_ptr = ker + p * ELEM_SIZE * cin;

        for(i = (block_start & -4); i < (block_end & -4); i += 4)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
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

            for(int s = 0; s < ELEM_SIZE; s++)
            {
                {
                    wino_sgemm_4x16_A72(out_buffer + s * BLOCK_HW_UNIT * KER_COUT_UNIT,
                                        inp_ptr + s * BLOCK_HW_UNIT * cin, ker_ptr + s * KER_COUT_UNIT * cin, cin, 0);
                }

            }
            {
                float buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
                float* buffer_ptr0 = buffer;
                for(int pp = 0; pp < KER_COUT_UNIT; pp++)
                {
                    for(int t = 0; t < BLOCK_HW_UNIT; t++)
                    {
                        for(int ss = 0; ss < ELEM_SIZE; ss++)
                        {
                            *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT + pp * BLOCK_HW_UNIT + t];
                            buffer_ptr0++;
                        }
                    }
                }
                // end interleave

                float tmp_buffer[TILE * TILE];
                const float* bias_ptr = NULL;
                for(int pss = 0; pss < KER_COUT_UNIT; pss++)
                {
                    int cout_idx = p + pss;
                    float* out_ptr = output + cout_idx;
                    if(bias_term)
                    {
                        bias_ptr = bias + cout_idx;
                    }
                    for(int ii = 0; ii < BLOCK_HW_UNIT; ii++)
                    {
                        int i_h = idx_h[ii];
                        int j_w = idx_w[ii];

                        int ret_h = TILE - resi_h;
                        if(i_h < block_h - 1)
                            ret_h = TILE;
                        int ret_w = TILE - resi_w;
                        if(j_w < block_w - 1)
                            ret_w = TILE;
                        // tmp_buffer
                        trans_output_f43_ordinary(buffer + ii * ELEM_SIZE + pss * ELEM_SIZE * BLOCK_HW_UNIT, tmp_buffer,
                                                  ( const float* )bias_ptr);
                        float* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE) * out_c;
                        for(int hh = 0; hh < ret_h; hh++)
                        {
                            for(int ww = 0; ww < ret_w; ww++)
                            {
                                out_pointer[(hh * out_w + ww) * out_c] =
                                    do_activation(tmp_buffer[hh * 4 + ww], activation);
                            }
                        }
                    }
                }
                // end transform
            }
        }

        for(; i < block_end; i++)
        {
            inp_ptr = inp + i * ELEM_SIZE * cin;

            float out_buffer[KER_COUT_UNIT * ELEM_SIZE];
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                wino_sgemm_1x16(out_buffer + s * KER_COUT_UNIT, inp_ptr + s * cin, ker_ptr + s * KER_COUT_UNIT * cin,
                                cin);
            }
            // interleave
            float buffer[KER_COUT_UNIT * ELEM_SIZE];
            float* buffer_ptr0 = buffer;

            for(int pp = 0; pp < KER_COUT_UNIT; pp++)
            {
                for(int ss = 0; ss < ELEM_SIZE; ss++)
                {
                    *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT + pp];
                    buffer_ptr0++;
                }
            }
            // end interleave
            transform_output_f43_1tile_nhwc(( const float* )buffer, output, p, i, block_h, block_w, out_c, out_w,
                                            resi_h, resi_w, KER_COUT_UNIT, bias, bias_term, activation);
            // end transform
        }
    }
}
void wino_sgemm_4x16_func(const float* ker, const float* inp, float* output, const float* bias, int bias_term,
                                   int cin, int cpu_type, int cout_start, int cout_end, int is_4block,int block_start,int resi,
                                   int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                   int activation)
{
    int p;
    int flag_outw = 1;
    if(out_w < 16)
        flag_outw = 0;
    const float* ker_ptr;
    const float* inp_ptr;
    int i=block_start;
    // printf("cin=%d cout[%d-%d],is_4block=%d, block_start=%d,block_hw[%d,%d],out_hw=%d,out_w=%d,resi_hw[%d,%d]\n", 
    //         cin,  cout_start,  cout_end,  is_4block, block_start,
    //         block_h,  block_w,  out_hw,  out_w,  resi_h,  resi_w );
    if(is_4block==1)
    {
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
        for(p = (cout_start & -KER_COUT_UNIT); p < (cout_end & -KER_COUT_UNIT); p += KER_COUT_UNIT)
        {
            ker_ptr = ker + p * ELEM_SIZE * cin;
            inp_ptr = inp;

            float out_buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];

            int wino_out_4_tiles = 0;
            int mulitplier = KER_COUT_UNIT;
            if(flag_outw)
                if((idx_h[0] == idx_h[3]) && (idx_h[0] < (block_h - 1)) && (idx_w[3] < (block_w - 1)))
                {
                    wino_out_4_tiles = 1;
                    mulitplier = 1;
                }

            for(int s = 0; s < ELEM_SIZE; s++)
            {
                {
                    wino_sgemm_4x16_A72(out_buffer + s * BLOCK_HW_UNIT * mulitplier, inp_ptr + s * BLOCK_HW_UNIT * cin,
                                        ker_ptr + s * KER_COUT_UNIT * cin, cin, wino_out_4_tiles);
                }

            }
            if(wino_out_4_tiles == 1)
            {
                float* bias_ptr = NULL;
                for(int pss = 0; pss < KER_COUT_UNIT; pss++)
                {
                    int cout_idx = p + pss;
                    float* out_ptr = output + cout_idx * out_hw + idx_h[0] * TILE * out_w + idx_w[0] * TILE;
                    if(bias_term)
                    {
                        bias_ptr = ( float* )(bias + cout_idx);
                    }
                    // single_out(out_buffer+pss*ELEM_SIZE*BLOCK_HW_UNIT,
                    //             out_ptr,
                    //             out_w, (const float*)bias_ptr);
                    float ker00[4] = {2, 4, 8, 0};

                    tran_out_4(out_buffer + pss * ELEM_SIZE * BLOCK_HW_UNIT, out_ptr, out_w * sizeof(float), ker00,
                               bias_ptr, activation);
                }
            }
            else
            {
                float buffer[KER_COUT_UNIT * BLOCK_HW_UNIT * ELEM_SIZE];
                float* buffer_ptr0 = buffer;
                for(int pp = 0; pp < KER_COUT_UNIT; pp++)
                {
                    for(int t = 0; t < BLOCK_HW_UNIT; t++)
                    {
                        for(int ss = 0; ss < ELEM_SIZE; ss++)
                        {
                            *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT + pp * BLOCK_HW_UNIT + t];
                            buffer_ptr0++;
                        }
                    }
                }
                // end interleave

                {
                    float tmp_buffer[TILE * TILE];
                    const float* bias_ptr = NULL;
                    for(int pss = 0; pss < KER_COUT_UNIT; pss++)
                    {
                        int cout_idx = p + pss;
                        float* out_ptr = output + cout_idx * out_hw;
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
                                trans_output_f43(buffer + ii * ELEM_SIZE + pss * 36 * 4,
                                                 out_ptr + (i_h * TILE * out_w + j_w * TILE), out_w,
                                                 ( const float* )bias_ptr, activation);
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
                                trans_output_f43_ordinary(buffer + ii * ELEM_SIZE + pss * 36 * 4, tmp_buffer,
                                                          ( const float* )bias_ptr);
                                float* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                                for(int hh = 0; hh < ret_h; hh++)
                                {
                                    for(int ww = 0; ww < ret_w; ww++)
                                    {
                                        out_pointer[hh * out_w + ww] =
                                            do_activation(tmp_buffer[hh * 4 + ww], activation);
                                    }
                                }

                            }    // end else, tmp_buff
                        }
                    }
                }
                // end transform
            }
        }
    }
    else
    {
        for(p = (cout_start & -KER_COUT_UNIT); p < (cout_end & -KER_COUT_UNIT); p += KER_COUT_UNIT)
        {
            ker_ptr = ker + p * ELEM_SIZE * cin;
            for(int bi=0;bi<resi;bi++)
            {
                inp_ptr = inp+bi*ELEM_SIZE*cin;

                float out_buffer[KER_COUT_UNIT * ELEM_SIZE];
                for(int s = 0; s < ELEM_SIZE; s++)
                {
                    wino_sgemm_1x16(out_buffer + s * KER_COUT_UNIT, inp_ptr + s * cin, ker_ptr + s * KER_COUT_UNIT * cin,
                                    cin);
                }
                // interleave
                float buffer[KER_COUT_UNIT * ELEM_SIZE];
                float* buffer_ptr0 = buffer;

                for(int pp = 0; pp < KER_COUT_UNIT; pp++)
                {
                    for(int ss = 0; ss < ELEM_SIZE; ss++)
                    {
                        *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT + pp];
                        buffer_ptr0++;
                    }
                }
                // end interleave
                transform_output_f43_1tile(( const float* )buffer, output, p, i+bi, block_h, block_w, out_hw, out_w, resi_h,
                                            resi_w, KER_COUT_UNIT, bias, bias_term, activation);
            }
        }
        // end transform
    }
    
}
void wino_sgemm_4x4_func(const float* ker, const float* inp, float* output, const float* bias, int bias_term,
                                   int cin, int cpu_type, int cout_start, int cout_end, int is_4block ,int block_start,int resi,
                                   int block_h, int block_w, int out_hw, int out_w, int resi_h, int resi_w,
                                   int activation)
{
    int p;
    int flag_outw = 1;
    if(out_w < 16)
        flag_outw = 0;
    const float* ker_ptr;
    const float* inp_ptr;
    int i=block_start;
    if(is_4block==1)
    {
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

        for(p = (cout_start & -KER_COUT_UNIT4); p < (cout_end & -KER_COUT_UNIT4); p += KER_COUT_UNIT4)
        {
            ker_ptr = ker + p * ELEM_SIZE * cin;
            inp_ptr = inp;

            float out_buffer[KER_COUT_UNIT4 * BLOCK_HW_UNIT * ELEM_SIZE];

            int wino_out_4_tiles = 0;
            int mulitplier = KER_COUT_UNIT4;
            if(flag_outw)
                if((idx_h[0] == idx_h[3]) && (idx_h[0] < (block_h - 1)) && (idx_w[3] < (block_w - 1)))
                {
                    wino_out_4_tiles = 1;
                    mulitplier = 1;
                }

            for(int s = 0; s < ELEM_SIZE; s++)
            {
                wino_sgemm_4x4_A72(out_buffer + s * BLOCK_HW_UNIT * mulitplier, inp_ptr + s * BLOCK_HW_UNIT * cin,
                                        ker_ptr + s * KER_COUT_UNIT4 * cin, cin, wino_out_4_tiles);
            }
            if(wino_out_4_tiles == 1)
            {
                // #ifdef D_WINO
                // DumpFloat("pr_buffer",buffer,ELEM_SIZE*BLOCK_HW_UNIT*KER_COUT_UNIT,BLOCK_HW_UNIT*6,6);
                // #endif
                float* bias_ptr = NULL;
                for(int pss = 0; pss < KER_COUT_UNIT4; pss++)
                {
                    int cout_idx = p + pss;
                    float* out_ptr = output + cout_idx * out_hw + idx_h[0] * TILE * out_w + idx_w[0] * TILE;
                    if(bias_term)
                    {
                        bias_ptr = ( float* )(bias + cout_idx);
                    }
                    // single_out(out_buffer+pss*ELEM_SIZE*BLOCK_HW_UNIT,
                    //             out_ptr,
                    //             out_w, (const float*)bias_ptr);
                    float ker00[4] = {2, 4, 8, 0};

                    tran_out_4(out_buffer + pss * ELEM_SIZE * BLOCK_HW_UNIT, out_ptr, out_w * sizeof(float), ker00,
                               bias_ptr, activation);
                }
                // #ifdef D_WINO
                // DumpFloat("pr_out_4",output,cout_end*out_hw,out_w,out_hw/out_w);
                // # endif
            }
            else
            {
                float buffer[KER_COUT_UNIT4 * BLOCK_HW_UNIT * ELEM_SIZE];
                float* buffer_ptr0 = buffer;
                for(int pp = 0; pp < KER_COUT_UNIT4; pp++)
                {
                    for(int t = 0; t < BLOCK_HW_UNIT; t++)
                    {
                        for(int ss = 0; ss < ELEM_SIZE; ss++)
                        {
                            *buffer_ptr0 = out_buffer[ss * BLOCK_HW_UNIT * KER_COUT_UNIT4 + pp * BLOCK_HW_UNIT + t];
                            buffer_ptr0++;
                        }
                    }
                }
                // end interleave
                // transform_output_f43_4tile((const float*)buffer, output, p, i, block_h, block_w, out_hw, out_w,
                // resi_h, resi_w,
                //                            KER_COUT_UNIT, bias, bias_term);
                {
                    float tmp_buffer[TILE * TILE];
                    const float* bias_ptr = NULL;
                    for(int pss = 0; pss < KER_COUT_UNIT4; pss++)
                    {
                        int cout_idx = p + pss;
                        float* out_ptr = output + cout_idx * out_hw;
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
                                trans_output_f43(buffer + ii * ELEM_SIZE + pss * 36 * 4,
                                                 out_ptr + (i_h * TILE * out_w + j_w * TILE), out_w,
                                                 ( const float* )bias_ptr, activation);
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
                                trans_output_f43_ordinary(buffer + ii * ELEM_SIZE + pss * 36 * 4, tmp_buffer,
                                                          ( const float* )bias_ptr);
                                float* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                                for(int hh = 0; hh < ret_h; hh++)
                                {
                                    for(int ww = 0; ww < ret_w; ww++)
                                    {
                                        out_pointer[hh * out_w + ww] =
                                            do_activation(tmp_buffer[hh * 4 + ww], activation);
                                    }
                                }

                            }    // end else, tmp_buff
                        }
                    }
                }
                // end transform
            }
        }
        for(p = (cout_end & -KER_COUT_UNIT4); p < cout_end; p ++)
        {
            ker_ptr = ker + p * ELEM_SIZE * cin;
            inp_ptr = inp;
            
            float buffer[BLOCK_HW_UNIT * ELEM_SIZE];
            //gemm+interleave buffer[4][36]
            for(int s = 0; s < ELEM_SIZE; s++)
            {
                float* inp_ = (float*)(inp_ptr + s * BLOCK_HW_UNIT * cin);
                float* ker_ = (float*)(ker_ptr + s * cin);

                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
                for(int k = 0; k < cin; k++)
                {
                    sum0 += inp_[k * 4    ] * ker_[k];
                    sum1 += inp_[k * 4 + 1] * ker_[k];
                    sum2 += inp_[k * 4 + 2] * ker_[k];
                    sum3 += inp_[k * 4 + 3] * ker_[k];
                }
                buffer[      s] = sum0;
                buffer[ 36 + s] = sum1;
                buffer[ 72 + s] = sum2;
                buffer[108 + s] = sum3;
            }
            //trans_out buffer[4][36]
            float tmp_buffer[TILE * TILE];
            const float* bias_ptr = NULL;     
            float* out_ptr = output + p * out_hw;
            if(bias_term)
            {
                bias_ptr = bias + p;
            }
            for(int ii = 0; ii < BLOCK_HW_UNIT; ii++)
            {
                int i_h = idx_h[ii];
                int j_w = idx_w[ii];
                if((resi_h == 0 && resi_w == 0) || (resi_h == 0 && (j_w < block_w - 1)) ||
                    (resi_w == 0 && (i_h < block_h - 1)) || ((j_w < block_w - 1) && (i_h < block_h - 1)))
                {
                    trans_output_f43(buffer + ii * ELEM_SIZE,
                                        out_ptr + (i_h * TILE * out_w + j_w * TILE), out_w,
                                        ( const float* )bias_ptr, activation);
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
                    trans_output_f43_ordinary(buffer + ii * ELEM_SIZE, tmp_buffer,
                                                ( const float* )bias_ptr);
                    float* out_pointer = out_ptr + (i_h * TILE * out_w + j_w * TILE);
                    for(int hh = 0; hh < ret_h; hh++)
                    {
                        for(int ww = 0; ww < ret_w; ww++)
                        {
                            out_pointer[hh * out_w + ww] =
                                do_activation(tmp_buffer[hh * 4 + ww], activation);
                        }
                    }

                }    // end else, tmp_buff
            }// end transform
        }
    }
    else
    {
        for(p = (cout_start & -KER_COUT_UNIT4); p < (cout_end & -KER_COUT_UNIT4); p += KER_COUT_UNIT4)
        {
            ker_ptr = ker + p * ELEM_SIZE * cin;
            for(int bi=0;bi<resi;bi++)
            {
                inp_ptr = inp+bi*ELEM_SIZE*cin;

                float out_buffer[KER_COUT_UNIT4 * ELEM_SIZE];
                for(int s = 0; s < ELEM_SIZE; s++)
                {
                    wino_sgemm_1x4(out_buffer + s * KER_COUT_UNIT4, inp_ptr + s * cin, ker_ptr + s * KER_COUT_UNIT4 * cin,
                                    cin);
                }
                // interleave
                float buffer[KER_COUT_UNIT4 * ELEM_SIZE];
                float* buffer_ptr0 = buffer;

                for(int pp = 0; pp < KER_COUT_UNIT4; pp++)
                {
                    for(int ss = 0; ss < ELEM_SIZE; ss++)
                    {
                        *buffer_ptr0 = out_buffer[ss * KER_COUT_UNIT4 + pp];
                        buffer_ptr0++;
                    }
                }
                // end interleave
                transform_output_f43_1tile(( const float* )buffer, output, p, i+bi, block_h, block_w, out_hw, out_w, resi_h,
                                        resi_w, KER_COUT_UNIT4, bias, bias_term, activation);
                // end transform
            }
        }
        for(p = (cout_end & -KER_COUT_UNIT4); p < cout_end; p ++)
        {
            ker_ptr = ker + p * ELEM_SIZE * cin;
            for(int bi=0;bi<resi;bi++)
            {
                inp_ptr = inp+bi*ELEM_SIZE*cin;

                float buffer[ELEM_SIZE];
                for(int s = 0; s < ELEM_SIZE; s++)
                {
                    float* inp_ = (float*)(inp_ptr + s * cin);
                    float* ker_ = (float*)(ker_ptr + s * cin);

                    float sum = 0;
                    for(int k = 0; k < cin; k++)
                    {
                        sum += inp_[k] * ker_[k];
                    }
                    buffer[s] = sum;
                }
                // end interleave
                transform_output_f43_1tile(( const float* )buffer, output, p, i+bi, block_h, block_w, out_hw, out_w, resi_h,
                                        resi_w, 1, bias, bias_term, activation);
                // end transform
            }
        }
    }
}


// pour debug
void wino_sgemm_4x16_cpu(float* output, const float* input, const float* kernel, long cin)
{
    for(int i = 0; i < KER_COUT_UNIT; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            float sum = 0;
            for(int k = 0; k < cin; k++)
            {
                sum += input[k * 4 + j] * kernel[k * KER_COUT_UNIT + i];
            }
            output[i * 4 + j] = sum;
        }
    }
}

void wino_sgemm_1x16_cpu(float* output, const float* input, const float* kernel, long cin)
{
    for(int i = 0; i < KER_COUT_UNIT; i++)
    {
        float sum = 0;
        for(int k = 0; k < cin; k++)
        {
            sum += input[k] * kernel[k * 16 + i];
        }
        output[i] = sum;
    }
}

// pour debug trans_out_4.S
void single_out(float* mid, float* out__, int outw, const float* bias_ptr)
{
    float tmp[24] = {0};
    float r1_add_r2[6];
    float r1_minus_r2[6];
    float r3_add_r4[6];
    float r3_minus_r4_x2[6];

    for(int ii = 0; ii < 4; ii++)
    {
        // float* mid = mid0+ii;
        float* out = out__ + ii * TILE;
        for(int j = 0; j < 6; j++)
        {
            r1_add_r2[j] = mid[24 * 1 + j * 4 + ii] + mid[24 * 2 + j * 4 + ii];
            r1_minus_r2[j] = mid[24 * 1 + j * 4 + ii] - mid[24 * 2 + j * 4 + ii];
            r3_add_r4[j] = mid[24 * 3 + j * 4 + ii] + mid[24 * 4 + j * 4 + ii];
            r3_minus_r4_x2[j] = (mid[24 * 3 + j * 4 + ii] - mid[24 * 4 + j * 4 + ii]) * 2;
        }
        for(int j = 0; j < 6; j++)
        {
            tmp[j] = mid[j * 4 + ii] + r1_add_r2[j] + r3_add_r4[j];
            tmp[6 + j] = r1_minus_r2[j] + r3_minus_r4_x2[j];
            tmp[12 + j] = r1_add_r2[j] + 4 * r3_add_r4[j];
            tmp[18 + j] = r1_minus_r2[j] + 4 * r3_minus_r4_x2[j] + mid[24 * 5 + j * 4 + ii];
        }

        float* out0 = out;
        float* out1 = out0 + outw;
        float* out2 = out1 + outw;
        float* out3 = out2 + outw;

        float _r1_add_r2[4];
        float _r1_minus_r2[4];
        float _r3_add_r4[4];
        float _r3_minus_r4_x2[4];
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
            float bias = bias_ptr[0];
            out0[0] = tmp[0 * 6] + _r1_add_r2[0] + _r3_add_r4[0] + bias;
            out1[0] = tmp[1 * 6] + _r1_add_r2[1] + _r3_add_r4[1] + bias;
            out2[0] = tmp[2 * 6] + _r1_add_r2[2] + _r3_add_r4[2] + bias;
            out3[0] = tmp[3 * 6] + _r1_add_r2[3] + _r3_add_r4[3] + bias;

            out0[1] = _r1_minus_r2[0] + _r3_minus_r4_x2[0] + bias;
            out1[1] = _r1_minus_r2[1] + _r3_minus_r4_x2[1] + bias;
            out2[1] = _r1_minus_r2[2] + _r3_minus_r4_x2[2] + bias;
            out3[1] = _r1_minus_r2[3] + _r3_minus_r4_x2[3] + bias;

            out0[2] = _r1_add_r2[0] + 4 * _r3_add_r4[0] + bias;
            out1[2] = _r1_add_r2[1] + 4 * _r3_add_r4[1] + bias;
            out2[2] = _r1_add_r2[2] + 4 * _r3_add_r4[2] + bias;
            out3[2] = _r1_add_r2[3] + 4 * _r3_add_r4[3] + bias;

            out0[3] = _r1_minus_r2[0] + 4 * _r3_minus_r4_x2[0] + tmp[0 * 6 + 5] + bias;
            out1[3] = _r1_minus_r2[1] + 4 * _r3_minus_r4_x2[1] + tmp[1 * 6 + 5] + bias;
            out2[3] = _r1_minus_r2[2] + 4 * _r3_minus_r4_x2[2] + tmp[2 * 6 + 5] + bias;
            out3[3] = _r1_minus_r2[3] + 4 * _r3_minus_r4_x2[3] + tmp[3 * 6 + 5] + bias;
        }
        else
        {
            out0[0] = tmp[0 * 6] + _r1_add_r2[0] + _r3_add_r4[0];
            out1[0] = tmp[1 * 6] + _r1_add_r2[1] + _r3_add_r4[1];
            out2[0] = tmp[2 * 6] + _r1_add_r2[2] + _r3_add_r4[2];
            out3[0] = tmp[3 * 6] + _r1_add_r2[3] + _r3_add_r4[3];

            out0[1] = _r1_minus_r2[0] + _r3_minus_r4_x2[0];
            out1[1] = _r1_minus_r2[1] + _r3_minus_r4_x2[1];
            out2[1] = _r1_minus_r2[2] + _r3_minus_r4_x2[2];
            out3[1] = _r1_minus_r2[3] + _r3_minus_r4_x2[3];

            out0[2] = _r1_add_r2[0] + 4 * _r3_add_r4[0];
            out1[2] = _r1_add_r2[1] + 4 * _r3_add_r4[1];
            out2[2] = _r1_add_r2[2] + 4 * _r3_add_r4[2];
            out3[2] = _r1_add_r2[3] + 4 * _r3_add_r4[3];

            out0[3] = _r1_minus_r2[0] + 4 * _r3_minus_r4_x2[0] + tmp[0 * 6 + 5];
            out1[3] = _r1_minus_r2[1] + 4 * _r3_minus_r4_x2[1] + tmp[1 * 6 + 5];
            out2[3] = _r1_minus_r2[2] + 4 * _r3_minus_r4_x2[2] + tmp[2 * 6 + 5];
            out3[3] = _r1_minus_r2[3] + 4 * _r3_minus_r4_x2[3] + tmp[3 * 6 + 5];
        }
    }
}


