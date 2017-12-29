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
 * Author: chunyinglv@openailab.com
 */
#ifndef __POOLING_KERNEL_H__
#define __POOLING_KERNEL_H__

#include<arm_neon.h>

/**
* MaxPool_2x2: pooling for ksize=2x2,stride=2, pad=0(default pad=0)
* @param[in]    input     input data (const float pointer)
* @param[in]    output    output data (float pointer)
* @param[in]    inc       input channel (int)
* @param[in]    inh       input height (int)
* @param[in]    inw       input width (int)
* @param[in]    outh      output height (int)
* @param[in]    outw      output width (int)
* @param[in]    inc       input channel (int)
* @param[in]    htail     htail=(inh-ksize_h)%stride_h (int)
* @param[in]    wtail     wtail=(inw-ksize_w)%stride_w (int)
* @return		None
*/
static void MaxPool_2x2s2(const float*input,float* output,
    int inc,int inh,int inw,
    int outh,int outw,
    int htail,int wtail)
{
    int in_hw=inw*inh;
    int out_hw=outh*outw;
    int block_w=outw >>2;
    int tail_w=outw & ~3;
    
    int wtail_resi=inw+1;

    if(wtail)
    {
        if (outw%4==0)
        {
            block_w-=1;
            tail_w-=4;
        }
        outw-=1;
    }
    if(htail)
    {
        outh-=1;
    }

    for(int c=0;c<inc;c++)
    {
        const float* line0=input +c*in_hw;
        const float* line1=line0 + inw;
        float* out_ptr=output + c*out_hw;
        for(int i=0;i<outh;i++)
        {
            for(int j=0;j<block_w;j++)
            {
                float32x4_t p00=vld1q_f32(line0);
                float32x4_t p10=vld1q_f32(line1);
                float32x4_t max0=vmaxq_f32(p00,p10);

                float32x4_t p01=vld1q_f32(line0+4);
                float32x4_t p11=vld1q_f32(line1+4);
                float32x4_t max1=vmaxq_f32(p01,p11);
                /* pairwaise max */
                float32x4_t _max=vpmaxq_f32(max0,max1);
                vst1q_f32(out_ptr,_max);
                line0+=8;
                line1+=8;
                out_ptr+=4;
            }
            for(int j=tail_w;j<outw;j++)
            {
                float32x2_t p1=vld1_f32(line0);
                float32x2_t p2=vld1_f32(line1);    
                float32x2_t _max=vmax_f32(p1,p2);
                *out_ptr=std::max(_max[0],_max[1]);
                out_ptr++;
                line0+=2;
                line1+=2;
            }
            if(wtail)
            {
                *out_ptr =std::max (line0[0],line1[0]);
                out_ptr++;
                line0+=wtail_resi;
                line1+=wtail_resi;
            }
            else
            {
                line0+=inw;
                line1+=inw;
            }
        }
        if(htail)
        {
            for(int j=0;j<block_w;j++)
            {
                float32x4_t p00=vld1q_f32(line0);
                float32x4_t p01=vld1q_f32(line0+4);
   
                p00=vpmaxq_f32(p00,p01);
                vst1q_f32(out_ptr,p00);
                line0+=8;
                out_ptr+=4;
            }
            for(int j=tail_w;j<outw;j++)
            {
                float32x2_t p1=vld1_f32(line0); 
                *out_ptr=std::max(p1[0],p1[1]);
                out_ptr++;
                line0+=2;
            }
            if(wtail)
            {
                *out_ptr =line0[0];
                out_ptr++;
            }
        }
    }
}

static void AvgPool_2x2s2(const float*input,float* output,
    int inc,int inh,int inw,
    int outh,int outw,
    int htail,int wtail)
{
    int in_hw=inw*inh;
    int out_hw=outh*outw;
    int block_w=outw >>2;
    int tail_w=outw & ~3;

    int wtail_resi=inw+1;

    if(wtail)
    {
        if (outw%4==0)
        {
            block_w-=1;
            tail_w-=4;
        }
        outw-=1;
    }
    if(htail)
    {
        outh-=1;
    }
    for(int c=0;c<inc;c++)
    {
        const float* line0=input +c*in_hw;
        const float* line1=line0 + inw;
        float* out_ptr=output + c*out_hw;
        for(int i=0;i<outh;i++)
        {
            for(int j=0;j<block_w;j++)
            {
                float32x4_t p00=vld1q_f32(line0);
                float32x4_t p10=vld1q_f32(line1);
                float32x4_t sum0=vaddq_f32(p00,p10);

                float32x4_t p01=vld1q_f32(line0+4);
                float32x4_t p11=vld1q_f32(line1+4);
                float32x4_t sum1=vaddq_f32(p01,p11);
            
                sum0=vpaddq_f32(sum0,sum1);
                sum0=vmulq_n_f32(sum0, 0.25f);
                vst1q_f32(out_ptr,sum0);
                line0+=8;
                line1+=8;
                out_ptr+=4;
            }
            for(int j=tail_w;j<outw;j++)
            {
                float32x2_t p1=vld1_f32(line0);
                float32x2_t p2=vld1_f32(line1);    
                float32x2_t sum=vadd_f32(p1,p2);

                *out_ptr=(sum[0]+sum[1])*0.25;
                out_ptr++;
                line0+=2;
                line1+=2;
            }
            if(wtail)
            {
                *out_ptr = (line0[0] +line1[0])*0.5f;
                out_ptr++;
                line0+=wtail_resi;
                line1+=wtail_resi;
            }
            else
            {
                line0+=inw;
                line1+=inw;
            }

        }
        if(htail)
        {
            for(int j=0;j<block_w;j++)
            {
                float32x4_t p00=vld1q_f32(line0);
                float32x4_t p01=vld1q_f32(line0+4);
            
                p00=vpaddq_f32(p00,p01);
                p00=vmulq_n_f32(p00, 0.5f);
                vst1q_f32(out_ptr,p00);
                line0+=8;
                out_ptr+=4;
            }
            for(int j=tail_w;j<outw;j++)
            {
                float32x2_t p1=vld1_f32(line0);   
                *out_ptr=(p1[0]+p1[1])*0.5;
                out_ptr++;
                line0+=2;
            }
            if(wtail)
            {
                *out_ptr = line0[0];
                out_ptr++;
            }

        }
    }
}

static void MaxPool_3x3s2(const float*input,float* output,
    int inc,int inh,int inw,
    int outh,int outw,
    int htail,int wtail)
{
    int in_hw=inw*inh;
    int out_hw=outh*outw;
    int block_w=outw >>2;
    int tail_w=outw & ~3;

    int wtail_resi=inw+2;
    int w_resi=inw+1;
    if(wtail)
    {
        if (outw%4==0)
        {
            block_w-=1;
            tail_w-=4;
        }
        outw-=1;
    }
    if(htail)
    {
        outh-=1;
    }

    for(int c=0;c<inc;c++)
    {
        const float* line0=input +c*in_hw;
        const float* line1=line0 + inw;
        const float* line2=line1 + inw;
        float* out_ptr=output + c*out_hw;
        for(int i=0;i<outh;i++)
        {
            float32x4x2_t p00=vld2q_f32(line0);
            float32x4x2_t p10=vld2q_f32(line1);
            float32x4x2_t p20=vld2q_f32(line2);
            for(int j=0;j<block_w;j++)
            {
                /*
                p00     = [1,2,3,4,5,6,7,8]
                p00.val[0]=[1,3,5,7]
                
                max0    = [2,4,6,8]
                p00_new = [9,10,11,12,13,14,15,16]
                p01     = [3,5,7,9]
                max0=max(max0,p01)=[3,5,7,9]
                */
                float32x4x2_t p00_new=vld2q_f32(line0+8);
                float32x4_t max0=vmaxq_f32(p00.val[0],p00.val[1]);
                float32x4_t p01=vextq_f32(p00.val[0],p00_new.val[0],1);
                max0=vmaxq_f32(max0,p01);

                float32x4x2_t p10_new=vld2q_f32(line1+8);
                float32x4_t max1=vmaxq_f32(p10.val[0],p10.val[1]);
                float32x4_t p11=vextq_f32(p10.val[0],p10_new.val[0],1);
                max1=vmaxq_f32(max1,p11);

                float32x4x2_t p20_new=vld2q_f32(line2+8);
                float32x4_t max2=vmaxq_f32(p20.val[0],p20.val[1]);
                float32x4_t p21=vextq_f32(p20.val[0],p20_new.val[0],1);
                max2=vmaxq_f32(max2,p21);

                max0 = vmaxq_f32(vmaxq_f32(max0, max1), max2);
                vst1q_f32(out_ptr,max0);

                p00=p00_new;
                p10=p10_new;
                p20=p20_new;

                line0+=8;
                line1+=8;
                line2+=8;
                out_ptr+=4;
            }
            for(int j=tail_w;j<outw;j++)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
                float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
                float max2 = std::max(std::max(line2[0], line2[1]), line2[2]);
                *out_ptr = std::max(std::max(max0, max1), max2);
                
                out_ptr++;
                line0+=2;
                line1+=2;
                line2+=2;
            }
            if(wtail)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), 
                                      std::max(line1[0], line1[1]));
                *out_ptr = std::max(std::max(line2[0], line2[1]), max0);
                out_ptr++;
                line0+=wtail_resi;
                line1+=wtail_resi;
                line2+=wtail_resi;
            }
            else
            {
                line0+=w_resi;
                line1+=w_resi;
                line2+=w_resi;
            }

        }
        if(htail)
        {
            float32x4x2_t p00=vld2q_f32(line0);
            float32x4x2_t p10=vld2q_f32(line1);
            for(int j=0;j<block_w;j++)
            {
                float32x4x2_t p00_new=vld2q_f32(line0+8);
                float32x4_t max0=vmaxq_f32(p00.val[0],p00.val[1]);
                float32x4_t p01=vextq_f32(p00.val[0],p00_new.val[0],1);
                max0=vmaxq_f32(max0,p01);

                float32x4x2_t p10_new=vld2q_f32(line1+8);
                float32x4_t max1=vmaxq_f32(p10.val[0],p10.val[1]);
                float32x4_t p11=vextq_f32(p10.val[0],p10_new.val[0],1);
                max1=vmaxq_f32(max1,p11);
                
                vst1q_f32(out_ptr,vmaxq_f32(max0, max1));

                p00=p00_new;
                p10=p10_new;

                line0+=8;
                line1+=8;
                out_ptr+=4;
            }
            for(int j=tail_w;j<outw;j++)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
                float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
    
                *out_ptr = std::max(max0, max1);
                
                out_ptr++;
                line0+=2;
                line1+=2;
            }
            if(wtail)
            {
                *out_ptr = std::max(std::max(line0[0], line0[1]), 
                                      std::max(line1[0], line1[1]));
                out_ptr++;
            }
        }
    }
}

static void AvgPool_3x3s2(const float*input,float* output,
    int inc,int inh,int inw,
    int outh,int outw,
    int htail,int wtail)
{
    int in_hw=inw*inh;
    int out_hw=outh*outw;
    int block_w=outw >>2;
    int tail_w=outw & ~3;

    int wtail_resi=inw+2;
    int w_resi=inw+1;
    if(wtail)
    {
        if (outw%4==0)
        {
            block_w-=1;
            tail_w-=4;
        }
        outw-=1;
    }
    if(htail)
    {
        outh-=1;
    }
    for(int c=0;c<inc;c++)
    {
        const float* line0=input +c*in_hw;
        const float* line1=line0 + inw;
        const float* line2=line1 + inw;
        float* out_ptr=output + c*out_hw;
        for(int i=0;i<outh;i++)
        {
            float32x4x2_t p00=vld2q_f32(line0);
            float32x4x2_t p10=vld2q_f32(line1);
            float32x4x2_t p20=vld2q_f32(line2);
            for(int j=0;j<block_w;j++)
            {
                
                float32x4x2_t p00_new=vld2q_f32(line0+8);
                float32x4_t sum0=vaddq_f32(p00.val[0],p00.val[1]);
                float32x4_t p01=vextq_f32(p00.val[0],p00_new.val[0],1);
                sum0=vaddq_f32(sum0,p01);

                float32x4x2_t p10_new=vld2q_f32(line1+8);
                float32x4_t sum1=vaddq_f32(p10.val[0],p10.val[1]);
                float32x4_t p11=vextq_f32(p10.val[0],p10_new.val[0],1);
                sum1=vaddq_f32(sum1,p11);

                float32x4x2_t p20_new=vld2q_f32(line2+8);
                float32x4_t sum2=vaddq_f32(p20.val[0],p20.val[1]);
                float32x4_t p21=vextq_f32(p20.val[0],p20_new.val[0],1);
                sum2=vaddq_f32(sum2,p21);

                sum0 = vaddq_f32(vaddq_f32(sum0, sum1), sum2);
                sum0=vmulq_n_f32(sum0, 0.11111111f);
                vst1q_f32(out_ptr,sum0);

                p00=p00_new;
                p10=p10_new;
                p20=p20_new;

                line0+=8;
                line1+=8;
                line2+=8;
                out_ptr+=4;
            }
            for(int j=tail_w;j<outw;j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2] + 
                             line1[0] + line1[1] + line1[2] +
                             line2[0] + line2[1] + line2[2])*0.11111111f;
                out_ptr++;
                line0+=2;
                line1+=2;
                line2+=2;
            }
            if(wtail)
            {
                *out_ptr = (line0[0] + line0[1] +
                             line1[0] + line1[1] +
                             line2[0] + line2[1])*0.16666667f;
                out_ptr++;
                line0+=wtail_resi;
                line1+=wtail_resi;
                line2+=wtail_resi;
            }
            else
            {
                line0+=w_resi;
                line1+=w_resi;
                line2+=w_resi;
            }

        }
        if(htail)
        {
            float32x4x2_t p00=vld2q_f32(line0);
            float32x4x2_t p10=vld2q_f32(line1);
            for(int j=0;j<block_w;j++)
            {
                float32x4x2_t p00_new=vld2q_f32(line0+8);
                float32x4_t sum0=vaddq_f32(p00.val[0],p00.val[1]);
                float32x4_t p01=vextq_f32(p00.val[0],p00_new.val[0],1);
                sum0=vaddq_f32(sum0,p01);

                float32x4x2_t p10_new=vld2q_f32(line1+8);
                float32x4_t sum1=vaddq_f32(p10.val[0],p10.val[1]);
                float32x4_t p11=vextq_f32(p10.val[0],p10_new.val[0],1);
                sum1=vaddq_f32(sum1,p11);

                sum0 = vaddq_f32(sum0, sum1);
                sum0=vmulq_n_f32(sum0, 0.16666667f);
                vst1q_f32(out_ptr,sum0);

                p00=p00_new;
                p10=p10_new;
    
                line0+=8;
                line1+=8;
                out_ptr+=4;
            }
            for(int j=tail_w;j<outw;j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2] + 
                             line1[0] + line1[1] + line1[2])*0.16666667f;
                out_ptr++;
                line0+=2;
                line1+=2;
            }
            if(wtail)
            {
                *out_ptr = (line0[0] + line0[1] +
                             line1[0] + line1[1] )*0.25f;
                out_ptr++;

            }
        }
    }
}

static void MaxPool_2x2s2_pad1(const float*input,float* output,
    int inc,int inh,int inw,
    int outh,int outw)
{
    int in_hw=inw*inh;
    int out_hw=outh*outw;

    int wtail_resi=inw+1;


    for(int c=0;c<inc;c++)
    {
        const float* line00=input +c*in_hw;
        float* out_ptr=output + c*out_hw;
        //h begin
        *out_ptr = line00[0];
        out_ptr++;
        for(int i=1;i<inw-1;i+=2)
        {
            *out_ptr=std::max(line00[i],line00[i+1]);
            out_ptr++;
        }
        *out_ptr = line00[inw-1];
        out_ptr++;
        // h center
        const float* line0=line00 + inw;
        const float* line1=line0 + inw;
        for(int i=1;i<outh-1;i++)
        {
            //w begin
            *out_ptr=std::max(line0[0],line1[0]);
            out_ptr++;
            line0++;
            line1++;
            // w center
            for(int j=1;j<outw-1;j++)
            {
                float32x2_t p1=vld1_f32(line0);
                float32x2_t p2=vld1_f32(line1);    
                float32x2_t _max=vmax_f32(p1,p2);
                *out_ptr=std::max(_max[0],_max[1]);
                out_ptr++;
                line0+=2;
                line1+=2;
            }
            // w end
            *out_ptr =std::max (line0[0],line1[0]);
            out_ptr++;
            line0+=wtail_resi;
            line1+=wtail_resi;
        }
     
        // h end
        *out_ptr = line0[0];
        out_ptr++;
        for(int i=1;i<inw-1;i+=2)
        {
            *out_ptr=std::max(line0[i],line0[i+1]);
            out_ptr++;
        }
        *out_ptr = line0[inw-1];
        out_ptr++;
    }
}

static void AvgPool_2x2s2_pad1(const float*input,float* output,
    int inc,int inh,int inw,
    int outh,int outw)
{
    int in_hw=inw*inh;
    int out_hw=outh*outw;

    int wtail_resi=inw+1;


    for(int c=0;c<inc;c++)
    {
        const float* line00=input +c*in_hw;
        float* out_ptr=output + c*out_hw;
        //h begin
        *out_ptr = line00[0]*0.25;
        out_ptr++;
        for(int i=1;i<inw-1;i+=2)
        {
            *out_ptr=(line00[i]+line00[i+1])*0.25;
            out_ptr++;
        }
        *out_ptr = line00[inw-1]*0.25;
        out_ptr++;
        // h center
        const float* line0=line00 + inw;
        const float* line1=line0 + inw;
        for(int i=1;i<outh-1;i++)
        {
            //w begin
            *out_ptr=(line0[0]+line1[0])*0.25;
            out_ptr++;
            line0++;
            line1++;
            // w center
            for(int j=1;j<outw-1;j++)
            {
                float32x2_t p1=vld1_f32(line0);
                float32x2_t p2=vld1_f32(line1);    
                float32x2_t sum=vadd_f32(p1,p2);
                *out_ptr=(sum[0]+sum[1])*0.25;
                out_ptr++;
                line0+=2;
                line1+=2;
            }
            // w end
            *out_ptr =(line0[0]+line1[0])*0.25;
            out_ptr++;
            line0+=wtail_resi;
            line1+=wtail_resi;
        }
     
        // h end
        *out_ptr = line0[0]*0.25;
        out_ptr++;
        for(int i=1;i<inw-1;i+=2)
        {
            *out_ptr=(line0[i]+line0[i+1])*0.25;
            out_ptr++;
        }
        *out_ptr = line0[inw-1]*0.25;
        out_ptr++;
    }
}

static void MaxPool_3x3s2_pad1(const float*input,float* output,
    int inc,int inh,int inw,
    int outh,int outw,
    int htail,int wtail)
{
    int in_hw=inw*inh;
    int out_hw=outh*outw;

    int mid_w = (inw-3+wtail)/2;
    int mid_h = (inh-3+htail)/2;

    int inw_2=inw+2;
    int inw_1=inw+1;

    for(int c=0;c<inc;c++)
    {
        const float* line1=input +c*in_hw;
        const float* line2=line1 + inw;
        float* out_ptr=output + c*out_hw;
        
        //h begin ---------------------------------------
        *out_ptr =std::max( std::max(line1[0],line1[1]),
                            std::max(line2[0],line2[1]));
        out_ptr++;
        line1+=1; 
        line2+=1;

        for(int j=0;j<mid_w;j++)
        {
            //float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
            float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
            float max2 = std::max(std::max(line2[0], line2[1]), line2[2]);
            //*out_ptr = std::max(std::max(max0, max1), max2);
            *out_ptr = std::max(max2, max1);
            out_ptr++;
            //line0+=2;
            line1+=2;
            line2+=2;
        }
        if(wtail==0)
        {
            *out_ptr =std::max( std::max(line1[0],line1[1]),
                                std::max(line2[0],line2[1]));
            out_ptr++;
            line1+=2;
            line2+=2;
        }
        else
        {
            *out_ptr =std::max(line1[0],line2[0]);
                            
            out_ptr++;
            line1+=1;
            line2+=1;
        }

        // h center ---------------------------------------
        const float* line0=line1;
        line1=line2;
        line2=line1+inw;
        for(int i=0;i<mid_h;i++)
        {
            // left 
            float max0=std::max( std::max(line1[0],line1[1]),
                                std::max(line2[0],line2[1]));
            *out_ptr =std::max(std::max(line0[0],line0[1]),max0);
            out_ptr++;
            line0+=1;
            line1+=1; 
            line2+=1;
            //mid
            for(int j=0;j<mid_w;j++)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
                float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
                float max2 = std::max(std::max(line2[0], line2[1]), line2[2]);
                *out_ptr = std::max(std::max(max0, max1), max2);
                out_ptr++;
                line0+=2;
                line1+=2;
                line2+=2;
            }
            if(wtail==0)
            {
                max0 =std::max( std::max(line1[0],line1[1]),
                                    std::max(line2[0],line2[1]));
                *out_ptr =std::max(std::max(line0[0],line0[1]),max0);
                out_ptr++;
                line0+=inw_2;
                line1+=inw_2;
                line2+=inw_2;
            }
            else
            {
                *out_ptr =std::max(std::max(line0[0],line1[0]),line2[0]);
                out_ptr++;
                line0+=inw_1;
                line1+=inw_1;
                line2+=inw_1; 
            }
        }


        // h end ------------------------------------------
        if(htail==0)
        {
            *out_ptr =std::max( std::max(line1[0],line1[1]),
                                std::max(line0[0],line0[1]));
            out_ptr++;
            line1+=1; 
            line0+=1;

            for(int j=0;j<mid_w;j++)
            {
                float max0 = std::max(std::max(line0[0], line0[1]), line0[2]);
                float max1 = std::max(std::max(line1[0], line1[1]), line1[2]);
                //float max2 = std::max(std::max(line2[0], line2[1]), line2[2]);
                //*out_ptr = std::max(std::max(max0, max1), max2);
                *out_ptr = std::max(max0, max1);
                out_ptr++;
                line0+=2;
                line1+=2;
                //line2+=2;
            }
            if(wtail==0)
            {
                *out_ptr =std::max( std::max(line1[0],line1[1]),
                                std::max(line0[0],line0[1]));
                out_ptr++;
            }
            else
            {
                *out_ptr = std::max(line1[0],line0[0]);
                out_ptr++;
            }
        }
        else
        {
            *out_ptr = std::max(line0[0],line0[1]);
            out_ptr++;
            line0+=1;

            for(int j=0;j<mid_w;j++)
            {
                *out_ptr = std::max(std::max(line0[0], line0[1]), line0[2]);
                out_ptr++;
                line0+=2;
            }
            if(wtail==0)
            {
                *out_ptr =std::max(line0[0],line0[1]);
                out_ptr++;
            }
            else
            {
                *out_ptr = line0[0];
                out_ptr++;
            }
        }
   
    }
}

static void AvgPool_3x3s2_pad1(const float*input,float* output,
    int inc,int inh,int inw,
    int outh,int outw,
    int htail,int wtail)
{
    int in_hw=inw*inh;
    int out_hw=outh*outw;

    int mid_w = (inw-3+wtail)/2;
    int mid_h = (inh-3+htail)/2;

    int inw_2=inw+2;
    int inw_1=inw+1;

    for(int c=0;c<inc;c++)
    {
        const float* line1=input +c*in_hw;
        const float* line2=line1 + inw;
        float* out_ptr=output + c*out_hw;
        
        //h begin ---------------------------------------
        *out_ptr =(line1[0]+line1[1]+line2[0]+line2[1])*0.11111111f;
        out_ptr++;
        line1+=1; 
        line2+=1;

        for(int j=0;j<mid_w;j++)
        {
            *out_ptr = (line1[0]+ line1[1]+ line1[2]+
                        line2[0]+ line2[1]+ line2[2])*0.11111111f;
            out_ptr++;
            //line0+=2;
            line1+=2;
            line2+=2;
        }
        if(wtail==0)
        {
            *out_ptr =(line1[0]+line1[1]+line2[0]+line2[1])*0.11111111f;
            out_ptr++;
            line1+=2;
            line2+=2;
        }
        else
        {
            *out_ptr =(line1[0]+line2[0])*0.16666667f;
            out_ptr++;
            line1+=1;
            line2+=1;
        }


        // h center ---------------------------------------
        const float* line0=line1;
        line1=line2;
        line2=line1+inw;
        for(int i=0;i<mid_h;i++)
        {
            // left 
            *out_ptr =(line0[0]+line0[1]+
                       line1[0]+line1[1]+
                       line2[0]+line2[1])*0.11111111f;
            out_ptr++;
            line0+=1;
            line1+=1; 
            line2+=1;
            //mid
            for(int j=0;j<mid_w;j++)
            {
                *out_ptr = (line0[0] + line0[1] + line0[2] +
                            line1[0] + line1[1] + line1[2] +
                            line2[0] + line2[1] + line2[2])*0.11111111f;
                out_ptr++;
                line0+=2;
                line1+=2;
                line2+=2;
            }
            if(wtail==0)
            {
                *out_ptr =(line0[0]+line0[1]+
                line1[0] + line1[1] +
                line2[0] + line2[1])*0.11111111f;

                out_ptr++;
                line0+=inw_2;
                line1+=inw_2;
                line2+=inw_2; 
            }
            else
            {
                *out_ptr =(line0[0]+line1[0]+line2[0])*0.16666667f;
                out_ptr++;
                line0+=inw_1;
                line1+=inw_1;
                line2+=inw_1; 
            }

        }

        // h end ------------------------------------------
        if(htail==0)
        {
            *out_ptr =(line1[0]+line1[1]+line0[0]+line0[1])*0.11111111f;
            out_ptr++;
            line1+=1; 
            line0+=1;

            for(int j=0;j<mid_w;j++)
            {
                *out_ptr = (line0[0] + line0[1] +line0[2]+ 
                            line1[0] + line1[1] +line1[2])*0.11111111f;
                out_ptr++;
                line0+=2;
                line1+=2;
                //line2+=2;
            }
            if(wtail==0)
            {
                *out_ptr =(line1[0] + line1[1]+line0[0]+line0[1])*0.11111111f;
                out_ptr++;
            }
            else
            {
                *out_ptr =(line1[0] + line0[0])*0.16666667f;
                out_ptr++;
            }

        }
        else
        {
            *out_ptr =(line0[0]+line0[1])*0.16666667f;
            out_ptr++;
            line0+=1;

            for(int j=0;j<mid_w;j++)
            {
                *out_ptr = (line0[0] + line0[1] +line0[2])*0.16666667f;
                out_ptr++;
                line0+=2;
            }
            if(wtail==0)
            {
                *out_ptr =(line0[0]+line0[1])*0.16666667f;
                out_ptr++;
            }
            else
            {
                *out_ptr =line0[0]*0.25f;
                out_ptr++;
            }

        }
   
    }
}

// TODO: parallel in channel
static void Global_AvgPool(const float*input,float* output,
    int inc,int inh,int inw)
{
    int in_hw=inw*inh;
    int block=in_hw >>3;
    int tail=in_hw & ~7;


    for(int c=0;c<inc;c++)
    {
        const float* line0=input +c*in_hw;
        float* out_ptr=output + c;
        float sum=0.f;
        for(int j=0;j<block;j++)
        {
            float32x4_t p00=vld1q_f32(line0);
            float32x4_t p01=vld1q_f32(line0+4);
            p00=vaddq_f32(p00,p01);
            // p00=vpaddq_f32(p00,p00);
            // sum+=(p00[0]+p00[1]);
            sum+=(p00[0]+p00[1]+p00[2]+p00[3]);
            line0+=8;
        }
        for(int j=tail;j<in_hw;j++)
        {
            sum+=line0[0];
            line0++;
        }
        *out_ptr=sum/in_hw;
    }
}

// TODO: parallel in channel
static void Global_MaxPool(const float*input,float* output,
    int inc,int inh,int inw)
{
    int in_hw=inw*inh;
    int block=in_hw >>3;
    int tail=in_hw & ~7;

    for(int c=0;c<inc;c++)
    {
        const float* line0=input +c*in_hw;
        float* out_ptr=output + c;
        float32x4_t p00=vld1q_f32(line0);
        float32x4_t res=p00;
        for(int j=0;j<block;j++)
        {
            float32x4_t p00=vld1q_f32(line0);
            float32x4_t p01=vld1q_f32(line0+4);
            float32x4_t max0=vmaxq_f32(p00,p01);
            res=vmaxq_f32(res,max0);
            line0+=8;
        }
        float max_=std::max(std::max(res[0],res[1]),std::max(res[2],res[3]));
        for(int j=tail;j<in_hw;j++)
        {
            max_=std::max(max_,line0[0]);
            line0++;
        }
        *out_ptr=max_;
    }
}

#endif
