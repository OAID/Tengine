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
#include <iostream>
#include <functional>

#include "graph.hpp"
#include "operator/pooling.hpp"
#include "executor.hpp"
#include "tensor_mem.hpp"
#include"pooling_kernel.h"
#include<stdlib.h>
namespace TEngine {


static bool PoolingPrerun(Node * node, ExecEngine *)
{
   return true;
}


bool PoolingRun(Node * node, ExecEngine *)
{
    // operator, param
    Pooling * pooling_op=dynamic_cast<Pooling*>(node->GetOp());
    PoolParam * param_=pooling_op->GetParam();

    // input, output, shape 
    Tensor * itensor=node->GetInputTensor(0);
    const TShape& ishape=itensor->GetShape();
    Tensor * otensor=node->GetOutputTensor(0);
    TShape& oshape=otensor->GetShape();
    // dim=[n,c,h,w]
    const std::vector<int>& in_dim=ishape.GetDim();
    const std::vector<int>& out_dim=oshape.GetDim();
    int in_hw=in_dim[3]*in_dim[2];
    int in_chw=in_dim[1]*in_hw;

    int out_hw=out_dim[2]*out_dim[3];
    int out_chw=out_dim[1]*out_hw;
    int n_skip,c_skip,on_skip,oc_skip;
    // data 
    float* input_data=(float*)get_tensor_mem(itensor);
    float* output_data=(float*)get_tensor_mem(otensor);


    // global
     if(param_->global)
    {
        if( param_->alg==kPoolMax)
        {
            for(int n=0;n<in_dim[0];n++)
            {
                n_skip=n*in_chw;
                float* out_ptr=output_data+n*in_dim[1];
                Global_MaxPool(input_data+n_skip,out_ptr,
                in_dim[1],in_dim[2],in_dim[3]);
            }
        }
        else if(param_->alg==kPoolAvg)
        {
            for(int n=0;n<in_dim[0];n++)
            {
                n_skip=n*in_chw;
                float* out_ptr=output_data+n*in_dim[1];

                Global_AvgPool(input_data+n_skip,out_ptr,
                in_dim[1],in_dim[2],in_dim[3]);
            }
        }
    return true;
    }

    // h_tail,w_tail
    int wtail=(in_dim[3]-param_->kernel_shape[1])%param_->strides[1];
    int htail=(in_dim[2]-param_->kernel_shape[0])%param_->strides[0];

    // max
    if( param_->alg==kPoolMax)
    {
        if(param_->kernel_shape[0]==2 && param_->kernel_shape[1]==2
             && param_->strides[0]==2 && param_->strides[1]==2)
        {
            if(param_->pads[0]==0 && param_->pads[1]==0 && 
                param_->pads[2]==0 && param_->pads[3]==0)
            {
                for(int n=0;n<in_dim[0];n++)
                {
                    MaxPool_2x2s2( input_data+n*in_chw,
                                output_data+n*out_chw,
                                in_dim[1],in_dim[2],in_dim[3],
                                out_dim[2],out_dim[3],
                                htail,wtail);  
                }
                return true;
            }
            if(param_->pads[0]==1 && param_->pads[1]==1 &&
              param_->pads[2]==1 && param_->pads[3]==1)
            {
                for(int n=0;n<in_dim[0];n++)
                {
                MaxPool_2x2s2_pad1( input_data+n*in_chw,
                                output_data+n*out_chw,
                                in_dim[1],in_dim[2],in_dim[3],
                                out_dim[2],out_dim[3]);  
                }
                return true;
            }
        }
        if(param_->kernel_shape[0]==3 && param_->kernel_shape[1]==3 
             && param_->strides[0]==2 && param_->strides[1]==2)
        {
            if(param_->pads[0]==0 && param_->pads[1]==0 && 
                param_->pads[2]==0 && param_->pads[3]==0)
            {
                for(int n=0;n<in_dim[0];n++)
                {
                    MaxPool_3x3s2( input_data+n*in_chw,
                                output_data+n*out_chw,
                                in_dim[1],in_dim[2],in_dim[3],
                                out_dim[2],out_dim[3],htail,wtail);  
                }   
                return true;
            }
            if(param_->pads[0]==1 && param_->pads[1]==1 &&
              param_->pads[2]==1 && param_->pads[3]==1)
            {
                for(int n=0;n<in_dim[0];n++)
                {
                    MaxPool_3x3s2_pad1( input_data+n*in_chw,
                                output_data+n*out_chw,
                                in_dim[1],in_dim[2],in_dim[3],
                                out_dim[2],out_dim[3],
                                htail,wtail);  
                }   
                return true;
            }
        }
        else
        {
            for(int n=0;n<in_dim[0];n++)
            {
                n_skip=n*in_chw;
                on_skip=n*out_chw;
            
                for(int c=0;c<in_dim[1];c++)
                {
                    c_skip=n_skip+c*in_hw;
                    oc_skip=on_skip+c*out_hw;
                    
                    for(int ph=0;ph<out_dim[2];ph++)
                    {
                        int h_start = ph * param_->strides[0] - param_->pads[0];
                        int h_end= std::min(h_start + param_->kernel_shape[0], in_dim[2]);
                        h_start = std::max(h_start,0);

                        for(int pw=0;pw<out_dim[3];pw++)
                        {
                            int w_start = pw * param_->strides[1] - param_->pads[1];
                            int w_end = std::min(w_start + param_->kernel_shape[1], in_dim[3]);
                            w_start = std::max(w_start,0);

                            const int out_index = oc_skip + ph * out_dim[3] + pw;
                            output_data[out_index] = input_data[c_skip + h_start*in_dim[3]+w_start];
                            for(int h=h_start;h<h_end;h++)
                            {
                                for(int w=w_start;w<w_end;w++)
                                {
                                    int in_index= c_skip + h*in_dim[3] + w;
                                    
                                    if(input_data[in_index]>output_data[out_index])
                                    {
                                        
                                        output_data[out_index]=input_data[in_index];
                                    
                                    }
                                }
                            }// end ksize_h,ksize_w
                        }
                    
                    }
                }
            }
            return true;
        }
    }
    if (param_->alg==kPoolAvg)
    {
        if(param_->kernel_shape[0]==2 && param_->kernel_shape[1]==2
             && param_->strides[0]==2 && param_->strides[1]==2)
        {
            if(param_->pads[0]==0 && param_->pads[1]==0 && 
                param_->pads[2]==0 && param_->pads[3]==0)
            {
                for(int n=0;n<in_dim[0];n++)
                {
                    AvgPool_2x2s2( input_data+n*in_chw,
                                output_data+n*out_chw,
                                in_dim[1],in_dim[2],in_dim[3],
                                out_dim[2],out_dim[3],
                                htail,wtail);  
                }
                return true;
            }
            // all pads=1
            if(param_->pads[0]==1)
            {
                for(int n=0;n<in_dim[0];n++)
                {
                    AvgPool_2x2s2_pad1( input_data+n*in_chw,
                                output_data+n*out_chw,
                                in_dim[1],in_dim[2],in_dim[3],
                                out_dim[2],out_dim[3]);  
                }
                return true;
            }
        }

        if(param_->kernel_shape[0]==3 && param_->kernel_shape[1]==3 
             && param_->strides[0]==2 && param_->strides[1]==2)
        {
            if(param_->pads[0]==0)
            {
                for(int n=0;n<in_dim[0];n++)
                {
                    AvgPool_3x3s2( input_data+n*in_chw,
                                output_data+n*out_chw,
                                in_dim[1],in_dim[2],in_dim[3],
                                out_dim[2],out_dim[3],htail,wtail);  
                }  
                return true;
            }
            if(param_->pads[0]==1 && param_->pads[1]==1 &&
              param_->pads[2]==1 && param_->pads[3]==1)
            {
                for(int n=0;n<in_dim[0];n++)
                {
                AvgPool_3x3s2_pad1( input_data+n*in_chw,
                                output_data+n*out_chw,
                                in_dim[1],in_dim[2],in_dim[3],
                                out_dim[2],out_dim[3],
                                htail,wtail);  
                }  
                return true;
            }
        }
        else
        {

            for(int n=0;n<in_dim[0];n++)
            {
                n_skip=n*in_chw;
                on_skip=n*out_chw;
                for(int c=0;c<in_dim[1];c++)
                {
                    c_skip=n_skip+c*in_hw;
                    oc_skip=on_skip+c*out_hw;
                    
                    for(int ph=0;ph<out_dim[2];ph++)
                    {
                        int h_start = ph * param_->strides[0] - param_->pads[0];
                        int h_end=  std::min(h_start + param_->kernel_shape[0], in_dim[2]);
                        h_start = std::max(h_start,0);

                        for(int pw=0;pw<out_dim[3];pw++)
                        {
                            int w_start = pw * param_->strides[1] - param_->pads[1];
                            int w_end= std::min(w_start + param_->kernel_shape[1], in_dim[3]);
                            w_start = std::max(w_start,0);
                            int pool_size=(h_end-h_start)*(w_end-w_start);
                            const int out_index = oc_skip + ph * out_dim[3] + pw;
                            output_data[out_index] = 0.f;
                            for(int h=h_start;h<h_end;h++)
                            {
                                for(int w=w_start;w<w_end;w++)
                                {
                                    output_data[out_index]+=input_data[c_skip + h*in_dim[3] + w];    
                                }
                            }// end ksize_h,ksize_w
                            output_data[out_index]/=pool_size;
                        }
                    
                    }
                }
            }
            return true;
        }
    }

   return true;
}


static bool PoolingPostrun(Node * node, ExecEngine *)
{
   return true;
}



void RegisterPoolingNodeExec(void)
{
    NodeExec pool_exec={nullptr,PoolingPrerun,PoolingRun,PoolingPostrun};
    RegisterNodeExec("Pooling",pool_exec);
}



} //namespace TEngine
