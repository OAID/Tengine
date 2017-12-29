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
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "executor.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/batch_norm.hpp"
#include<arm_neon.h>
namespace TEngine {

namespace BatchNormImpl {

bool OnBind(Node * node, ExecEngine * engine)
{
    //set the inplace feature
    inplace_t io_map;
   
    io_map[0]=0;

    node->SetAttr(ATTR_INPLACE,io_map);

    return true;
}


bool Prerun(Node * node, ExecEngine * engine)
{
    const Tensor * input_tensor=node->GetInputTensor(0);
    const TShape&  shape=input_tensor->GetShape();
    const std::vector<int> dims=shape.GetDim();

    int channel_num=dims[1];
  
    float * scale_mean=(float *)std::malloc(channel_num*sizeof(float));
    float * scale_var_inv=(float *)std::malloc(channel_num*sizeof(float));

    const Tensor * mean_tensor=node->GetInputTensor(3);
    const Tensor * var_tensor=node->GetInputTensor(4);
    const float * mean=(const float *)get_tensor_mem(mean_tensor);
    const float * var=(const float *)get_tensor_mem(var_tensor);

    BatchNorm * bn_op=dynamic_cast<BatchNorm *>(node->GetOp());
    BatchNormParam * param=bn_op->GetParam();

    if(param->caffe_flavor)
    {
        float rescale_factor;
        float eps=param->eps;

        rescale_factor=param->rescale_factor?1/param->rescale_factor:0;
        for(int c=0;c<channel_num;c++)
        {
           scale_var_inv[c]=1.f/std::sqrt(var[c]*rescale_factor + eps);
           scale_mean[c]=-mean[c]*rescale_factor*scale_var_inv[c];
        }
    }

    node->SetAttr("scale_mean",scale_mean);
    node->SetAttr("scale_var_inv",scale_var_inv);

    return true;
}

bool Run(Node * node, ExecEngine * engine)
{
    const Tensor * input_tensor=node->GetInputTensor(0);
    Tensor * output_tensor=node->GetOutputTensor(0);


    const TShape&  shape=input_tensor->GetShape();
    const std::vector<int> dims=shape.GetDim();

    int batch_number=dims[0];
    int channel_num=dims[1];
    int channel_size=dims[2]*dims[3];
    int img_size=channel_num*channel_size;

    BatchNorm * bn_op=dynamic_cast<BatchNorm *>(node->GetOp());
    BatchNormParam * param=bn_op->GetParam();


    const float * input=(const float *)get_tensor_mem(input_tensor);
    float * output=(float *)get_tensor_mem(output_tensor);

    if(param->caffe_flavor)
    {
        

        float * scale_mean=any_cast<float *>(node->GetAttr("scale_mean"));
        float * scale_var_inv=any_cast<float *>(node->GetAttr("scale_var_inv"));
        
        /* only use mean and var */
        for(int i=0;i<batch_number;i++)
        {
            for(int c=0;c<channel_num;c++)
            {
                float s_mean=scale_mean[c];
                float s_var=scale_var_inv[c];
                float32x4_t _mean = vdupq_n_f32(s_mean);
                float32x4_t _var = vdupq_n_f32(s_var);
                int offset=i*img_size+c*channel_size;
                const float* input_ptr=input +offset;
                float* output_ptr=output+offset;

                for(int l=0;l<channel_size-3;l+=4)
                {
                    float32x4_t _input=vld1q_f32(input_ptr);
                    vst1q_f32(output_ptr,vmlaq_f32(_mean,_input,_var));
                    input_ptr+=4;
                    output_ptr+=4;
                    //output[offset]= input[offset]*scale_var_inv[c] - scale_mean[c];
                }
                for(int l=channel_size&~3;l<channel_size;l++)
                {
                    *output_ptr = (*input_ptr )*s_var +s_mean;
                    input_ptr++;
                    output_ptr++;
                }
            }
        }

    }
    else
    {
        LOG_ERROR()<<"TODO: support not caffe_flavor\n";
    }

    return true;
}

bool Postrun(Node * node, ExecEngine * engine)
{
    float * scale_mean=any_cast<float *>(node->GetAttr("scale_mean"));
    float * scale_var=any_cast<float *>(node->GetAttr("scale_var_inv"));

    std::free(scale_mean);
    std::free(scale_var);

    return true;
}


} //namespace BatchNormImpl

void RegisterBatchNormNodeExec(void)
{
    NodeExec bn_exec={BatchNormImpl::OnBind,
                      BatchNormImpl::Prerun,
                      BatchNormImpl::Run,
                      BatchNormImpl::Postrun};

    RegisterNodeExec(BatchNormName,bn_exec);
}


} //namespace TEngine
