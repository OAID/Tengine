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
 * Author: haitao@openailab.com
 *         chunyinglv@openailab.com
 */
#include "operator/pooling.hpp"
#include "static_graph.hpp"
#include <cmath>

namespace TEngine {

bool Pooling::InferShape(const std::vector<TEngine::TShape>& ishape, std::vector<TEngine::TShape>& oshape)
{
    const TShape& input_shape=ishape[0];

    int input_h=input_shape.GetH();
    int input_w=input_shape.GetW();

    if(param_.global)
    {
        param_.kernel_h=input_h;
        param_.kernel_w=input_w;
        param_.pad_h=0;
        param_.pad_w=0;
        param_.stride_h=1;
        param_.stride_w=1;

        param_.kernel_shape[0]=input_h;
        param_.kernel_shape[1]=input_w;
        param_.pads[0]=param_.pads[1]=param_.pads[2]=param_.pads[3]=0;
        param_.strides[0]=param_.strides[1]=1;
    }

    int output_h;
    int output_w;

    if(param_.pad_h>=0)
    {
       if(param_.caffe_flavor)
           output_h=std::ceil(((float)(input_h-param_.kernel_shape[0]+2*param_.pads[0]))/param_.strides[0])+1;
       else
           output_h=(input_h-param_.kernel_shape[0]+2*param_.pads[0])/param_.strides[0]+1;

    }
    else
    {
        int n=(input_h-1)/param_.strides[0]+1;
        int total_len=(n-1)*param_.strides[0]+param_.kernel_shape[0];
        int pad_num=total_len-input_h;

        if(param_.pad_h==-1)
        {
            param_.pads[0]=pad_num/2;
            param_.pads[2]=pad_num-pad_num/2;
        }
        else
        {
            param_.pads[0]=pad_num/2;
            param_.pads[2]=pad_num-pad_num/2;
        }

        output_h=(input_h-param_.kernel_shape[0]+param_.pads[0]+param_.pads[2])/param_.strides[0]+1;
    }

    if(param_.pad_w>=0)
    {
       if(param_.caffe_flavor)
           output_w=std::ceil(((float)(input_w-param_.kernel_shape[1]+2*param_.pads[1]))/param_.strides[1])+1;
       else
           output_w=(input_w-param_.kernel_shape[1]+2*param_.pads[1])/param_.strides[1]+1;
    }
    else
    {
        int n=(input_w-1)/param_.strides[1]+1;
        int total_len=(n-1)*param_.strides[1]+param_.kernel_shape[1];
        int pad_num=total_len-input_w;

        if(param_.pad_w==-1)
        {
            param_.pads[1]=pad_num/2;
            param_.pads[3]=pad_num-pad_num/2;
        }
        else
        {
            param_.pads[1]=pad_num/2;
            param_.pads[3]=pad_num-pad_num/2;
        }

        output_w=(input_w-param_.kernel_shape[1]+param_.pads[1]+param_.pads[3])/param_.strides[1]+1;
    }

    std::vector<int> dim={input_shape.GetN(),input_shape.GetC(),output_h,output_w};

    TShape shape;
    shape.SetDim(dim);
    shape.SetDataLayout("NCHW");

    oshape[0]=shape;

    return true;
}

float Pooling::GetFops(const std::vector<TShape>& inputs, const std::vector<TShape>& outputs)
{
    float patch_fops=param_.kernel_shape[0]*param_.kernel_shape[1];

    return (patch_fops*outputs[0].GetSize());
}

void Pooling::SetSchema(void)
{
    Input({"input:float32"})
    .Output({"output:float32"})
    .SetLayout("NCHW")
    .SetAttr("method","max")
    .SetAttr("kernel_h",2)
    .SetAttr("kernel_w",2)
    .SetAttr("stride_h",1)
    .SetAttr("stride_w",1)
    .SetAttr("pad_h",0)
    .SetAttr("pad_w",0)
    .SetAttr("global",0)
    .SetAttr("caffe_flavor",0)
    .SetDoc(R"DOC(Pooling Layer)DOC");
}


} //namespace TEngine
