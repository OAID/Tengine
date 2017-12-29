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
 */
#include <iostream>
#include <cstring>
#include <cstdlib>

#include "logger.hpp"
#include "executor.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"

#include "conv_implementor.hpp"


namespace TEngine {

namespace conv_2d_dw {

const char * conv_name="CONV_DW";
const int default_prio=10;

class Conv2dDepth : public ConvImplementor {
public:
 bool Prerun(Node * node, ExecEngine * engine);
 bool Run(Node * node, ExecEngine * engine);
 bool Postrun(Node * node, ExecEngine * engine);
 bool Support(ConvParam * param, Tensor * input_tensor, Tensor * weight_tensor);

 Conv2dDepth() { name=conv_name;}

};



bool Conv2dDepth::Prerun(Node * node, ExecEngine * engine)
{
   return true;
}

extern "C" void dw_k3s1p1(float * data, int h, int w, float * kernel, float * output);
extern "C" void dw_k3s2p1(float * data, int h, int w, float * kernel, float * output);
extern "C" void dw_k3s1p1_relu_fused(float * data, int h, int w, float * kernel, float * output);
extern "C" void dw_k3s2p1_relu_fused(float * data, int h, int w, float * kernel, float * output);

bool Conv2dDepth::Run(Node * node, ExecEngine * engine)
{
   Tensor * input_tensor=node->GetInputTensor(0);
   Convolution * conv_op=dynamic_cast<Convolution *>(node->GetOp());
   ConvParam*  param=conv_op->GetParam();

   const TShape& input_shape=input_tensor->GetShape();

   int  input_c=input_shape.GetC();
   int  input_h=input_shape.GetH();
   int  input_w=input_shape.GetW();

   /* output */
   Tensor * output_tensor=node->GetOutputTensor(0);
   TShape& output_shape=output_tensor->GetShape();

   int output_h=output_shape.GetH();
   int output_w=output_shape.GetW();
   int output_n=output_shape.GetN();

   Tensor * weight_tensor=node->GetInputTensor(1);
   float * weight_buf=(float *)get_tensor_mem(weight_tensor);
   float * input_buf=(float *)get_tensor_mem(input_tensor);
   float * output_buf=(float *)get_tensor_mem(output_tensor);

   int kernel_h=param->kernel_h;
   int kernel_w=param->kernel_w;
   int pad_h=param->pad_h;
   int pad_w=param->pad_w;
   int stride_h=param->stride_h;
   int stride_w=param->stride_w;
   int dilation_h=param->dilation_h;
   int dilation_w=param->dilation_w;

   //parameter check
   //currently, only depthwise in Mobilenet is supported

   if(kernel_h != 3 || kernel_w!=3 || pad_h!=1 
       || pad_w!=1 || dilation_h!=1 || dilation_w!=1
       || stride_w!=stride_h)
   {
       XLOG_ERROR()<<"unsupported depthwise convolution\n";
       return false;
   }

   bool relu_fused=false;

   if(node->ExistAttr("Fused.ReLu"))
        relu_fused=true;


   for(int i=0;i<output_n;i++)
   { 
      if(stride_h==1)
      {
       //processed per channel
       int channel_size=input_h*input_w;

       for(int i=0;i<input_c;i++)
       {
            if(relu_fused)
                dw_k3s1p1_relu_fused(input_buf,input_h,input_w,weight_buf,output_buf);
            else
                dw_k3s1p1(input_buf,input_h,input_w,weight_buf,output_buf);

            input_buf+=channel_size;
            output_buf+=channel_size;
            weight_buf+=9;
       }
     }
     else if(stride_h==2)
     {
       int channel_size=input_h*input_w;

       for(int i=0;i<input_c;i++)
       {
            if(relu_fused)
                dw_k3s2p1_relu_fused(input_buf,input_h,input_w,weight_buf,output_buf);
            else
                dw_k3s2p1(input_buf,input_h,input_w,weight_buf,output_buf);
            input_buf+=channel_size;
            output_buf+=output_h*output_w;
            weight_buf+=9;
       }
     }
     
     return true;
   }

   return true;
}

bool Conv2dDepth::Postrun(Node * node, ExecEngine * engine)
{
   return true;
}

bool Conv2dDepth::Support(ConvParam * param, Tensor * input_tensor, Tensor * weight_tensor)
{

   const TShape& input_shape=input_tensor->GetShape();

   int  input_c=input_shape.GetC();

   if(param->group>1 && input_c==param->group)
      return true;
   else
      return false;
}

} //namespace conv_2d_dw

void RegisterConv2dDepth(void)
{
    conv_2d_dw::Conv2dDepth * conv=new conv_2d_dw::Conv2dDepth();

    char * prio=std::getenv(conv_2d_dw::conv_name);

    if(prio)
        conv->priority=strtoul(prio,NULL,10);
    else
        conv->priority=conv_2d_dw::default_prio;

    ConvImplementorManager::Register(conv);
}

} //namespace TEngine


