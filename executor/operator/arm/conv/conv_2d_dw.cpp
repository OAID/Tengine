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
#include "conv_selector.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"
#include "soc_runner.hpp"


namespace TEngine {

namespace conv_2d_dw {

const char * conv_name="CONV_DW";
const int default_prio=10;


extern "C" void dw_k3s1p1(float * data, int h, int w, float * kernel, float * output);
extern "C" void dw_k3s2p1(float * data, int h, int w, float * kernel, float * output);
extern "C" void dw_k3s1p1_relu_fused(float * data, int h, int w, float * kernel, float * output);
extern "C" void dw_k3s2p1_relu_fused(float * data, int h, int w, float * kernel, float * output);


struct  dw_param {
   float * input_buf;
   int input_h;
   int input_w;
   float * output_buf;
   int output_h;
   int output_w;
   float * weight_buf;
   int   channel_num;
   int stride;

};

struct  Conv2dDepth : public MTNodeOps {

     bool Run(Node * node);

     bool relu_fused;
     int  cpu_number;
     
     std::vector<CPUInfo> cpu_info;
     std::vector<int> cpu_list;

     void DirectConv(float * input_buf, int input_h, int input_w, float * output_buf, 
                             int output_h, int output_w, float * weight_buf, int channel_num, int stride);

     bool Aider(int cpu, int seq, void * data);


};

bool Conv2dDepth::Aider(int cpu, int seq, void * data)
{
    dw_param * param=(dw_param *)data;

    DirectConv(param->input_buf,param->input_h,param->input_w,
               param->output_buf,param->output_h,param->output_w,
               param->weight_buf,param->channel_num,param->stride);

    IncDone(1);


    return true;
}

void Conv2dDepth::DirectConv(float * input_buf, int input_h, int input_w, float * output_buf, 
                              int output_h, int output_w, float * weight_buf, int channel_num, int stride)
{
    int channel_size=input_h*input_w;

    for(int i=0;i<channel_num;i++)
    {
       if(stride==1)
       {
            if(relu_fused)
                dw_k3s1p1_relu_fused(input_buf,input_h,input_w,weight_buf,output_buf);
            else
                dw_k3s1p1(input_buf,input_h,input_w,weight_buf,output_buf);

            input_buf+=channel_size;
            output_buf+=channel_size;
            weight_buf+=9;

       }
       else if(stride ==2)
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
}

bool Conv2dDepth::Run(Node * node)
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


   for(int i=0;i<output_n;i++)
   { 
      if(cpu_number==1)
          DirectConv(input_buf,input_h,input_w,output_buf,output_h,output_w,weight_buf,input_c,stride_h);
      else
      {  
         //partition into 4 tasks
         std::vector<sub_op_task> task_list;
         std::vector<dw_param> param_list;

         auto f=std::bind(&Conv2dDepth::Aider,this,std::placeholders::_1,std::placeholders::_2,
                                          std::placeholders::_3);
                             
         task_list.resize(2);
         param_list.resize(2);

         int step=input_c/2;
         int channel_size=input_h*input_w;

         for(int i=0;i<2;i++)
         {
            dw_param * param=&param_list[i];
            sub_op_task * task=&task_list[i];

             task->exec_func=f;
             task->seq=i;
             task->data=param;

             param->input_buf=input_buf;
             param->input_h=input_h;
             param->input_w=input_w;
             param->output_buf=output_buf;
             param->output_h=output_h;
             param->output_w=output_w;
             param->weight_buf=weight_buf;
             param->channel_num=step;
             param->stride=stride_h;


             input_buf+=channel_size*step;
             if(stride_h==1)
                output_buf+=channel_size*step;
             else 
                output_buf+=output_h*output_w*step;
             weight_buf+=9*step;
         }

           //the last left ones
           param_list[1].channel_num=input_c-step;
         
           IncRequest(2);

           task_dispatch(task_list,-1);

           WaitDone();
      }
     
   }

   return true;
}

NodeOps * SelectFunc(SocInfo * soc_info, Node * node)
{
     Operator * op=node->GetOp();

     Convolution * conv_op=dynamic_cast<Convolution *>(op);

     ConvParam * param=conv_op->GetParam();

     const TShape& input_shape=node->GetInputTensor(0)->GetShape();

     int  input_c=input_shape.GetC();
     
     if(param->group==1 || input_c!=param->group)
          return nullptr;
     

     Conv2dDepth * ops=new Conv2dDepth();

     if(node->ExistAttr("Fused.ReLu"))
         ops->relu_fused=true;
     else
         ops->relu_fused=false;

     ops->cpu_list=soc_info->cpu_list; 
     ops->cpu_info=soc_info->cpu_info; 
     ops->cpu_number=ops->cpu_list.size();
	 
     ops->need_free=true;

     return ops;
}

} //namespace conv_2d_dw

void RegisterConv2dDepth(void)
{
     PrioSelector * selector=conv_selector::GetSelector("arm64");

     selector->Register(conv_2d_dw::default_prio,conv_2d_dw::SelectFunc);
}

} //namespace TEngine


