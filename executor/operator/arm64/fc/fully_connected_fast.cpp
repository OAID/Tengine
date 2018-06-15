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
 * Author: xiaowei@openailab.com
 */

#include <iostream>
#include <functional>
#include <cstring>

#include "logger.hpp"
#include "operator/fully_connected.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"

extern "C" void sgemv_1x8_a72(float* biases, float* input, float* kernel, long kernel_size, float* output);
extern "C" void sgemv_1x2_a72(float* biases, float* input, float* kernel, long kernel_size, float* output);
extern "C" void sgemv_1x8_a53(float* biases, float* input, float* kernel, long kernel_size, float* output);
extern "C" void sgemv_1x2_a53(float* biases, float* input, float* kernel, long kernel_size, float* output);

#define TYPE_A72 0
#define TYPE_A53 1

namespace TEngine {

namespace FC_fast {

int default_prio=100;

// interleave most kernels in 8, last 7 interleave in 2, copy the lat kernels
void interleave_kernel(const float * kernel , const float * kernel_interleaved, int kernel_chan ,int kernel_size){

  int i,j,k;
  float *cur_kernel[8];
  float *cur_kernel_interleaved;

  // interleave 8 kernel
  for( i = 0 ; i < (kernel_chan & -8) ; i += 8 ){
    for(j = 0 ; j < 8 ; j ++)
      cur_kernel[j] = (float*)kernel + kernel_size *  (i + j);
    cur_kernel_interleaved = (float*)kernel_interleaved + kernel_size * i;
    for(k = 0 ; k < kernel_size ; k ++)
      for( j = 0 ; j < 8; j ++ )
        cur_kernel_interleaved[8*k+j] = *(cur_kernel[j]+k);
  }

  // interleave 2 kernel
  for(  ; i < (kernel_chan & -2) ; i += 2 ){
    for(j = 0 ; j < 2 ; j ++)
      cur_kernel[j] = (float*)kernel + kernel_size *  (i + j);
    cur_kernel_interleaved = (float*)kernel_interleaved + kernel_size * i;
    for(k = 0 ; k < kernel_size ; k ++)
      for( j = 0 ; j < 2; j ++ )
        cur_kernel_interleaved[2*k+j] = *(cur_kernel[j]+k);
  }

  // copy last kernel
  if(kernel_chan & 0x1){
    cur_kernel[0] = (float*)kernel + kernel_size *  i ;
    cur_kernel_interleaved = (float*)kernel_interleaved + kernel_size * i;
    for(k = 0 ; k < kernel_size ; k ++)
      cur_kernel_interleaved[k] = *(cur_kernel[0]+k);
  }

  return;
}

// start and end channel must be 8 aligned
void sgemv1x8(const float * input , float * weight_interleaved, bool have_biases, const float * biases, const float * output, int weight_stride, int start_channel, int end_channel, int cpu_type){

     int ch;
     float *cur_kernel, *cur_biases, *cur_result;

     for(ch = (start_channel & -8); ch < (end_channel & -8) ; ch += 8 ){
       cur_kernel  = (float*)(weight_interleaved + weight_stride * ch);
       cur_result  = (float*)(output + ch);
       cur_biases  = have_biases ? (float*)(biases + ch) : nullptr;
       if(cpu_type==TYPE_A72)
         sgemv_1x8_a72(cur_biases, (float*)input, cur_kernel, weight_stride, cur_result);
       else
         sgemv_1x8_a53(cur_biases, (float*)input, cur_kernel, weight_stride, cur_result);
     }

     return;
}

// start channel must be 2 aligned
void sgemv1x2(const float * input , float * weight_interleaved, bool have_biases, const float * biases, const float * output,int weight_stride, int start_channel, int end_channel, int cpu_type){

     float sum;
     int ch;
     float *cur_kernel, *cur_biases, *cur_result;

     for(ch = (start_channel & -2); ch < (end_channel & -2) ; ch += 2 ){
       cur_kernel  = (float*)(weight_interleaved + weight_stride * ch);
       cur_result  = (float*)(output + ch);
       cur_biases  = have_biases ? (float*)(biases + ch) : nullptr;
       if(cpu_type==TYPE_A72)
         sgemv_1x2_a72(cur_biases, (float*)input, cur_kernel, weight_stride, cur_result);
       else
         sgemv_1x2_a53(cur_biases, (float*)input, cur_kernel, weight_stride, cur_result);
     }

     if(end_channel & 0x1){
       cur_kernel  = (float*)(weight_interleaved + weight_stride * ch);
       cur_result  = (float*)(output + ch);
       sum = have_biases ? *(biases + ch) : 0.0;
       for(int j = 0; j< weight_stride ; j++)
         sum += input[j] * cur_kernel[j];
       *cur_result = sum;
     }

     return;
}



struct FCOps: public MTNodeOps {

int cpu_type;
int mt_disabled;

using sgemv_func_t=std::function<void(const float *,float *,bool,
		             const float*,const float*,int,int,int,int)>;


struct SgemvParam {
   sgemv_func_t func;
   const float * input;
   float * weight_interleaved;
   bool have_biases;
   const float * biases;
   float * output;
   int   weight_stride;
   int   start_channel;
   int   end_channel;
};

bool SgemvAider(int cpu, int aider, void * data)
{

    SgemvParam * param=(SgemvParam *)(data);


    param->func(param->input,param->weight_interleaved,param->have_biases,param->biases,param->output,
			    param->weight_stride,param->start_channel,param->end_channel,cpu_type);

    return true;
}


bool Prerun(Node * node)
{

      Tensor * tensor;

      tensor = node->GetInputTensor(1);
      int M = tensor->GetShape().GetH();
      int K = tensor->GetShape().GetW();

      float * weight = (float *) get_tensor_mem(tensor);

      float * weight_interleaved = (float *)std::malloc(sizeof(float)* K * M);
      interleave_kernel(weight , weight_interleaved, M ,K);
 
      (*node)["weight_interleaved"] = weight_interleaved;

      bool free_weight=false;

      const char * free_weight_str=std::getenv("FREE_FC_WEIGHT");

      if(free_weight_str && free_weight_str[0]=='1')
         free_weight=true;

      if(free_weight)
      {
          printf("Free fc weight: %s %d\n",
               tensor->GetName().c_str(),
               tensor->GetTotalSize());

          tensor->FreeMem();
      }

      return true;
}


bool Run(Node * node)
{
      int cpu_number=cpu_info->GetCPUNumber();

      Tensor * tensor;

      /* input */
      tensor = node->GetInputTensor(0);
      float * input=(float *)get_tensor_mem(tensor);
      int batch=tensor->GetShape().GetN();

      /* weight */
      tensor = node->GetInputTensor(1);
      int M  = tensor->GetShape().GetH();
      int K  = tensor->GetShape().GetW();
      float * weight_interleaved = any_cast<float *>(node->GetAttr("weight_interleaved"));

      /* output */
      tensor = node->GetOutputTensor(0);
      float * output=(float *)get_tensor_mem(tensor);

      /* biases */
      bool    have_biases    = (node->GetInputNum() > 2);
      float * biases         = have_biases ? (float *) get_tensor_mem(node->GetInputTensor(2)) :  nullptr;

      for(int n=0;n<batch;n++)
      {
         if(cpu_number==1 || mt_disabled)
         {
             if(M>=8)
                 sgemv1x8(input,weight_interleaved,have_biases,biases,output,K,0,M&-8,cpu_type);
             if(M&0x7)
                 sgemv1x2(input,weight_interleaved,have_biases,biases,output,K,M&-8,M,cpu_type);
         }
         else
         {
             int m8_num=M/8;

             int step=(m8_num+(cpu_number-1))/cpu_number;

             if((m8_num-(cpu_number-1)*step)<= cpu_number/2)
             {
                step=m8_num/cpu_number;
             }

             step=step*8;

             std::vector<sub_op_task> task_list;
             std::vector<SgemvParam>  param_list;

             task_list.resize(cpu_number);
             param_list.resize(cpu_number);

             int start_channel=0;


             for(int i=0;i<cpu_number;i++)
             {
                 SgemvParam * param=&param_list[i];
                 sub_op_task * task=&task_list[i];

                 auto f=std::bind(&FCOps::SgemvAider,this,std::placeholders::_1,
                                std::placeholders::_2,std::placeholders::_3);

                 task->exec_func=f;
                 task->seq=i;
                 task->data=param;

                 param->func=sgemv1x8;
                 param->input=input;
                 param->weight_interleaved=weight_interleaved;
                 param->have_biases=have_biases;
                 param->biases=biases;
                 param->output=output;
                 param->weight_stride=K;
                 param->start_channel=start_channel;
                 param->end_channel=param->start_channel+step;

                 start_channel+=step;
             }

             param_list[cpu_number-1].end_channel=m8_num*8;

             task_dispatch(task_list,-1);


             if(M&0x7)
                 sgemv1x2(input,weight_interleaved,have_biases,biases,output,K,M&-8,M,cpu_type);

             wait_done();

         }

         input+=K;
         output+=M;
      }

      return true;
}

bool Postrun(Node * node)
{
       float * mem=any_cast<float *>(node->GetAttr("weight_interleaved"));
       std::free(mem);

       return true;  
}


};

NodeOps * SelectFunc(const CPUInfo * cpu_info, Node * node)
{
    FCOps * ops= new FCOps();

    int master_cpu=cpu_info->GetMasterCPU();

    if(cpu_info->GetCPUModelString(master_cpu) == std::string("A72"))
         ops->cpu_type=TYPE_A72;
    else
         ops->cpu_type=TYPE_A53;

	const char * mt_enabled=std::getenv("FC_MT_ENABLED");

	if(mt_enabled && mt_enabled[0]=='1')
		ops->mt_disabled=false;
	else
		ops->mt_disabled=true;

     return ops;
}



} //namespace FC_fast

using namespace FC_fast;

void RegisterFullyConnectedFast(void)
{

    NodeOpsRegistryManager::RegisterOPImplementor("arm64","FullyConnected",
				FC_fast::SelectFunc,FC_fast::default_prio);
}


} //namespace TEngine
