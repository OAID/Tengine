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


extern "C" void sgemv_1x8(float * biases,
                      float * input0,
                      float * kernel01,float * kernel23,float * kernel45,float * kernel67,
                      float * output0, float * output1, float * output2, float * output3, float * output4, float * output5,   float * output6,   float * output7,
                      long kernel_size);

namespace TEngine {

namespace FCImpl {

template <typename Dtype>
void interleave2_kernel(const Dtype * kernel , const Dtype * kernel_interleaved, int kernel_chan ,int kernel_size){

  int kernel_size_aligned4 = (kernel_size + 3) & (-4);
  int i,j;
  Dtype *cur_kernel0;
  Dtype *cur_kernel1;
  Dtype *cur_kernel_interleaved;

  // interleave 2 kernel
  for( i = 0 ; i < (kernel_chan & -2) ; i += 2 ){
    cur_kernel0 = (Dtype*)kernel + kernel_size *  i;
    cur_kernel1 = (Dtype*)kernel + kernel_size * (i + 1);
    cur_kernel_interleaved = (Dtype*)kernel_interleaved + kernel_size_aligned4 * i;
    for(j = 0 ; j < kernel_size ; j ++){
      cur_kernel_interleaved[2*j]   = cur_kernel0[j];
      cur_kernel_interleaved[2*j+1] = cur_kernel1[j];
    }
    for(j=kernel_size; j < kernel_size_aligned4; j++){
      cur_kernel_interleaved[2*j]   = 0.0;
      cur_kernel_interleaved[2*j+1] = 0.0;
    }
  }

  // last kernel
  if(kernel_chan & 0x1){
    cur_kernel0 = (Dtype*)kernel + kernel_size * (kernel_chan & -2);
    cur_kernel_interleaved = (Dtype*)kernel_interleaved + kernel_size_aligned4 * (kernel_chan & -2);
    for( j = 0 ; j < kernel_size ; j ++){
      cur_kernel_interleaved[j] = cur_kernel0[j];
    }
    for(j = kernel_size; j < kernel_size_aligned4; j++){
      cur_kernel_interleaved[j]   = 0.0;
    }
  }

  return;
}

template <typename Dtype>
void sgemv1x8(const Dtype * input , Dtype * weight_interleaved, Dtype * biases, const Dtype * output,int weight_stride, int start_channel){

     float * kernel01 = (float*)(weight_interleaved + weight_stride *   start_channel );
     float * kernel23 = (float*)(weight_interleaved + weight_stride *  (start_channel + 2 ));
     float * kernel45 = (float*)(weight_interleaved + weight_stride *  (start_channel + 4 ));
     float * kernel67 = (float*)(weight_interleaved + weight_stride *  (start_channel + 6 ));
     float * output0 = (float*)((Dtype*)output + start_channel);
     float * output1 = (float*)((Dtype*)output + start_channel + 1);
     float * output2 = (float*)((Dtype*)output + start_channel + 2);
     float * output3 = (float*)((Dtype*)output + start_channel + 3);
     float * output4 = (float*)((Dtype*)output + start_channel + 4);
     float * output5 = (float*)((Dtype*)output + start_channel + 5);
     float * output6 = (float*)((Dtype*)output + start_channel + 6);
     float * output7 = (float*)((Dtype*)output + start_channel + 7);
     sgemv_1x8((float*)biases,(float*)input,kernel01,kernel23,kernel45,kernel67,
              output0,output1,output2,output3,output4,output5,output6,output7,weight_stride/4);

}

struct FCOps: public NodeOps {

bool Prerun(Node * node)
{

      Tensor * tensor;

      tensor = node->GetInputTensor(1);
      int M = tensor->GetShape().GetH();
      int K = tensor->GetShape().GetW();

      float * weight = (float *) get_tensor_mem(tensor);

      float * weight_interleaved = (float *)std::malloc(sizeof(float)*((K+3) & (-4))*M);
      interleave2_kernel(weight , weight_interleaved, M ,K);
 
      (*node)["weight_interleaved"] = weight_interleaved;

      return true;
}

bool Run(Node * node)
{
      Tensor * tensor;

      /* input */
      tensor = node->GetInputTensor(0);
      float * input=(float *)get_tensor_mem(tensor);
      int batch=tensor->GetShape().GetN();

      /* weight */
      tensor = node->GetInputTensor(1);
      int M  = tensor->GetShape().GetH();
      int K  = tensor->GetShape().GetW();
      float * weight = (float *) get_tensor_mem(tensor);
      float * weight_interleaved = any_cast<float *>(node->GetAttr("weight_interleaved"));
      int weight_stride = (K + 3) & (-4);

      /* output */
      tensor = node->GetOutputTensor(0);
      float * output=(float *)get_tensor_mem(tensor);

      /* biases */
      bool    have_biases    = (node->GetInputNum() > 2);
      float   biases_zero[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      float * biases         = have_biases ? (float *) get_tensor_mem(node->GetInputTensor(2)) :  nullptr;

      for(int n=0;n<batch;n++)
      {

         for(int i=0; i< (M&-8) ; i+=8){
             float* cur_biases =  have_biases ? (biases + i) : biases_zero;
             sgemv1x8(input,weight_interleaved,cur_biases,output,weight_stride,i);
         }
         for(int i = (M & -8) ; i < M ; i++){
             float sum = have_biases ? *(biases + i) : 0.0;
             for(int j = 0; j < K; j++)
                sum += input[j] * weight[K*i+j];
            output[i] = sum;
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


} //namespace FCImpl

using namespace FCImpl;

void RegisterFullyConnectedNodeExec(void)
{
    FCOps * ops=new FCOps();

    NodeOpsRegistryManager::RegisterOPImplementor("arm64",
                  "FullyConnected",ops);
}


} //namespace TEngine
