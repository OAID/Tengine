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
#include <functional>
#include <cstring>

#include "logger.hpp"
#include "operator/fused_operator.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"


namespace TEngine {

namespace FusedBNScaleDemo {


struct FusedBNScaleOps: public NodeOps 
{

bool Run(Node * node)
{
    const Tensor * input_tensor=node->GetInputTensor(0);
    Tensor * output_tensor=node->GetOutputTensor(0);

    const TShape&  shape=input_tensor->GetShape();
    const std::vector<int> dims=shape.GetDim();

    int batch_number=dims[0];
    int channel_num=dims[1];
    int channel_size=dims[2]*dims[3];

    Tensor * gamma_tensor=node->GetInputTensor(1);
    Tensor * beta_tensor=node->GetInputTensor(2);
    Tensor * mean_tensor=node->GetInputTensor(3);
    Tensor * var_tensor=node->GetInputTensor(4);


    float * gamma=(float *)get_tensor_mem(gamma_tensor);
    float * beta=(float *)get_tensor_mem(beta_tensor);
    float * mean=(float *)get_tensor_mem(mean_tensor);
    float * var=(float *)get_tensor_mem(var_tensor);

    const float * input=(const float *)get_tensor_mem(input_tensor);
    float * output=(float *)get_tensor_mem(output_tensor);


    for(int i=0;i<batch_number;i++)
    {
        for(int c=0;c<channel_num;c++)
        {
           float s_mean=mean[c];
           float s_var=var[c];
           float s_gamma=gamma[c];
           float s_beta=beta[c];

           for(int l=0;l<channel_size;l++)
           {
              float data=input[l];
              data=data*s_var+s_mean;

              data=data*s_gamma+s_beta;

              if(data<0.0)
                  data=0;

              output[l]=data;
           }

           input+=channel_size;
           output+=channel_size;
        }
    }
    

    return true;

}

};


} //namespace FusedBNScaleDemo

using namespace FusedBNScaleDemo;

void RegisterFusedDemoNodeExec(void)
{
    FusedBNScaleOps * ops=new FusedBNScaleOps();

    NodeOpsRegistryManager::RegisterOPImplementor(REF_REGISTRY_NAME,
               FusedBNScaleReLu::class_name,ops);

}

} //namespace TEngine

