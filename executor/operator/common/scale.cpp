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
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <functional>
#include <stdlib.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/scale.hpp"
#include <math.h>

namespace TEngine
{

namespace ScaleImpl
{

struct ScaleOps : public NodeOps
{



    bool Run(Node * node)
    {

        const Tensor * input_tensor=node->GetInputTensor(0);
        const Tensor * gamma_tensor=node->GetInputTensor(1);
        const Tensor * beta_tensor=node->GetInputTensor(2);
        Tensor * output_tensor=node->GetOutputTensor(0);
        const TShape&  shape=input_tensor->GetShape();
        const std::vector<int> dims=shape.GetDim();

        int batch_number=dims[0];
        int channel_num=dims[1];
        int channel_size=dims[2]*dims[3];
        int img_size=channel_num*channel_size;


        const float * input=(const float *)get_tensor_mem(input_tensor);
        float * gamma=(float *)get_tensor_mem(gamma_tensor);
        float * output=(float *)get_tensor_mem(output_tensor);

        if(beta_tensor==nullptr)
        {
            for(int i=0;i<batch_number;i++)
            {
                for(int c=0;c<channel_num;c++)
                {
                    int offset=i*img_size+c*channel_size;
                    for(int l=0;l<channel_size;l++)
                    {
                        output[offset+l]=input[offset+l]*gamma[c];
                    }
                }
            }
        }
        else
        {
            float * beta=(float *)get_tensor_mem(beta_tensor); 
            for(int i=0;i<batch_number;i++)
            {
                for(int c=0;c<channel_num;c++)
                {
                    int offset=i*img_size+c*channel_size;
                    for(int l=0;l<channel_size;l++)
                    {
                        output[offset+l]=input[offset+l]*gamma[c]+beta[c];
                    }
                }
            }
        }

        return true;
    }




};

} //namespace ScaleImpl

using namespace ScaleImpl;

void RegisterScale_NodeExec(void)
{
    ScaleOps *ops = new ScaleOps();

    NodeOpsRegistryManager::RegisterOPImplementor("common",
                                                  "Scale", ops);
}

} //namespace TEngine
