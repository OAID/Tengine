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
#include <iostream>
#include <functional>
#include <cstring>
#include <algorithm>
#include <complex>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/softmax.hpp"

namespace TEngine {

namespace SoftmaxImpl {

struct SoftmaxOps : public NodeOps {

bool Run(Node * node)
{
 
    Tensor * input_tensor=node->GetInputTensor(0);
    Tensor * output_tensor=node->GetOutputTensor(0);

    const std::vector<int>& dims=input_tensor->GetShape().GetDim();

    // 
    Softmax * softmax_op=dynamic_cast<Softmax*>(node->GetOp());
    SoftmaxParam * param_=softmax_op->GetParam();
    int axis=param_->axis;
    int out_size,in_size, on_size;
    out_size=1;
    for(int i=0;i<axis;i++)
    {
        out_size*=dims[i];
    }
    in_size=1;
    for(int i=axis+1;i<4;i++)
    {
        in_size*=dims[i];
    }
    on_size=dims[axis];

    float * input=(float *)get_tensor_mem(input_tensor);
    float * output=(float *)get_tensor_mem(output_tensor);
    float * max_array=(float *) std::malloc(in_size*sizeof(float));
    float * sum_array=(float *) std::malloc(in_size*sizeof(float));
   
    int on_in_size=on_size*in_size;
    
    for(int i=0;i<out_size;i++)
    {
         /* get max */
        int img_base=i*on_in_size;
        std::memcpy(max_array,input+img_base,in_size*sizeof(float));

        for(int j=1;j<on_size;j++)
        {
            int channel_base=img_base+j*in_size;

            for(int l=0;l<in_size;l++)
            {
               float data=input[channel_base+l];
               if(max_array[l]<data)
                   max_array[l]=data;
            }
        }

        std::memset(sum_array,0x0,in_size*sizeof(float));

        /* get the exp and the summary */

        for(int j=0;j<on_size;j++)
        {
            int channel_base=img_base+j*in_size;

            for(int l=0;l<in_size;l++)
            {
               float data=input[channel_base+l];

               output[channel_base+l]=std::exp(data-max_array[l]);
               sum_array[l]+=output[channel_base+l];
            }
        }

        /* the final result */

        for(int j=0;j<on_size;j++)
        {
            int channel_base=img_base+j*in_size;

            for(int l=0;l<in_size;l++)
            {
               output[channel_base+l]=output[channel_base+l]/sum_array[l];
            }
       }

    }

    std::free(max_array);
    std::free(sum_array);

    return true;
}

};

} //namespace SoftmaxImpl

using namespace SoftmaxImpl;

void RegisterSoftmaxNodeExec(void)
{
   SoftmaxOps * ops=new SoftmaxOps();
            
   NodeOpsRegistryManager::RegisterOPImplementor("common",
                "Softmax",ops);
}


} //namespace TEngine
