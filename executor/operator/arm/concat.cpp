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
#include <algorithm>

#include "logger.hpp"
#include "executor.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/concat.hpp"

extern "C" void concat_neon(void * output, void * input, int input_size);

namespace TEngine {

namespace ConcatImpl {

bool Run(Node * node, ExecEngine * engine)
{
    Tensor * input_tensor=node->GetInputTensor(0);
    Tensor * output_tensor=node->GetOutputTensor(0);
    Concat * concat_op=dynamic_cast<Concat *>(node->GetOp());
    ConcatParam*  param=concat_op->GetParam();


    const std::vector<int>& dims=input_tensor->GetShape().GetDim();

    int n=1;
    int axis=param->axis;

    for(int i=0;i<axis;i++)
       n=n*dims[i];

    if(n!=1)
    {
       LOG_ERROR()<<"concat for batch number > 1 to be implemented\n";
       return false;
    }

    int input_size=input_tensor->GetTotalSize();

    void * input=get_tensor_mem(input_tensor);
    void * output=get_tensor_mem(output_tensor);
    int input_number=node->GetInputNum();

    for(int i=0;i<input_number;i++)
    {
       input_tensor=node->GetInputTensor(i);
       input=get_tensor_mem(input_tensor);
       input_size=input_tensor->GetTotalSize();

       concat_neon(output,input,input_size);
       output=(char *)output+input_size;
    }

    return true;
}

} //namespace ConcatImpl

void RegisterConcatNodeExec(void)
{
    NodeExec concat_exec={nullptr,nullptr,ConcatImpl::Run,nullptr};

    RegisterNodeExec("Concat",concat_exec);
}


} //namespace TEngine
