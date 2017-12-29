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
#include "executor.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"

namespace TEngine {

namespace ReLuImpl {

bool OnBind(Node * node, ExecEngine * engine)
{
    //set the inplace feature
    inplace_t io_map;
   
    io_map[0]=0;

    node->SetAttr(ATTR_INPLACE,io_map);

    return true;
}

extern "C" void relu_neon(float * ,int );

bool Run(Node * node, ExecEngine * engine)
{
    //input tensor and output tensor is the same
    Tensor * input_tensor=node->GetInputTensor(0);
    const TShape& shape=input_tensor->GetShape();
    int elem_num=shape.GetSize();
    float * data=(float *)get_tensor_mem(input_tensor);

    relu_neon(data,elem_num);
    return true;
}

} //namespace ReLuImpl

void RegisterReLuNodeExec(void)
{
    NodeExec relu_exec={ReLuImpl::OnBind,nullptr,ReLuImpl::Run,nullptr};

    RegisterNodeExec("ReLu",relu_exec);
}


} //namespace TEngine
