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

#include "logger.hpp"
#include "executor.hpp"
#include "tensor_mem.hpp"

#include "graph.hpp"
#include "operator/convolution.hpp"

#include "conv_implementor.hpp"



namespace TEngine {


namespace conv_interface {

#define CONV_IMPL "conv_impl"

bool PrerunConv(Node * node, ExecEngine * engine)
{
     /* select correct implementor*/
    Convolution * conv_op=dynamic_cast<Convolution *>(node->GetOp());
    ConvParam * param=conv_op->GetParam();

    Tensor * input_tensor=node->GetInputTensor(0);
    Tensor * weight_tensor=node->GetInputTensor(1);

    ConvImplementor * impl=ConvImplementorManager::Select(param,input_tensor,weight_tensor);

    if(impl==nullptr)
        return false;

    node->SetAttr(CONV_IMPL,impl);

    return impl->Prerun(node,engine);

}


bool RunConv(Node * node, ExecEngine * engine)
{
     ConvImplementor * impl=any_cast<ConvImplementor *>(node->GetAttr(CONV_IMPL));

     return impl->Run(node,engine);
}

bool PostrunConv(Node * node, ExecEngine * engine)
{
     ConvImplementor * impl=any_cast<ConvImplementor *>(node->GetAttr(CONV_IMPL));

     return impl->Postrun(node,engine);
}


static NodeExec node_exec={nullptr,PrerunConv,RunConv,PostrunConv};


} //namespace conv_interface


// the implementor initialize function 
extern void RegisterConv2dDepth(void);
extern void RegisterConvFast();

void RegisterConvolutionNodeExec(void)
{
   
   RegisterConv2dDepth();
   RegisterConvFast();

   RegisterNodeExec("Convolution",conv_interface::node_exec);
}

} //namespace TEngine


