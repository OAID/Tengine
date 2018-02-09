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
#include <cstring>
#include <algorithm>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/eltwise.hpp"



namespace TEngine 
{

namespace EltwiseImpl 
{

struct EltwiseOps : public NodeOps  {


bool Run(Node * node)
{
    //input
    Tensor * input_tensor0=node->GetInputTensor(0);
    const TShape& ishape=input_tensor0->GetShape();
    int input_count4=ishape.GetSize();
    void * input0=get_tensor_mem(input_tensor0);

    Tensor * input_tensor1=node->GetInputTensor(1);
    void* input1=get_tensor_mem(input_tensor1);

    // this version only support for input_num=2
   // int input_number=node->GetInputNum();

    // output
    Tensor * output_tensor=node->GetOutputTensor(0);
    void * output=get_tensor_mem(output_tensor);
    float* out_ptr=(float*)output;
    float* in0=(float*)input0;
    float* in1=(float*)input1;
    Eltwise * eltwise_op=dynamic_cast<Eltwise *>(node->GetOp());
    EltwiseParam*  param=eltwise_op->GetParam();
            
    switch (param->type)
	{
	case ELT_SUM:
        
		for (int i = 0; i < input_count4; ++i)
		{
			*out_ptr++ = (*in0++)+(*in1++);
        }
		break;
	case ELT_MAX:
		for (int i = 0; i < input_count4; ++i)
		{
			*out_ptr++ =std::max (*in0,*in1);
            in0++;
            in1++;
        }
		break;
    case ELT_PROD:
		{}
		break;
    }
    return true;
}// Run

};

} //namespace EltwiseImpl

using namespace EltwiseImpl;

void RegisterEltwiseNodeExec(void)
{
   EltwiseOps * ops=new EltwiseOps();

   NodeOpsRegistryManager::RegisterOPImplementor("arm64",
               "Eltwise",ops);
}


} //namespace TEngine
