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
#include <cmath>
#include <math.h>
#include <arm_neon.h>

#include "logger.hpp"
#include "node_ops.hpp"
#include "tensor_mem.hpp"
#include "graph.hpp"
#include "operator/eltwise.hpp"
#include "data_type.hpp"

namespace TEngine {

namespace EltwiseImpl {

struct Eltwise_param
{
	float *output;
	float *input0;
	float *input1;
	int type;
	int in_size0;
	int in_size1;
	int stride;

};

struct EltwiseOps : public NodeOps
{

	/*
		insiz0 <= in_size1 , input0's size is insize0, input1's size is in_size1
	*/
    bool kernel_run(float* output, float* input0, float* input1, int type, int in_size0, int in_size1,int stride)
    {
        float* out_ptr = output;
        float* in0 = input0;
        float* in1 = input1;
		int loop_time = 0;


        switch(type)
        {
            
            case ELT_SUM:
                if(in_size0 == 1)
                {
					float32x4_t data0 = vdupq_n_f32 (in0[0]);
					for(int i = 0; i < in_size1; i = i+ 4)
                    {
						float32x4_t data1 = vld1q_f32(in1 + i);
						float32x4_t sum = vaddq_f32(data0,data1);
                        vst1q_f32(out_ptr + i,sum);
                    }
					loop_time = in_size1 / 4;
					for(int i = loop_time * 4; i < in_size1;i++)
					{
						out_ptr[i] = in1[i] + in0[0];
					}
                }
                else if(in_size1 == in_size0)
                {
                    for(int i = 0; i < in_size1; i = i+ 4)
                    {
						float32x4_t data0 = vld1q_f32(in0 + i);
						float32x4_t data1 = vld1q_f32(in1 + i);
						float32x4_t sum = vaddq_f32(data0,data1);
                        vst1q_f32(out_ptr + i,sum);
                    }
					loop_time = in_size1 / 4;

                    for(int i = loop_time * 4; i < in_size1;i++)
					{
						out_ptr[i] = in1[i] + in0[i];
					}
                }
                else if(in_size0 < in_size1 && in_size0 != 1)
                {
                    for(int i = 0; i < in_size1; ++i)
                    {
                        *out_ptr++ = in1[i] + in0[i / stride];
                    }
                }
                else
                    return false;
                break;
            
            default:
                break;
        }

        return true;
    }
	bool Aider(int cpu, int seq, void* data)
	{
		Eltwise_param *param = (Eltwise_param*)data;
		return( kernel_run(param->output,param->input0,param->input1,param->type,param->in_size0,param->in_size1, param->stride));
	}
    bool Run(Node* node)

    {
        // input
        Tensor* input_tensor0 = node->GetInputTensor(0);
        int element_size = DataType::GetTypeSize(input_tensor0->GetDataType());
        float* input0 = (float*)get_tensor_mem(input_tensor0);
        
        Tensor* input_tensor1 = nullptr;
        float* input1 = nullptr;
        int in_size0 = input_tensor0->GetTotalSize() / element_size;
        int in_size1 = 0;
        if(node->GetInputNum() > 1)
        {
            input_tensor1 = node->GetInputTensor(1);
            input1 = (float*)get_tensor_mem(input_tensor1);
            in_size1 = input_tensor1->GetTotalSize() / element_size;
        }
		TShape ishape;
		int max_size = 0;
		int min_size = 0;
		float * main_data = NULL;
		float * scale_data = NULL;

		int batch_num = ishape.GetN();
		if(in_size0 >= in_size1)
		{
			ishape = input_tensor0->GetShape();
			main_data =  input0;
			scale_data =  input1;
			max_size = in_size0/batch_num;
			min_size = in_size1/batch_num;
		}
		else
		{
			ishape = input_tensor1->GetShape();
			main_data =  input1;
			scale_data =  input0;
			max_size = in_size1/batch_num;
			min_size = in_size0/batch_num;
		}

        // this version only support for input_num=2
        // int input_number=node->GetInputNum();

        // output
        Tensor* output_tensor = node->GetOutputTensor(0);
        float* output = (float*)get_tensor_mem(output_tensor);
        Eltwise* eltwise_op = dynamic_cast<Eltwise*>(node->GetOp());
        EltwiseParam* elt_param = eltwise_op->GetParam();
        bool result = true;

		int stride = ishape.GetH() * ishape.GetW();
		int channel = ishape.GetC();
		int cpu_number = cpu_info->GetCPUNumber();

		for(int n = 0; n < batch_num; n++)
		{
			float * input_data0 = scale_data + n * in_size0;
			float * input_data1 = main_data + n * in_size1;

			//only support eltwise sum mt mode

			if(cpu_number == 1 || elt_param->type != ELT_SUM)
			{

				result = kernel_run(output, input_data0, input_data1, elt_param->type,min_size, max_size,stride);
			}
			else
			{
				std::vector<sub_op_task> task_list;
				std::vector<Eltwise_param> param_list;
				auto f = std::bind(&EltwiseOps::Aider, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
				int max_task_num = cpu_number;
				int step0 = 0;
				int step1 = 0;

				if(channel == min_size)  // vector
				{
					if(channel < cpu_number)
					{
						max_task_num = channel;
					}
					step0 = channel/max_task_num;
					step1 = step0 * stride;
				}
				else
				{
					if(max_size < cpu_number)
					{
						max_task_num = max_size;
					}
					if(max_size == min_size) // tensor
					{
						step0 = max_size/max_task_num;
						step1 = step0;
					}
					else // scale
					{
						step0 = 0;
						step1 = max_size/max_task_num;
					}
				}

                //per channel
				task_list.resize(max_task_num);
				param_list.resize(max_task_num);
				for(int i = 0; i < max_task_num; i++)
				{
					Eltwise_param *param = &param_list[i];
					sub_op_task* task = &task_list[i];

					task->exec_func = f;
					task->seq = i;
					task->data = param;

					param->output = output + i * step1;
                    param->input0 = scale_data + i * step0;
					param->input1 = main_data + i * step1;
					param->type = elt_param->type;

					if(0 == step0)
					{
						param->in_size0 = 1;

					}
					else
					{
						param->in_size0 = step0;
					}
					param->in_size1 = step1;
					param->stride = stride;
				}
				if(step0 != 0)
				{
					param_list[max_task_num - 1].in_size0 += (min_size - max_task_num * step0);
				}
				param_list[max_task_num - 1].in_size1 += (max_size - max_task_num * step1);
				task_dispatch(task_list, -1);
				wait_done();
			}
		}
        return result;
    }    // Run

};    // struct EltwiseOps
 
NodeOps* SelectFunc(const CPUInfo* cpu_info, Node* node)
{
    Tensor* input = node->GetInputTensor(0);
    const int data_type = input->GetDataType();
    if(data_type != TENGINE_DT_FP32)
    {
        return nullptr;
    }
	Eltwise* eltwise_op = dynamic_cast<Eltwise*>(node->GetOp());
    EltwiseParam* elt_param = eltwise_op->GetParam();
	if( elt_param->type != ELT_SUM )
	{
		return nullptr;
	}

    EltwiseOps* ops = new EltwiseOps();

    return ops;
}

}    // namespace EltwiseImpl

using namespace EltwiseImpl;

void RegisterEltwiseFP32(void)
{
    NodeOpsRegistryManager::RegisterOPImplementor("arm64", "Eltwise", EltwiseImpl::SelectFunc, 1000);
}

}    // namespace TEngine
