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
 * Author: haoluo@openailab.com
 */
#ifndef __ACL_GRAPH_HPP
#define __ACL_GRAPH_HPP

#include <array>
#include <random>
#include <string>
#include <vector>

#include "graph.hpp"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLFunctions.h"

#include "tensor_mem.hpp"
#include "operator/convolution.hpp"
#include "operator/pooling.hpp"
#include "operator/batch_norm.hpp"

using namespace arm_compute;

namespace TEngine{

class MYSoftmaxLayer : public IFunction
{
public :
	CLSoftmaxLayer _layer;
	CLTensor *input_org;
	CLTensor _input;

	void configure(CLTensor *input, CLTensor *output)
	{
		input_org = input;
		TensorInfo *info = input->info();
		int size = info->dimension(0)*info->dimension(1)*info->dimension(2);
		TensorShape shape = TensorShape(size);
		_input.allocator()->init(TensorInfo(shape,1,DataType::F32));
		_layer.configure(&_input, output);
		_input.allocator()->allocate();
	}
	void run()
	{
		TensorInfo *info = input_org->info();
		int size = info->dimension(0)*info->dimension(1)*info->dimension(2);
		input_org->map();
		_input.map();	
		float* src = reinterpret_cast<float*>(input_org->buffer());
		float* dst = reinterpret_cast<float*>(_input.buffer());
		int w_align = info->dimension(1) +info->padding().right;
		for(int i=0;i<size;i++)
		{
			dst[i] = src[i*w_align];
		}
		input_org->unmap();
		_input.unmap();	
		_layer.run();
	}
};


class CLGraph 
{
public:
	
	std::string name_;
	std::vector<IFunction*> functions_map_;
	std::unordered_map<std::string ,CLTensor*> tensors_map_;

	
	CLGraph(std::string name){
		name_ = name;
	};

	~CLGraph()
	{
		functions_map_.clear();
	}
	
	void Run(void)
	{
		int size = functions_map_.size();
		for(int i =0;i<size;i++)
		{
			functions_map_[i]->run();
		}
		
	}

	bool AddInputLayer(Node * node)
	{
	/* output */
		Tensor* tensor=node->GetOutputTensor(0);
		std::string name = tensor->GetName();
		std::vector<int> dim_w = tensor->GetShape().GetDim();
		
		CLTensor *itensor = new CLTensor();
		itensor->allocator()->init(
				TensorInfo(TensorShape(dim_w[2],dim_w[3],dim_w[1],dim_w[0]),1,DataType::F32));
		tensors_map_[name]=itensor;
		
		itensor->allocator()->allocate();
		return true;
	}

	bool AddConvolutionLayer(Node* node)
	{
		float* acl_data=nullptr;
		float* data=nullptr;
		Convolution * conv_op=dynamic_cast<Convolution *>(node->GetOp());
		ConvParam*  param=conv_op->GetParam();
		
		int  pad_x       = param->pad_w;
		int  pad_y       = param->pad_h;
		int  stride_x    = param->stride_w;
		int  stride_y    = param->stride_h;
		int  group 		 = param->group;
		int  outchan     = param->output_channel;
		
		/* input */
		Tensor* input_tensor=node->GetInputTensor(0);
		std::string name = input_tensor->GetName();
		CLTensor *itensor = nullptr;
		if(tensors_map_.count(name))
		{
			itensor = tensors_map_[name];
		}
		else
		{
			LOG_DEBUG()<< "Can't find node ["<< node->GetName() <<"] tensor named :" << name <<"\n";
			return false;
		}
		/* weight */
		Tensor* w_tensor=node->GetInputTensor(1);
		std::vector<int> dim_w = w_tensor->GetShape().GetDim();
		name = w_tensor->GetName();
		CLTensor *wtensor = new CLTensor();
		wtensor->allocator()->init(
				TensorInfo(TensorShape(dim_w[2],dim_w[3],dim_w[1],dim_w[0]),1,DataType::F32));
		tensors_map_[name]=wtensor;

		/* bias */
		Tensor* b_tensor=node->GetInputTensor(2);
		CLTensor *btensor = nullptr;
		if(b_tensor)
		{
			int channel = b_tensor->GetShape().GetSize();
			
			name = b_tensor->GetName();
			btensor = new CLTensor();
			btensor->allocator()->init(
					TensorInfo(TensorShape(channel,1,1,1),1,DataType::F32));
			tensors_map_[name]=btensor;
		}

		/* output */
		Tensor* o_tensor=node->GetOutputTensor(0);
		TensorInfo* info = itensor->info();
		int out_h = (info->dimension(0) -dim_w[2]+2*pad_y)/stride_y +1;
		int out_w = (info->dimension(1) -dim_w[3]+2*pad_x)/stride_x +1;
		name = o_tensor->GetName();
		CLTensor *otensor = new CLTensor();
		otensor->allocator()->init(
				TensorInfo(TensorShape(out_h,out_w,outchan,1),1,DataType::F32));
		tensors_map_[name]=otensor;

		/* configure */
		if(group > 1 && group == outchan)
		{
			if(3==dim_w[2] && 3 == dim_w[3])
			{
				CLDepthwiseConvolutionLayer3x3* dwconv3x3 = new CLDepthwiseConvolutionLayer3x3();
				dwconv3x3->configure(itensor,wtensor,btensor,otensor,
							PadStrideInfo(stride_x,stride_y,pad_x,pad_y));
				functions_map_.push_back(dwconv3x3);
			}
			else
			{
				CLDepthwiseConvolutionLayer* dwconv = new CLDepthwiseConvolutionLayer();
				dwconv->configure(itensor,wtensor,btensor,otensor,
							PadStrideInfo(stride_x,stride_y,pad_x,pad_y));
				functions_map_.push_back(dwconv);
			}
		}
		else
		{
			CLConvolutionLayer* clconv = new CLConvolutionLayer(); 
			clconv->configure(
				itensor,wtensor,btensor,otensor,PadStrideInfo(stride_x,stride_y,pad_x,pad_y));
			functions_map_.push_back(clconv);
		}
		wtensor->allocator()->allocate();
		wtensor->map();
		data = (float*)get_tensor_mem(w_tensor);
		acl_data = reinterpret_cast<float*>(wtensor->buffer());
		int size = w_tensor->GetTotalSize();
		memcpy(acl_data,data,size);
		wtensor->unmap();
		if(btensor)
		{
			btensor->allocator()->allocate();
			btensor->map();
			data = (float*)get_tensor_mem(b_tensor);
			acl_data = reinterpret_cast<float*>(btensor->buffer());
			int size = b_tensor->GetTotalSize();
			memcpy(acl_data,data,size);
			btensor->unmap();
		}


		return true;
		
	}

	bool AddReLuLayer(Node* node)
	{
		Tensor* input_tensor=node->GetInputTensor(0);
		std::string name = input_tensor->GetName();
		CLTensor *itensor = nullptr;
		if(tensors_map_.count(name))
		{
			itensor = tensors_map_[name];
		}
		else
		{
			LOG_INFO()<< "can't find node ["<< node->GetName() <<"]tensor named :" << name <<"\n";
			return false;
		}
		
		Tensor* out_tensor=node->GetOutputTensor(0);
		name = out_tensor->GetName();
		CLTensor *otensor = new CLTensor();
		otensor->allocator()->init(*(itensor->info()));
		tensors_map_[name]= otensor;
		
		CLActivationLayer * relu = new CLActivationLayer();
		relu->configure(itensor, otensor ,ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		functions_map_.push_back(relu);
		
		return true;
	}
	
	bool AddConcatLayer(Node* node)
	{
		std::vector<ICLTensor *> inputs_vector;
		
		Tensor* tensor1=node->GetInputTensor(0);
		std::string name = tensor1->GetName();
		CLTensor *itensor1 = nullptr;
		if(tensors_map_.count(name))
		{
			itensor1 = tensors_map_[name];
		}
		else
		{
			LOG_INFO()<< "can't find node ["<< node->GetName() <<"]tensor named :" << name <<"\n";
			return false;
		}
		inputs_vector.push_back(itensor1);

		Tensor* tensor2=node->GetInputTensor(1);
		name = tensor2->GetName();
		CLTensor *itensor2 = nullptr;
		if(tensors_map_.count(name))
		{
			itensor2 = tensors_map_[name];
		}
		else
		{
			LOG_INFO()<< "can't find node ["<< node->GetName() <<"]tensor named :" << name <<"\n";
			return false;
		}
		inputs_vector.push_back(itensor2);

	/*output */
		TensorInfo* info = itensor1->info();
		Tensor* out=node->GetOutputTensor(0);
		name = out->GetName();
		CLTensor *otensor = new CLTensor();
		otensor->allocator()->init(
				TensorInfo(TensorShape(info->dimension(0),info->dimension(1),info->dimension(2)*2,info->dimension(3)),
							1,DataType::F32));
		tensors_map_[name]=otensor;
		
		CLDepthConcatenateLayer* concat = new CLDepthConcatenateLayer();
		concat->configure(inputs_vector,otensor);
		functions_map_.push_back(concat);
		
		return true;
	}
	
	bool AddPoolingLayer(Node* node)
	{
		Pooling * pool_op=dynamic_cast<Pooling *>(node->GetOp());
		PoolParam*  param=pool_op->GetParam();
		int  pad_x		 = param->pad_w;
		int  pad_y		 = param->pad_h;
		int  stride_x	 = param->stride_w;
		int  stride_y	 = param->stride_h;
		int  kernel_w 	 = param->kernel_w;
		int  kernel_h 	 = param->kernel_h;
		int  type 		 = param->alg;
		int  global 	 = param->global;

		Tensor* input_tensor=node->GetInputTensor(0);
		int channel = input_tensor->GetShape().GetC();
		std::string name = input_tensor->GetName();
		CLTensor *itensor = nullptr;
		if(tensors_map_.count(name))
		{
			itensor = tensors_map_[name];
		}
		else
		{
			LOG_INFO()<< "can't find node ["<< node->GetName() <<"]tensor named :" << name <<"\n";
			return false;
		}

		
		/* output */
		Tensor* o_tensor=node->GetOutputTensor(0);
		
		TensorInfo* info = itensor->info();
		int out_h = std::ceil((float)(info->dimension(0) -kernel_h+2*pad_y)/stride_y) +1;
		int out_w = std::ceil((float)(info->dimension(1) -kernel_w+2*pad_x)/stride_x) +1;
		name = o_tensor->GetName();
		
		CLTensor *otensor = new CLTensor();
		otensor->allocator()->init(
				TensorInfo(TensorShape(out_h,out_w,channel,1),1,DataType::F32));
		tensors_map_[name]=otensor;
	
		CLPoolingLayer * pooling = new CLPoolingLayer();
		PoolingLayerInfo pooling_info;
		if(global)
			pooling_info = PoolingLayerInfo(type?PoolingType::AVG:PoolingType::MAX);
		else
			pooling_info = PoolingLayerInfo(type?PoolingType::AVG:PoolingType::MAX, Size2D(kernel_w,kernel_h ), 
					PadStrideInfo(stride_x,stride_y,pad_x,pad_y,DimensionRoundingType::CEIL));

		pooling->configure(itensor, otensor , pooling_info);
		
		functions_map_.push_back(pooling);
		
		return true;
	}
	
	bool AddBNLayer(Node* node,Node* node_scale)
	{
		BatchNorm * bn_op=dynamic_cast<BatchNorm *>(node->GetOp());
		BatchNormParam * param=bn_op->GetParam();
    	float eps=param->eps;
		
		/* input */
		Tensor* input_tensor=node->GetInputTensor(0);
		std::string name = input_tensor->GetName();
		int channel = input_tensor->GetShape().GetC();
		CLTensor *itensor = nullptr;
		if(tensors_map_.count(name))
		{
			itensor = tensors_map_[name];
		}
		else
		{
			LOG_INFO()<< "can't find node ["<< node->GetName() <<"]tensor named :" << name <<"\n";
			return false;
		}

		/* gamma */
		Tensor* gamma_tensor=node_scale->GetInputTensor(1);
		CLTensor *gtensor = nullptr;
		if(gamma_tensor)
		{			
			name = gamma_tensor->GetName();
			gtensor = new CLTensor();
			gtensor->allocator()->init(
					TensorInfo(TensorShape(channel),1,DataType::F32));
			tensors_map_[name]=gtensor;
		}
		/* beta */
		Tensor* beta_tensor=node_scale->GetInputTensor(2);
		CLTensor *btensor = nullptr;
		if(beta_tensor)
		{			
			name = beta_tensor->GetName();
			btensor = new CLTensor();
			btensor->allocator()->init(
					TensorInfo(TensorShape(channel),1,DataType::F32));
			tensors_map_[name]=btensor;
		}
		

		/* means */
		Tensor* means_tensor=node->GetInputTensor(3);
		name = means_tensor->GetName();
		CLTensor *mtensor = new CLTensor();
		mtensor->allocator()->init(
				TensorInfo(TensorShape(channel),1,DataType::F32));
		tensors_map_[name]=mtensor;

		/* var */
		Tensor* var_tensor=node->GetInputTensor(4);
		CLTensor* vtensor = nullptr;
		if(var_tensor)
		{			
			name = var_tensor->GetName();
			vtensor = new CLTensor();
			vtensor->allocator()->init(
					TensorInfo(TensorShape(channel),1,DataType::F32));
			tensors_map_[name]=vtensor;
		}
/* output */
		Tensor* out_tensor=node_scale->GetOutputTensor(0);
		name = out_tensor->GetName();
		CLTensor *otensor = new CLTensor();
		otensor->allocator()->init(*(itensor->info()));
		tensors_map_[name]= otensor;
		

		CLBatchNormalizationLayer *bn = new CLBatchNormalizationLayer();
		bn->configure(itensor, otensor, mtensor, vtensor , btensor, gtensor, eps);

		functions_map_.push_back(bn);

		
		mtensor->allocator()->allocate();
		vtensor->allocator()->allocate();
		mtensor->map();
		vtensor->map();
		float* means_data = reinterpret_cast<float*>(mtensor->buffer());
		float* vars_data = reinterpret_cast<float*>(vtensor->buffer());
		float* means = (float*)get_tensor_mem(means_tensor);
		float* vars = (float*)get_tensor_mem(var_tensor);
		/*for(int i=0;i<channel;i++)
		{
			vars_data[i] = 1.f/std::sqrt(vars[i]*rescale + eps);
			means_data[i] = -means[i]*rescale*vars_data[i];
		}*/
		
		memcpy(means_data, means , channel*4);
		memcpy(vars_data, vars , channel*4);
		
		mtensor->unmap();
		vtensor->unmap();

		
		if(btensor)
		{
			btensor->allocator()->allocate();
			btensor->map();
			float* beta_data = reinterpret_cast<float*>(btensor->buffer());
			float* beta = (float*)get_tensor_mem(beta_tensor);
			memcpy(beta_data, beta , channel*4);
			btensor->unmap();
		}
		if(gtensor)
		{
			gtensor->allocator()->allocate();
			gtensor->map();
			float* gamma_data = reinterpret_cast<float*>(gtensor->buffer());
			float* gamma = (float*)get_tensor_mem(gamma_tensor);
			memcpy(gamma_data, gamma , channel*4);
			gtensor->unmap();
		}	
		
		return true;
	}
	
	bool AddDropoutLayer(Node* node)
	{
		Tensor* input_tensor=node->GetInputTensor(0);
		std::string name = input_tensor->GetName();
		CLTensor *itensor = nullptr;
		if(tensors_map_.count(name))
		{
			itensor = tensors_map_[name];
		}
		else
		{
			LOG_INFO()<< "can't find node ["<< node->GetName() <<"]tensor named :" << name <<"\n";
			return false;
		}

		/*output */
		Tensor* o_tensor=node->GetOutputTensor(0);
		name = o_tensor->GetName();	
		tensors_map_[name]=itensor;
		
		return true;
	}
	
	bool AddSoftmaxLayer(Node* node)
	{
		
		Tensor* input_tensor=node->GetInputTensor(0);
		std::string name = input_tensor->GetName();
		CLTensor *itensor = nullptr;
		if(tensors_map_.count(name))
		{
			itensor = tensors_map_[name];
		}
		else
		{
			LOG_INFO()<< "can't find node ["<< node->GetName() <<"] tensor named :" << name <<"\n";
			return false;
		}
		
		/*output */
		Tensor* o_tensor=node->GetOutputTensor(0);
		name = o_tensor->GetName();

		TensorInfo *info = itensor->info();
		int size = info->dimension(0)*info->dimension(1)*info->dimension(2);
		TensorShape shape(size);
		CLTensor *otensor = new CLTensor();
		otensor->allocator()->init(TensorInfo(shape,1,DataType::F32));
		tensors_map_[name]=otensor;
		
		MYSoftmaxLayer *softmax = new MYSoftmaxLayer();
		softmax->configure(itensor, otensor);
		functions_map_.push_back(softmax);
		
		return true;
	}

	CLTensor* GetCLTensor(std::string name)
	{
		return tensors_map_[name];
	}
	
};



}

#endif  // __ACL_GRAPH_HPP


