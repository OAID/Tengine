
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
* Author: haoluo@openailab.com
*/

#include "acl_graph_device.hpp"

using namespace arm_compute;

namespace TEngine {

static CLGraph* CreateACLGraph(Subgraph *graph)
{
	CLScheduler::get().default_init();
	CLGraph* acl_graph = new CLGraph(graph->GetName());

	int node_size=graph->seq_nodes.size();
	int i =0;
	for(i=0; i<node_size;i++)
	{
		bool ret = false;
		Node * node=graph->seq_nodes[i];
		Operator* op = node->GetOp();
		std::string name = op->GetName();
		if(name =="Const" )
			continue;
		//std::cout<<"node name: "<<node->GetName()<<" ,op name:	"<<name <<"\n";
		if(name =="Input")
		{
			ret = acl_graph->AddInputLayer( node);
		}
		else if(name =="Convolution")
		{
			ret = acl_graph->AddConvolutionLayer(node );
		}
		else if(name == "ReLu")
		{
			ret = acl_graph->AddReLuLayer( node);
		}
		else if(name == "Pooling")
		{
			ret = acl_graph->AddPoolingLayer( node);
		}
		else if(name == "Concat")
		{
			ret = acl_graph->AddConcatLayer( node);
		}
		else if(name == "Dropout")
		{
			ret = acl_graph->AddDropoutLayer(node);
		}
		else if(name == "Softmax")
		{
			ret = acl_graph->AddSoftmaxLayer(node);
		}
		else if(name == "BatchNormalization")
		{
			Node * node_next=graph->seq_nodes[++i];
			if(node_next->GetOp()->GetName() != "Scale")
				ret = false;
			else
				ret = acl_graph->AddBNLayer(node,node_next);
			break;
		}
		if(!ret)
		{
			LOG_INFO()<<"Create ACL for Op "<<name <<" failed! \n";
			return nullptr;
		}
	}
		
	return acl_graph;
}

bool ACLDevice::RealOptimizeGraph(DevContext * context, Subgraph * graph)
{
    context->optimized_graph=graph;

    GraphOptimizerManager::RunOpt("BNScale",context->optimized_graph);
	GraphOptimizerManager::RunOpt("ConvBN",context->optimized_graph);
	
    return true;
}

bool ACLDevice::RealPrerun(DevContext * context)
{
	CLGraph *graph = CreateACLGraph(context->sub_graph);
	context->graph = graph;
	
	if(graph == nullptr)
		return false;
	
	auto ir_start = graph->tensors_map_.begin();
	auto ir_end = graph->tensors_map_.end();
	
	for(auto ir =ir_start;ir!=ir_end;ir++)
	{
		CLTensor* tensor = ir->second;
		std::string name = ir->first;
		if(name.find("weight")!= std::string::npos ||
			name.find("gamma") != std::string::npos ||
			name.find("beta") != std::string::npos ||
			name.find("means") != std::string::npos ||
			name.find("vars") != std::string::npos ||
			name.find("bias") != std::string::npos ||
			name.find("data") != std::string::npos  )
			continue;
		if(tensor->allocator()->info().is_resizable())
			tensor->allocator()->allocate();
	}
	
	
	int node_size =context->sub_graph->seq_nodes.size();
	Node *node = context->sub_graph->seq_nodes[node_size-1];
	Tensor *output = node->GetOutputTensor(0);
	void * mem_addr = get_tensor_mem(output);
	if(mem_addr)
		return true;
	else
	{
		std::string name = output->GetName();
		TShape& shape=output->GetShape();
		TensorInfo *info = graph->GetCLTensor(name)->info();
		std::vector<int> output_dims={(int)(info->dimension(3)),
				(int)(info->dimension(2)),
				(int)(info->dimension(0)),
				(int)(info->dimension(1))};
		shape.SetDim(output_dims);
		void* addr = std::malloc(output->GetTotalSize());
		set_tensor_mem(output , addr, output->GetTotalSize(), nullptr);
	}
	return true;

}


bool ACLDevice::RealSyncRun( DevContext * context )
{
	return true;
}

bool ACLDevice::RealRun( DevContext * context )
{
	int node_size = context->sub_graph->seq_nodes.size();
	CLGraph *graph = context->graph;
	
	Node * node = context->sub_graph->input_nodes[0];
	Tensor * out = node->GetOutputTensor(0);
	std::string name = out->GetName();
	CLTensor *acl_input = graph->GetCLTensor(name);
	acl_input->map();
	float* acl_buf = reinterpret_cast<float*>(acl_input->buffer());
	float* buf = (float*)get_tensor_mem(out);
	int size = out->GetTotalSize();
	std::memcpy(acl_buf, buf , size);
	acl_input->unmap();
	

	graph->Run();

	node = context->sub_graph->seq_nodes[node_size-1];
	Tensor *output = node->GetOutputTensor(0);
	std::string output_name = output->GetName() ;
	CLTensor *cltensor =graph->GetCLTensor(output_name);

	int out_size = (output->GetTotalSize())>>2;
	
	float* output_buf = (float*)get_tensor_mem(output);
	cl::copy<float*>(cltensor->cl_buffer(),output_buf,output_buf+out_size);

	return true; 

}


bool ACLDevice::RealPostrun( DevContext * context )
{
	CLGraph * graph = context->graph;
	auto ir_start = graph->tensors_map_.begin();
	auto ir_end = graph->tensors_map_.end();
	
	for(auto ir =ir_start;ir!=ir_end;ir++)
	{
		CLTensor* tensor = ir->second;
		if(!tensor->allocator()->info().is_resizable())
			tensor->allocator()->free();
	}
	return true;
}

void ACLDevice::RunGraph(DevContext * context, dev_graph_cb_t graph_cb)
{
    bool ret=RealRun(context);

 	if(graph_cb)
 		graph_cb(context->optimized_graph,ret);
}

void ACLDevice::Process(const acl_task& task, int acl_id)
{
	
    RunGraph(task.context,task.context->graph_cb);
}

void ACLDevice::Launch(void)
{
     auto f=std::bind(&ACLDevice::Process,this,std::placeholders::_1, std::placeholders::_2);

     thread_=new WorkerThread<acl_task>(f);

     thread_->SetQueue(&task_queue_,&queue_lock_,&queue_cv_);

     thread_->LaunchWorker();

}

void ACLDevice::IncRequest(int req_number)
{
     request_+=req_number;
}

void ACLDevice::IncDone(int done_number)
{
     uint64_t prev_val=done_.fetch_add(done_number);

     if(prev_val+done_number== request_)
     {
           std::unique_lock<std::mutex> lock(wait_mutex_);

           wait_cv_.notify_all();

           lock.unlock();

     }
}

void ACLDevice::PushTask(std::vector<acl_task>& task_list)
{
    thread_->PushTask(task_list);
}

void ACLDevice::WaitDone(void)
{
     std::unique_lock<std::mutex> lock(wait_mutex_);

     if(done_!=request_)
           wait_cv_.wait(lock,[this]{return done_==request_;});

     lock.unlock(); 

}

void ACLDevice::Kill(void)
{
   if(thread_)
   {
      delete thread_;
      thread_=nullptr;
   }
}


}
