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
 * Author: haitao@openailab.com
 */
#include <algorithm>

#include "graph.hpp"
#include "cpu_runner.hpp"
#include "tensor_mem.hpp"
#include "prof_record.hpp"
#include "graph_optimizer.hpp"
#include "cpu_driver.hpp"
#include "operator/convolution.hpp"

namespace TEngine {

#define ENABLE_TIME_PROFILING


static std::unordered_map<std::string, CPUInfo> predefined_list;

using tensor_map_t=std::unordered_map<Tensor *, int>;
using tensor_addr_t=std::unordered_map<void *, int>;


struct MemPool {


	struct MemBlock {
		void* addr;
		int  size;
		int  ref_count;
		int  alloc_count; 
	};



	MemPool(const std::vector<int>& mem_block, mem_alloc_t alloc_func, mem_free_t free_func) 
	{
		mem_alloc=alloc_func;
		mem_free=free_func;

		int block_number=mem_block.size();

		bool max_block_policy=false;

		const char * block_policy=std::getenv("MAX_BLOCK_POLICY");

		if(block_policy && block_policy[0]=='1')
			max_block_policy=true;

		if(max_block_policy)
		{ 
			int block_size=mem_block[block_number-1];

			for(int i=0;i<block_number;i++)
			{
				MemBlock b;
				b.addr=mem_alloc(block_size+128);
				b.size=block_size;
				b.ref_count=0;
				b.alloc_count=0;

				block_list.push_back(b);
			}
		}
		else
		{    

			//fill block number of least one, to ensure the bigger ones won't be occupied
			for(int i=0;i<block_number;i++)
			{
				MemBlock b;
				b.addr=mem_alloc(mem_block[0]+128);
				b.size=mem_block[0];
				b.ref_count=0;
				b.alloc_count=0;

				block_list.push_back(b);
			}

			for(int i=1;i<block_number;i++)
			{
				MemBlock b;
				b.addr=mem_alloc(mem_block[i]+128);
				b.size=mem_block[i];
				b.ref_count=0;

				block_list.push_back(b);
			}

		}
	}

	~MemPool()
	{
		for(unsigned int i=0;i< block_list.size();i++)
		{
			void * addr=block_list[i].addr;
			mem_free(addr);

			//printf("block [%d %p %d] allocated[%d]\n", 
			//		i,addr,block_list[i].size,block_list[i].alloc_count);
		}
	}

	void * Allocate(Tensor * tensor, int size)
	{
		int block_num=block_list.size();

		MemBlock *p_block=nullptr;
		int i;

		for(i=0;i<block_num;i++)
		{
			if(block_list[i].ref_count==0 &&
					block_list[i].size>=size)
			{
				p_block=&block_list[i];
				break;
			}
		}

		if(p_block==nullptr)
		{
			XLOG_ERROR()<<"cannot allocate memory for tensor: "<<tensor->GetName()<<"\n";
			return nullptr;
		}

		int ref_count=tensor->consumer.size()?tensor->consumer.size():1;
		p_block->ref_count=ref_count;
		p_block->alloc_count++;

		//record this 
		addr_map[p_block->addr]=i;

		return p_block->addr;
	}

	void AddRef(Tensor * tensor)
	{
		void * addr=get_tensor_mem(tensor);

		if(addr_map.count(addr)==0)
			return;

		auto ir=addr_map.find(addr);

		MemBlock * p_block=&block_list[ir->second];

		p_block->ref_count+=tensor->consumer.size()?tensor->consumer.size():1;

	}

	void Free(Tensor * tensor)
	{

		void * addr=get_tensor_mem(tensor);

		if(addr_map.count(addr)==0)
			return;

		auto ir=addr_map.find(addr);

		int idx=ir->second;

		MemBlock * p_block=&block_list[idx];  

		p_block->ref_count--;

		if(p_block->ref_count==0)
		{
			addr_map.erase(ir);
		}
	}

	mem_alloc_t mem_alloc;
	mem_free_t  mem_free;

	std::vector<MemBlock> block_list;

	tensor_addr_t addr_map;

};


bool debug_graph=false;

bool CPURunner::Prerun(Subgraph  * sub_graph)
{

	if(!BindNodeOps(sub_graph))
		return false;

	if(!AllocateMem(sub_graph))
		return false;

	for(unsigned int i=0;i<sub_graph->seq_nodes.size();i++)
	{
		Node * node=sub_graph->seq_nodes[i];

		if(!node->ExistAttr(ATTR_NODE_OPS))
			continue;

		NodeOps  * node_ops=any_cast<NodeOps *>(node->GetAttr(ATTR_NODE_OPS));

		if(!node_ops->Prerun(node))
			return false;
	}

	return true;
}

#ifdef ENABLE_TIME_PROFILING

static void parse_node(void * data, int repeat_count, uint64_t total_time)
{
	Node * node=(Node *)data;
	Tensor * input_tensor=node->GetInputTensor(0);
	Tensor * output_tensor=node->GetOutputTensor(0);
	Operator * op=node->GetOp();
	TShape& ishape=input_tensor->GetShape();
	TShape& oshape=output_tensor->GetShape();
	printf(" Node_idx:%3d",node->GetNodeIndex());
	// float* outdata=(float *)get_tensor_mem(output_tensor);
	// float* indata=(float *)get_tensor_mem(input_tensor);
	// printf("\t%s\n",output_tensor->GetName().c_str());

	printf("\t%10s\t%4d x%4d x%6d  -> %4d x%4d x%6d\t",
			op->GetName().c_str(),
			ishape.GetC(), ishape.GetH(), ishape.GetW(),
			oshape.GetC(), oshape.GetH(), oshape.GetW());
	if(op->GetName()=="Convolution")
	{
		Convolution * conv_op=dynamic_cast<Convolution *>(node->GetOp());
		ConvParam*  param=conv_op->GetParam();
		printf("%2d x %d / %d_p%d",param->kernel_h,param->kernel_w,param->stride_h,param->pads[0]);
		// if(param->kernel_h==3 && param->stride_h==1)
		// {
		// 	printf(" [%d]",param->mth);
		// }
		// else
		// {
		// 	printf("    ");
		// }
		if(param->group!=1)
		{
			printf(" (%2d)",param->group);
		}
		else
		{
			printf("     ");
		}
	}
	else
	{
		printf("   x    /   _     ");
	}
	if(op->GetName()=="Convolution" || op->GetName()=="FullyConnected" || op->GetName()=="Deconvolution")
	{
		std::printf(" Mfops: %.2f\tRate: %.0f",1.0f*node->GetFops()/1000000, 
				1.0f*node->GetFops()*repeat_count/total_time);
	}

	// printf("\t%s\n",output_tensor->GetName().c_str());


}

#endif


bool CPURunner::Run(Subgraph * sub_graph) 
{
	std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;

#ifdef ENABLE_TIME_PROFILING
	ProfRecord * prof=nullptr;

	bool do_prof=false;

	const char * prof_env=std::getenv("PROF_TIME");

	if(prof_env && prof_env[0]=='1')
	{
		do_prof=true;

		if(sub_graph->ExistAttr("PROF_TIME"))
			prof=any_cast<ProfRecord *>(sub_graph->GetAttr("PROF_TIME"));
		else
		{
			prof=new ProfTime(seq_nodes.size(),parse_node);
			sub_graph->SetAttr("PROF_TIME",prof);
		}
	}

#endif
	bool ret=true;

	for(unsigned int i=0; i<seq_nodes.size();i++)
	{
		Node * node=seq_nodes[i];

		if(!node->ExistAttr(ATTR_NODE_OPS))
			continue;

		NodeOps * node_ops=any_cast<NodeOps*>(node->GetAttr(ATTR_NODE_OPS));

		/* dynamic shape process */
		if(node->IsDynamicShape() || node->InputReshaped())
		{

			int output_number=node->GetOutputNum();

			/* free output tensor first */

			for(int i=0;i<output_number;i++)
			{
				Tensor * tensor=node->GetOutputTensor(i);
				free_tensor_mem(tensor);
			}

			/* do infer shape */
			Operator * op=node->GetOp();

			std::vector<TShape> inputs;

			for(unsigned int i=0;i<node->GetInputNum();i++)
			{
				Tensor * tensor=node->GetInputTensor(i);
				inputs.push_back(tensor->GetShape());

				if(tensor->Reshaped())
					tensor->UpdateReshapeCount();
			}

			std::vector<TShape> outputs(output_number);

			if(!op->InferShape(inputs,outputs))
			{
				XLOG_ERROR()<<"infer shaped for node: "<<node->GetName()
					<<" op: "<<op->GetName()<<" failed\n";
				ret=false;
				break;
			}


			for(int i=0;i<output_number;i++)
			{
				Tensor * tensor=node->GetOutputTensor(i);
				TShape shape=tensor->GetShape();

				shape=outputs[i];

				tensor->Reshape(shape);
			}

			/* allocate output memory */

			for(int i=0;i<output_number;i++)
			{
				Tensor * tensor=node->GetOutputTensor(i);

				int input_idx=-1;

				if(node->ExistAttr(ATTR_INPLACE))
				{
					const inplace_t & inplace=any_cast<inplace_t>(node->GetAttr("inplace"));

					if(inplace.count(i))
						input_idx=inplace.at(i);
				}

				if(input_idx>=0)
				{
					Tensor * input_tensor=node->GetInputTensor(input_idx);

					if(input_tensor->consumer.size()==1)
					{
						void * tensor_addr=get_tensor_mem(input_tensor);
						int total_size=tensor->GetTotalSize();

						set_tensor_mem(tensor,tensor_addr, total_size,nullptr);

						continue;
					}

				}

				//non-inplace or cannot do in-place
				int total_size=tensor->GetTotalSize();
				void * tensor_addr=mem_alloc(total_size);

				set_tensor_mem(tensor,tensor_addr,total_size,mem_free); 

			}

			/* call the Reshape() to prepare for run */
			node_ops->Reshape(node);

		}



#ifdef ENABLE_TIME_PROFILING
		if(do_prof)
			prof->Start(i,node);
#endif


		if(!node_ops->Run(node))
		{
			Operator * op=node->GetOp();
			LOG_ERROR()<<"Failed to execute on: "<<node->GetName() <<" Op: "<<op->GetName()<<std::endl;
			ret=false;
			break;
		}

#ifdef ENABLE_TIME_PROFILING
		if(do_prof)
			prof->Stop(i);
#endif

	}

#if 0

	std::printf("master cpu: %d run subgraph: %s --  %s\n",cpu_info_->GetMasterCPU(),
			sub_graph->GetName().c_str(),ret?"OK":"FAIL");

#endif
	return ret;
}

bool CPURunner::Postrun(Subgraph * sub_graph)
{
	std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;

	for(unsigned int i=0; i<seq_nodes.size();i++)
	{
		Node * node=seq_nodes[i];

		if(!node->ExistAttr(ATTR_NODE_OPS))
			continue;

		NodeOps * node_ops=any_cast<NodeOps *>(node->GetAttr(ATTR_NODE_OPS));

		if(!node_ops->Postrun(node))
		{
			LOG_ERROR()<<"Postrun failed for node: "<<node->GetName()<<"\n";
		}
	}


#ifdef ENABLE_TIME_PROFILING

	bool do_prof=false;

	const char * prof_env=std::getenv("PROF_TIME");

	if(prof_env && prof_env[0]=='1')
		do_prof=true;

	if(do_prof)
	{
		std::unordered_map<std::string,uint64_t > time_stats;
		ProfRecord * prof=any_cast<ProfRecord *>(sub_graph->GetAttr("PROF_TIME"));
		ProfTime * time_prof=dynamic_cast<ProfTime *>(prof);
		float total_fops=0;
		int repeat_count=1;

		int number=time_prof->GetRecordNum();
		uint64_t total_time=0;

		for(int i=0;i<number;i++)
		{
			const ProfTime::TimeRecord * p_record=time_prof->GetRecord(i);

			if(p_record->count==0)
				continue;

			total_time+=p_record->total_used_time;
			repeat_count=p_record->count;

			Node * node=reinterpret_cast<Node *>(p_record->ident);
			Operator * op=node->GetOp();

			uint64_t op_time;

			if(time_stats.count(op->GetName()))
				op_time=time_stats[op->GetName()];
			else
				op_time=0;

			op_time+=p_record->total_used_time;

			time_stats[op->GetName()] = op_time;
			if (op->GetName() == "Convolution" || op->GetName() == "FullyConnected" || op->GetName()=="Deconvolution")
			{
				total_fops += node->GetFops();
			}
		}

		std::printf("\n==== %s: time stats by operator: ====\n",
				sub_graph->GetName().c_str());

		std::printf("total time: %lu us, repeat %d\n",(unsigned long)total_time,repeat_count);

		std::printf("PER RUN: time %lu us on %.2f Mfops, RATE: %.2f Mfops\n",
				(unsigned long)(total_time/repeat_count),total_fops/1000000,
				total_fops*repeat_count/total_time);

		int n=0;

		for(auto ir=time_stats.begin(); ir!=time_stats.end();ir++)
		{
			std::printf("%d: %s used %lu us (%.2f%%)\n",n++,
					ir->first.c_str(),(unsigned long)ir->second, 100.0f*ir->second/total_time);
		}
		std::printf("\n\n");
		int method = 1;
		const char * debug_env=std::getenv("DEBUG_G");
		if((debug_env) && (debug_env[0]=='1'))
		{
			debug_graph=true;
		}
		if (debug_graph) method = 0;

		prof->Dump(method);

		delete prof;

		sub_graph->RemoveAttr("PROF_TIME");

	}
#endif

	FreeMem(sub_graph);
	UnbindNodeOps(sub_graph);

	return true;
}

bool CPURunner::FreeMem(Subgraph * sub_graph)
{
	std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;

	for(unsigned int i=0; i<seq_nodes.size();i++)
	{
		Node * node=seq_nodes[i];

		for(unsigned int i=0;i<node->GetOutputNum();i++)
		{
			Tensor * tensor=node->GetOutputTensor(i);
			free_tensor_mem(tensor);
		}
	}

	if(sub_graph->ExistAttr("shared_temp_memory"))
	{
		void * mem_addr=any_cast<void *>(sub_graph->GetAttr("shared_temp_memory"));

		mem_free(mem_addr);

		sub_graph->RemoveAttr("shared_temp_memory");
	}


	MemPool * mem_pool=any_cast<MemPool *>(sub_graph->GetAttr("MemPool"));
	delete mem_pool;

	sub_graph->RemoveAttr("MemPool");

	return true;
}

bool CPURunner::UnbindNodeOps(Subgraph * sub_graph)
{
	std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;

	for(unsigned int i=0; i<seq_nodes.size();i++)
	{
		Node * node=seq_nodes[i];

		if(!node->ExistAttr(ATTR_NODE_OPS))
			continue;

		NodeOps * node_ops=any_cast<NodeOps *>(node->GetAttr(ATTR_NODE_OPS));

		node_ops->OnUnbind(node);

		node_ops->Release();

		node->RemoveAttr(ATTR_NODE_OPS);
	}

	return true;
}



bool CPURunner::OptimizeGraph(Subgraph  * optimized_graph)
{
	GraphOptimizerManager::RunOpt("BNScale",optimized_graph);
	GraphOptimizerManager::RunOpt("ConvBN",optimized_graph);
	GraphOptimizerManager::RunOpt("ConvReLu",optimized_graph);

	return true;
}


static void CalculateMemBlocks(std::vector<int>& mem_block, Subgraph * sub_graph)
{
	//first calculate max var tensor exists
	tensor_map_t tensor_map;

	const std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;

	int node_number=seq_nodes.size();
	int max_active_num=0;
	int active_num=0;

	for(int i=0;i<node_number;i++)
	{
		Node * node=seq_nodes[i];

		//first, add output tensor into map
		if(!node->IsDynamicShape() && node->ExistAttr(ATTR_NODE_OPS))
		{
			for(unsigned int j=0;j<node->GetOutputNum();j++)
			{
				Tensor * tensor=node->GetOutputTensor(j);

				if(get_tensor_mem(tensor))
					continue;

				int consumer_number=tensor->consumer.size();

				tensor_map[tensor]=consumer_number;
				active_num++;
			}
		}

		if(active_num>max_active_num)
			max_active_num=active_num;

		//second, reduce the active_num by  release input
		for(unsigned int j=0;j<node->GetInputNum();j++)
		{
			Tensor * tensor=node->GetInputTensor(j);

			if(tensor_map.count(tensor)==0)
				continue;

			auto ir=tensor_map.find(tensor);

			ir->second--;

			if(ir->second==0)
			{
				active_num--;
				tensor_map.erase(ir);
			}            

		}

	}


	//suppose each output node only has one output tensor
	if(active_num>(int)sub_graph->output_nodes.size())
	{
		XLOG_ERROR()<<"graph: "<<sub_graph->GetName()
			<<" active num is: "<<active_num<<" output nodes: "
			<<sub_graph->output_nodes.size()<<"\n";
	}

	//collect all tensor size
	std::vector<int> mem_record;

	for(int i=0;i<node_number;i++)
	{
		Node * node=seq_nodes[i];

		if(node->IsDynamicShape())
			continue;

		for(unsigned int j=0;j<node->GetOutputNum();j++)
		{
			Tensor * tensor=node->GetOutputTensor(j);

			if(get_tensor_mem(tensor))
				continue;

			mem_record.push_back(tensor->GetTotalSize());
		}
	}

	//sort mem_record
	std::sort(mem_record.begin(),mem_record.end(),std::greater<int>());

	//save the most max_active_num into mem_blocks
	for(int i=0;i<max_active_num;i++)
	{
		mem_block.insert(mem_block.begin(),mem_record[i]);
	}
}

bool CPURunner::AllocateMem(Subgraph * sub_graph)
{
	const std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;

	/* 
	   first, check if any nodes supports new memory interface 
	   this memory block is only for tempory use and so that it can be shared 
	   between operators
	 */


	unsigned int max_shared_mem_size=0;

	for(unsigned int i=0;i<seq_nodes.size();i++)
	{
		Node * node=seq_nodes[i];

		if(!node->ExistAttr(ATTR_NODE_OPS))
			continue;

		NodeOps * node_ops=any_cast<NodeOps *>(node->GetAttr(ATTR_NODE_OPS));
		unsigned int mem_size;

		if(node_ops->GetSharedMemorySize(node,mem_size) 
				&&  mem_size>max_shared_mem_size)
		{
			max_shared_mem_size=mem_size;
		}
	}

	if(max_shared_mem_size>0)
	{
		void * shared_memory=mem_alloc(max_shared_mem_size+128);
		sub_graph->SetAttr("shared_temp_memory",shared_memory);

		for(unsigned int i=0;i<seq_nodes.size();i++)
		{
			Node * node=seq_nodes[i];

			if(!node->ExistAttr(ATTR_NODE_OPS))
				continue;

			NodeOps * node_ops=any_cast<NodeOps *>(node->GetAttr(ATTR_NODE_OPS));

			unsigned int mem_size;

			if(node_ops->GetSharedMemorySize(node,mem_size))
				node_ops->SetSharedMemoryAddr(node,shared_memory,mem_size);
		}

		// std::cout<<"max shared memory: "<<max_shared_mem_size<<"\n";
	}


	/*
	 *  now, calculate the maximum input and output memory blocks to run the graph 
	 */

	std::vector<int> mem_blocks;

	CalculateMemBlocks(mem_blocks,sub_graph);

	MemPool * mem_pool= new MemPool(mem_blocks,mem_alloc,mem_free);

	sub_graph->SetAttr("MemPool",mem_pool);


	/*
	 *  Real allocate memory
	 *
	 */

	for(unsigned int i=0; i<seq_nodes.size();i++)
	{
		Node * node=seq_nodes[i];

		if(node->IsDynamicShape() || !node->ExistAttr(ATTR_NODE_OPS))
			continue;

		for(unsigned int i=0;i<node->GetOutputNum();i++)
		{
			Tensor * tensor=node->GetOutputTensor(i);

			if(get_tensor_mem(tensor))
				continue;

			int input_idx=-1;

			if(node->ExistAttr(ATTR_INPLACE))
			{
				const inplace_t & inplace=any_cast<inplace_t>(node->GetAttr("inplace"));

				if(inplace.count(i))
					input_idx=inplace.at(i);
			}

			if(input_idx>=0)
			{
				Tensor * input_tensor=node->GetInputTensor(input_idx);

				if(input_tensor->consumer.size()==1)
				{
					void * tensor_addr=get_tensor_mem(input_tensor);
					int total_size=tensor->GetTotalSize();
					set_tensor_mem(tensor,tensor_addr, total_size,nullptr);

					mem_pool->AddRef(tensor);

					continue;
				}
			}

			//still allocate memory
			{
				int total_size=tensor->GetTotalSize();
				void * tensor_addr=mem_pool->Allocate(tensor,total_size);
				set_tensor_mem(tensor,tensor_addr,total_size,nullptr);

			}
		}
		/* input tensor */
		for(unsigned int i=0;i<node->GetInputNum();i++)
		{
			Tensor * input_tensor=node->GetInputTensor(i);
			mem_pool->Free(input_tensor);
		}
	}

	return true;


}


bool CPURunner::BindNodeOps(Subgraph * sub_graph)
{
	std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;
	int node_size=seq_nodes.size();

	const ExecAttr * exec_attr=any_cast<const ExecAttr *>(sub_graph->GetAttr("exec_attr"));

	for(int i=0; i<node_size;i++)
	{
		Node * node=seq_nodes[i];
		Operator * op=node->GetOp();

		if(op->GetName()=="Const" || op->GetName()=="Input")
			continue;

		node->SetAttr(ATTR_EXEC_ATTR,exec_attr);

		NodeOps * node_ops=NodeOpsRegistryManager::FindNodeOps(cpu_info_,node);

		if(node_ops==nullptr)
		{
			LOG_ERROR()<<"failed to set node ops for node: "<<node->GetName()
				<<" op: "<<op->GetName()<<"\n";
			return false;
		}

		auto dispatch=std::bind(&CPUDevice::PushAiderTask,cpu_dev_,std::placeholders::_1,
				std::placeholders::_2);

		auto wait=std::bind(&CPUDevice::WaitDone,cpu_dev_);

		node_ops->SetHelper(mem_alloc,mem_free,dispatch,wait);

		node->SetAttr(ATTR_NODE_OPS,node_ops);

		node_ops->exec_attr=exec_attr;

		node_ops->OnBind(node);
	}


	return true;

}

void CPURunner::AttachCPUDevice(CPUDevice * cpu_dev)
{
	cpu_dev_=cpu_dev;
	cpu_info_=cpu_dev_->GetCPUInfo();

}


} //namespace TEngine
