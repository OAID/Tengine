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
#include "graph.hpp"
#include "soc_runner.hpp"
#include "tensor_mem.hpp"
#include "prof_record.hpp"
#include "graph_optimizer.hpp"

namespace TEngine {

#define ENABLE_TIME_PROFILING

using GraphContext=CPURunner::GraphContext;

static std::unordered_map<std::string, SocInfo> predefined_list;

bool GetPredefinedSoc(const std::string& soc_name, SocInfo& soc_info)
{
	if(predefined_list.count(soc_name)==0)
		return false;

	soc_info=predefined_list.at(soc_name);

	return true;
}

bool RegisterPredefinedSoc(const std::string& soc_name, const SocInfo& soc_info)
{
	if(predefined_list.count(soc_name))
		return false;

	predefined_list[soc_name]=soc_info;

	return true;
}   


bool CPURunner::SetWorkingCPU(const std::vector<int>& cpu_list, int master)
{
	soc_info.SetWorkingCPU(cpu_list,master);

	return true;
}

void * CPURunner::CreateGraphHandle(Subgraph * sub_graph)
{
	GraphContext * context=new GraphContext();

	context->optimized_graph=nullptr;

	context->orig_graph=sub_graph;

	return context;
}

void CPURunner::ReleaseGraphHandle(void * graph_handle)
{
	GraphContext * context=reinterpret_cast<GraphContext *>(graph_handle);

	if(context->optimized_graph &&
			context->optimized_graph!=context->orig_graph)
	{
		delete context->optimized_graph;
	}

	delete context;
}

void CPURunner::SetHelper(const mem_alloc_t& alloc, const mem_free_t& free,const task_dispatch_t&  dispatch)
{
	mem_alloc=alloc;
	mem_free=free;
	task_dispatch=dispatch;
}

bool CPURunner::Prerun(void * graph_handle)
{
	GraphContext * context=reinterpret_cast<GraphContext *>(graph_handle);

	Subgraph * sub_graph= context->optimized_graph;

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

	std::printf("Node: %d %s ",node->GetNodeIndex(),node->GetName().c_str());

	std::printf(" op: %s",node->GetOp()->GetName().c_str());

	std::cout<<" input: ";
	Tensor * input_tensor=node->GetInputTensor(0);
	TShape& ishape=input_tensor->GetShape();
	ishape.DumpShape(std::cout);

	std::cout<<" output: ";

	Tensor * output_tensor=node->GetOutputTensor(0);
	TShape& oshape=output_tensor->GetShape();
	oshape.DumpShape(std::cout);

	std::printf(" Mfops: %.2f Rate: %.0f",1.0f*node->GetFops()/1000000, 
               1.0f*node->GetFops()*repeat_count/total_time);
}

#endif


bool CPURunner::Run(void * graph_handle) 
{
	GraphContext * context=reinterpret_cast<GraphContext *>(graph_handle);

	Subgraph * sub_graph=context->optimized_graph;

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
	{
		const CPUInfo& cpu_info=soc_info.cpu_list[soc_info.master_cpu_idx];

		std::printf("master cpu: %d run subgraph: %s --  %s\n",cpu_info.cpu_id,
				sub_graph->GetName().c_str(),ret?"OK":"FAIL");
	}

#endif
	return ret;
}

bool CPURunner::Postrun(void * graph_handle)
{
	GraphContext * context=reinterpret_cast<GraphContext *>(graph_handle);

	Subgraph * sub_graph=context->optimized_graph;

	std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;

	for(unsigned int i=0; i<seq_nodes.size();i++)
	{
		Node * node=seq_nodes[i];

		if(!node->ExistAttr(ATTR_NODE_OPS))
			continue;

		for(unsigned int i=0;i<node->GetOutputNum();i++)
		{
			Tensor * tensor=node->GetOutputTensor(i);
			free_tensor_mem(tensor);
		}

		NodeOps * node_ops=any_cast<NodeOps *>(node->GetAttr(ATTR_NODE_OPS));

		if(!node_ops->Postrun(node))
		{
			LOG_ERROR()<<"Postrun failed for node: "<<node->GetName()<<"\n";
		}

		node_ops->Release();
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

			time_stats[op->GetName()]=op_time;

			total_fops+=node->GetFops();
		}

		std::printf("\n==== %s: time stats by operator: ====\n",
                               sub_graph->GetName().c_str());

		std::printf("total time: %lu us, repeat %d\n",total_time,repeat_count);

                std::printf("PER RUN: time %lu us on %.2f Mfops, RATE: %.2f Mfops\n",
                             total_time/repeat_count,total_fops/1000000,
                             total_fops*repeat_count/total_time);

		int n=0;

		for(auto ir=time_stats.begin(); ir!=time_stats.end();ir++)
		{
			std::printf("%d: %s used %lu us (%.2f%%)\n",n++,
					ir->first.c_str(),ir->second, 100.0f*ir->second/total_time);
		}
		std::printf("\n\n");

		prof->Dump(1);

	}
#endif



	return true;
}


bool CPURunner::OptimizeGraph(void * graph_handle)
{
	GraphContext * context=reinterpret_cast<GraphContext *>(graph_handle);

	context->optimized_graph=context->orig_graph;

        GraphOptimizerManager::RunOpt("BNScaleReLu",context->optimized_graph);
        GraphOptimizerManager::RunOpt("ConvReLu",context->optimized_graph);

	return true;
}

bool CPURunner::AllocateMem(Subgraph * sub_graph)
{
	std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;

	for(unsigned int i=0; i<seq_nodes.size();i++)
	{
		Node * node=seq_nodes[i];

		for(unsigned int i=0;i<node->GetOutputNum();i++)
		{

			Tensor * tensor=node->GetOutputTensor(i);

			if(get_tensor_mem(tensor))
				continue;

			int input_idx=-1;

			/* process inplace 
TODO: strict check if the input tensor's consumer is just one 
			 */

			if(node->ExistAttr(ATTR_INPLACE))
			{
				const inplace_t & inplace=any_cast<inplace_t>(node->GetAttr("inplace"));

				if(inplace.count(i))
					input_idx=inplace.at(i);
			}

			if(input_idx>=0)
			{
				Tensor * input_tensor=node->GetInputTensor(input_idx);
				void * tensor_addr=get_tensor_mem(input_tensor);
				int total_size=tensor->GetTotalSize();

				set_tensor_mem(tensor,tensor_addr, total_size,nullptr);
			}
			else
			{
				int total_size=tensor->GetTotalSize();
				void * tensor_addr=mem_alloc(total_size);

				set_tensor_mem(tensor,tensor_addr,total_size,mem_free);
			}
		}
	}

	return true;


}


bool CPURunner::BindNodeOps(Subgraph * sub_graph)
{
	std::vector<Node *>& seq_nodes=sub_graph->seq_nodes;
	int node_size=seq_nodes.size();

	for(int i=0; i<node_size;i++)
	{
		Node * node=seq_nodes[i];
		Operator * op=node->GetOp();

		if(op->GetName()=="Const" || op->GetName()=="Input")
			continue;

		NodeOps * node_ops=NodeOpsRegistryManager::FindNodeOps(&soc_info,node);

		if(node_ops==nullptr)
		{
			LOG_ERROR()<<"failed to set node ops for node: "<<node->GetName()
				<<" op: "<<op->GetName()<<"\n";
			return false;
		}

		node_ops->SetHelper(mem_alloc,mem_free,task_dispatch);

		node->SetAttr(ATTR_NODE_OPS,node_ops);
		node_ops->OnBind(node);
	}


	return true;

}


/** register a few predefined soc */

void  RegisterDefaultSoc(void)
{
	SocInfo soc_info;

	soc_info.cpu_number=6;
	soc_info.soc_name="RK3399";
	soc_info.master_cpu=4;

	CPUInfo cpu_info;

	for(int i=0;i<4;i++)
	{
		cpu_info.cpu_id=i;
		cpu_info.cpu_type="A53";
		cpu_info.cpu_arch="arm64";
		cpu_info.l1_size=32*1024;
		cpu_info.l2_slice=256*1024;

		soc_info.cpu_info.push_back(cpu_info);
		soc_info.cpu_list.push_back(i);
	}

	for(int i=4;i<6;i++)
	{
		cpu_info.cpu_id=i;
		cpu_info.cpu_type="A72";
		cpu_info.cpu_arch="arm64";
		cpu_info.l1_size=32*1024;
		cpu_info.l2_slice=512*1024;

		soc_info.cpu_info.push_back(cpu_info);
		soc_info.cpu_list.push_back(i);
	}

	RegisterPredefinedSoc(soc_info.soc_name,soc_info);

	soc_info.cpu_number=8;
	soc_info.soc_name="HIKEY960";
	soc_info.master_cpu=4;

	for(int i=0;i<4;i++)
	{
		cpu_info.cpu_id=i;
		cpu_info.cpu_type="A53";
		cpu_info.cpu_arch="arm64";
		cpu_info.l1_size=32*1024;
		cpu_info.l2_slice=256*1024;

		soc_info.cpu_info.push_back(cpu_info);
		soc_info.cpu_list.push_back(i);
	}

	for(int i=4;i<8;i++)
	{
		cpu_info.cpu_id=i;
		cpu_info.cpu_type="A73";
		cpu_info.cpu_arch="arm64";
		cpu_info.l1_size=32*1024;
		cpu_info.l2_slice=512*1024;

		soc_info.cpu_info.push_back(cpu_info);
		soc_info.cpu_list.push_back(i);
	}

	RegisterPredefinedSoc(soc_info.soc_name,soc_info);
}



} //namespace TEngine
