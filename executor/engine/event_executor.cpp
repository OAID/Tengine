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
#include <string>
#include <unordered_map>

#include "graph.hpp"
#include "graph_executor.hpp"
#include "tengine_manager.hpp"
#include "event_executor.hpp"

namespace TEngine {


static std::thread * engine_thread;

static executor_t * get_executor(void)
{
        static executor_t executor;
	return &executor;
}

void  tengine_init_event_executor(void)
{
     //kick off the running thread
    engine_thread=new std::thread(TEngineManager::Run,get_executor(),false);

    //register the engine

    EventExecutor * p_engine= new EventExecutor();

    ExecEngineManager::SafeAdd("event",ExecEnginePtr(p_engine));
}

static void link_segment(nn_t * nn)
{
	int count = nn->gfx.size();
	for(int fl = 0; fl < count; ++fl){
		//link sg1.out with sg2.in 
		subgraph_desc_t sg1 = nn->gfx[fl];
		int sg1_c = sg1->size();
		int out1_n = (*sg1)[sg1_c-1]->node->GetOutputNum();
		for(int out1_i = 0; out1_i < out1_n; ++out1_i){
			Tensor* ot = (*sg1)[sg1_c-1]->node->GetOutputTensor(out1_i);
			if(ot == nullptr) continue;
			for(int sl = fl; sl < count; ++sl){
				subgraph_desc_t sg2 = nn->gfx[sl];
				int in2_n =  (*sg2)[0]->node->GetInputNum();
				for(int in2_i = 0; in2_i < in2_n; ++in2_i){
					Tensor* it = (*sg2)[0]->node->GetInputTensor(in2_i);
					if(it == nullptr) continue;
					if(it == ot){
						Pipe * pp = new Pipe((*sg1)[sg1_c-1]->out[out1_i], (*sg2)[0]->in[in2_i]);
						size_t pp_h = nn->cntxt.pp_ctx.GetHandle(pp);
						std::cout << "1: pipe handle <" << pp_h << ">";                
						std::printf("<%p=>%p>  T<%p, %p>\n", (*sg1)[sg1_c-1]->node, (*sg2)[0]->node, it, ot);            
						nn->gfx.connections.push_back(pp);
					}                       
				}
			}
		}                 
		//link sg1.in with sg2.out 
		int in1_n =  (*sg1)[0]->node->GetInputNum();
		for(int in1_i = 0; in1_i < in1_n; ++in1_i){
			Tensor* it = (*sg1)[0]->node->GetInputTensor(in1_i);
			if(it == nullptr) continue;
			for(int sl = fl; sl < count; ++sl){
				subgraph_desc_t sg2 = nn->gfx[sl];
				int sg2_c = sg2->size();
				int out2_n = (*sg2)[sg2_c-1]->node->GetOutputNum();                    
				for(int out2_i = 0; out2_i < out2_n; ++out2_i){
					Tensor* ot = (*sg2)[sg2_c-1]->node->GetOutputTensor(out2_i);
					if(ot == nullptr) continue;
					if(it == ot){
						Pipe * pp = new Pipe((*sg2)[sg2_c-1]->out[out2_i], (*sg1)[0]->in[in1_i]);
						size_t pp_h = nn->cntxt.pp_ctx.GetHandle(pp);
						std::cout << "2: pipe handle <" << pp_h << ">";  
						std::printf("<%p=>%p> T<%p, %p>\n", (*sg2)[sg2_c-1]->node, (*sg1)[0]->node, it, ot);              
						nn->gfx.connections.push_back(pp);
					}                       
				}
			}
		}                 
	}

}

static nn_t *  tengine_create_nn(Graph * graph)
{
	Configure dev_config = 
	{
		{"device", 1 /*TEngine::DEVICE::DLA*/} //TEngine::DEVICE::AUTO
	};

	executor_t * executor=get_executor();

	nn_t * nn=executor->NewNN(std::move(dev_config));

	nn->graph=graph;

	std::vector<Node *>& seq_nodes = nn->graph->seq_nodes;

	for(unsigned int i=0; i<seq_nodes.size();i++)
	{
		Node* node=seq_nodes[i];
		nn->AppendNode(node);
	}

	std::printf("------------------------------------------------------------\n"); 

	int seg_index = 0;

	for (auto a : nn->gfx){
		//debug ==>
		int sg1_c = a->size();
		Tensor* it = (*a)[0]->node->GetInputTensor(0);
		Tensor* ot = (*a)[sg1_c-1]->node->GetOutputTensor(0);
		std::printf("Segmen<%d, %p, %p> : ", seg_index, it, ot);
		//debug <==
		for(auto b : *a){
			Node* node = b->node;
			Operator * op = node->GetOp();
			const std::string& name = op->GetName();                        
			std::printf("Node<%p>(%s) =>", node, name.c_str());			
		}
		seg_index++;
		std::printf("\n");
	}

	link_segment(nn);

	return nn;
}



#define TENSOR_BUF_KEY "outer_buffer"

exec_handle_t EventExecutor::AddGraphExecutor(GraphExecutor *graph_executor)
{
    Graph * graph=graph_executor->GetGraph();

    nn_t *  nn=tengine_create_nn(graph);

    if(nn==nullptr)
           return nullptr;

     exec_env * h=new exec_env();

     h->graph_executor=graph_executor;
     h->nn=nn;
     h->status=EXEC_STATUS_CREATED;
     h->executor=get_executor();

     any * ret=new any();

     (*ret)=h;

     return ret;
}

void * EventExecutor::GetTensorBuffer(Tensor * tensor, exec_handle_t h) 
{
     if(tensor->GetType()==kConstTensor)
           return tensor->GetMemAddr();

     if(tensor->ExistAttr(TENSOR_BUF_KEY))
           return any_cast<void *>(tensor->GetAttr(TENSOR_BUF_KEY));

      return nullptr;
}

bool EventExecutor::SetTensorBuffer(Tensor * tensor, void *buffer, int buffer_size, exec_handle_t h) 
{
     Node * node=tensor->producer->owner;
     const TShape&  shape=tensor->GetShape();
     std::vector<int> dim=shape.GetDim();

     exec_env * env=any_cast<exec_env *>(*h);

     GraphExecutor * graph_executor=env->graph_executor;
     Graph * graph=graph_executor->GetGraph();

     int input_tensor=0;

     for(unsigned int i=0;i<graph->input_nodes.size();i++)
     {
          Node * input_node=graph->input_nodes[i];
          if(input_node==node)
           {
              input_tensor=1;
              break;
           }
     }

     if(input_tensor)
     {
        (*tensor)[TENSOR_BUF_KEY]=buffer;

        TEngine::MemoryBlockHal *memblock = new TEngine::MemoryBlockHal();

        memblock->Init(buffer,buffer_size);
        memblock->SetDataType(DT_FLOAT32,dim);

        nn_t * nn=env->nn;

        env->executor->SetInput(nn,node->GetName().c_str(),memblock);

        env->indata_list.push_back(memblock);

        return true;
     }

     int output_tensor=0;
 
     for(unsigned int i=0;i<graph->output_nodes.size();i++)
     {
          Node * output_node=graph->output_nodes[i];
          if(output_node==node)
           {
              output_tensor=1;
              break;
           }
     }

     if(output_tensor)
     {
        (*tensor)[TENSOR_BUF_KEY]=buffer;

        TEngine::MemoryBlockHal *memblock = new TEngine::MemoryBlockHal();

        memblock->Init(buffer,buffer_size);
        memblock->SetDataType(DT_FLOAT32,dim);

        nn_t * nn=env->nn;

        env->executor->SetOutput(nn,node->GetName().c_str(),memblock);

        env->outdata_list.push_back(memblock);

        return true;
     }

     return false;
}

bool  EventExecutor::Prerun(exec_handle_t h)
{
     exec_env * env=any_cast<exec_env *>(*h);

     env->executor->ScheduleNN(env->nn);

     return true;
}


bool EventExecutor::Run(exec_handle_t h,exec_event_t& event)
{
     exec_env * env=any_cast<exec_env *>(*h);

     nn_t * nn=env->nn;

    for(unsigned int i=0;i<env->indata_list.size();i++)
    {
         auto indata=env->indata_list[i];
         indata->SetReady();
    }

    nn->Start();

    nn->Wait(1000000);

         
    for(unsigned int i=0;i<env->outdata_list.size();i++)
    {
         auto outdata=env->outdata_list[i];
         outdata->ResetReady();
    }

    return true;
}

bool EventExecutor::Postrun(exec_handle_t h) 
{
     return true; 
}

exec_status_t EventExecutor::GetStatus(exec_handle_t h) 
{
     return exec_status_t();
}

const std::string& EventExecutor::GetStatusStr(const exec_status_t&)
{
    static std::string msg="Not Implemented\n";

    return msg;
}

int EventExecutor::GetStatusCode(const exec_status_t& s) 
{
    return 0;
}

std::string  EventExecutor::GetErrorStr(exec_handle_t h) 
{
   return "No Error. Not Implemented yet\n";
}

bool EventExecutor::RemoveGraphExecutor(exec_handle_t h)
{
     exec_env * env=any_cast<exec_env *>(*h);
     //delete env->nn;  //TODO: HOW TO REMOVE THE NN?
     delete env;
     delete h;
     return true; 
}

} //namespace TEngine
