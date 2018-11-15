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
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>

#include "graph_executor.hpp"
#include "tengine_config.hpp"


namespace TEngine {


bool GraphExecutor::CreateGraph(const std::string& graph_name, const std::string& model_name)
{

   StaticGraphPtr static_graph;

  if(!StaticGraphManager::SafeGet(model_name,static_graph))
       return false;

   Graph * graph=Graph::CreateFromStatic(static_graph);

   if(graph==nullptr)
       return false;

   graph_name_=graph_name;
   model_name_=model_name;

   std::string exec_engine_name;

   TEngineConfig::Get("exec.engine",exec_engine_name);

   if(!ExecEngineManager::SafeGet(exec_engine_name,exec_engine_))
   {
      LOG_ERROR()<<"No executor engine registered with name: "<<exec_engine_name<<"\n";
      delete graph;
      return false;
   }

   graph_=graph;

   exec_handle_=exec_engine_->AddGraphExecutor(this);

   if(exec_handle_==nullptr)
   {
       delete graph;
       return false;
   }

   exec_attr_.model_format=graph->GetModelFormat();

   //check kernel mode variable
   const char * mode=std::getenv("KERNEL_MODE");

   if(mode)
   {
	   int kernel_mode=strtoul(mode,NULL,10);

	   LOG_INFO()<<"ENV Set kernel mode: ["<<kernel_mode<<"]\n";

	   exec_attr_.kernel_mode=kernel_mode;
   }
   
   //check low_mem_mode env var

   const char * mem=std::getenv("LOW_MEM_MODE");

   if(mem)
   {
       if(mem[0]=='0')
           exec_attr_.low_mem_mode=false;
       else
           exec_attr_.low_mem_mode=true;

   }
   
    
   return true; 
  
}

bool GraphExecutor::AttachGraph(Graph * graph)
{
    graph_name_=graph->GetName();
    model_name_="none model";

    std::string exec_engine_name;

    TEngineConfig::Get("exec.engine",exec_engine_name);

    if(!ExecEngineManager::SafeGet(exec_engine_name,exec_engine_))
    {
       LOG_ERROR()<<"No executor engine registered with name: "<<exec_engine_name<<"\n";
       return false;
    }

    graph_=graph;
    graph_attached_=true;

    exec_handle_=exec_engine_->AddGraphExecutor(this);

    if(exec_handle_==nullptr) return false;

    return true;
}

int GraphExecutor::GetGraphInputNodeNum(void)
{
    return graph_->input_nodes.size();
}

const std::string&  GraphExecutor::GetGraphInputNodeName(int idx)
{
    std::vector<Node *>& inputs=graph_->input_nodes;
    Node * node=inputs[idx];
     
    return node->GetName(); 
}

int GraphExecutor::GetNodeInputNum(const std::string& node_name)
{
    Node * node=FindNode(node_name);

    if(node==nullptr)
          return -1;

    return node->GetInputNum();
}

const std::string&  GraphExecutor::GetNodeInputTensor(const std::string& node_name, int idx)
{
    Node * node=FindNode(node_name);

    const Tensor * tensor=node->GetInputTensor(idx);

    return tensor->GetName();
}


int GraphExecutor::GetGraphOutputNodeNum(void)
{
    Graph * optimized_graph=GetOptimizedGraph();

	if(optimized_graph)
         return optimized_graph->output_nodes.size();

    return graph_->output_nodes.size();
}

const std::string&  GraphExecutor::GetGraphOutputNodeName(int idx)
{
    Graph * optimized_graph=GetOptimizedGraph();

	Graph * cur_graph;

	if(optimized_graph)
		 cur_graph=optimized_graph;
	else
		 cur_graph=graph_;


    std::vector<Node *>& outputs=cur_graph->output_nodes;
    Node * node=outputs[idx];

    return node->GetName();
}

int GraphExecutor::GetNodeOutputNum(const std::string& node_name)
{
    Node * node=FindNode(node_name);

    if(node==nullptr)
          return -1;

    return node->GetOutputNum();
}

const std::string&  GraphExecutor::GetNodeOutputTensor(const std::string& node_name, int idx)
{
    Node * node=FindNode(node_name);

    const Tensor * tensor=node->GetOutputTensor(idx);

    return tensor->GetName();
}


bool GraphExecutor::SetGraphInputNode(const std::vector<std::string>& node_name_list)
{
      graph_->ResetInputNode();

     for(unsigned int i=0;i<node_name_list.size();i++)
     {
         if(!graph_->AddInputNode(node_name_list[i]))
            return false;
     }

     return true;
}

bool GraphExecutor::SetGraphOutputNode(const std::vector<std::string>& node_name_list)
{
     graph_->ResetOutputNode();

     for(unsigned int i=0;i<node_name_list.size();i++)
     {
         if(!graph_->AddOutputNode(node_name_list[i]))
            return false;
     }

     graph_->StripGraph();

     return true;
}

Node  * GraphExecutor::FindNode(const std::string& name)
{
     Graph * optimized_graph=GetOptimizedGraph();

     if(optimized_graph)
     {
          Node * node=optimized_graph->FindNode(name);
          if(node)
            return node;
     }

     return graph_->FindNode(name);
}

Tensor * GraphExecutor::FindTensor(const std::string& name)
{
     //try to search in optmized graph first

     Graph * optimized_graph=GetOptimizedGraph();

     if(optimized_graph)
     {
         Tensor * tensor;

         tensor=optimized_graph->FindTensor(name);
         
         if(tensor) 
              return tensor;    
     }

     return graph_->FindTensor(name);
}

bool GraphExecutor::InferShape(void)
{
     int node_number=graph_->seq_nodes.size();
     Node * node;

     for(int i=0;i<node_number;i++)
     {
         node=graph_->seq_nodes[i];

         Operator * op=node->GetOp();

         // std::cout<<"Process Node: "<<node->GetName()<<" Op: "<<op->GetName()<<std::endl;

         if(op->GetName()=="Const" || op->GetName()=="Input")
                 continue;

         if(node->IsDynamicShape())
               continue;

          bool skip=false;
          unsigned int j;

          for(j=0;j<node->GetInputNum();j++)
          {
			 Tensor * input=node->GetInputTensor(j);
             TShape& shape=input->GetShape();

             if(shape.GetSize()==0)
             {
                 XLOG_ERROR()<<"infer shape failed on node: "<<node->GetName()
                        <<" due to input: "<<input->GetName()
                        <<" size is zero\n";
                 return false;
             }

             if(shape.GetSize()<0)
             {
                 skip=true;
                 break;
             }

			 if(input->Reshaped())
				  input->UpdateReshapeCount();
         }

        if(skip==true)
        {
             XLOG_ERROR()<<"infer shape failed on node: "<<node->GetName()
                           <<" due to input: "<<node->GetInputTensor(j)->GetName()
                           <<" not ready\n";
             return false;
        }

       std::vector<TShape> inputs;

       for(unsigned int i=0;i<node->GetInputNum();i++)
       {
              Tensor * tensor=node->GetInputTensor(i);

              inputs.push_back(tensor->GetShape());
       }

       std::vector<TShape> outputs;

       outputs.resize(node->GetOutputNum());

       if(!op->InferShape(inputs,outputs))
       {
            std::cout<<"infer shaped for node: "<<node->GetName()
                     <<" op: "<<op->GetName()<<" failed\n"; 
            return false;
       }

       for(unsigned int i=0;i<node->GetOutputNum();i++)
       {
             Tensor * tensor=node->GetOutputTensor(i);
             TShape& shape=tensor->GetShape();
             TShape& new_shape=outputs[i];

             if(new_shape.GetSize())
                  shape=new_shape;
       }

   }

   return true;

}

bool  GraphExecutor::RunPass(const std::string& pass_name,const any& param)
{
    return true;
}

Tensor * GraphExecutor::GetInputNodeTensor(unsigned int node_idx, unsigned int tensor_idx)
{
     if(node_idx>= graph_->input_nodes.size())
          return nullptr;

     Node *  node=graph_->input_nodes[node_idx];

     if(tensor_idx>=node->GetOutputNum())
          return nullptr;

     return node->GetOutputTensor(tensor_idx);
    
}

Tensor * GraphExecutor::GetOutputNodeTensor(unsigned int node_idx,unsigned int tensor_idx)
{
     if(node_idx>= graph_->output_nodes.size())
          return nullptr;

     Node *  node=graph_->output_nodes[node_idx];

     if(tensor_idx>=node->GetOutputNum())
          return nullptr;

     return node->GetOutputTensor(tensor_idx);
}


bool GraphExecutor::SetTensorBuffer(Tensor * tensor, void * input_data, int data_size)
{

     return exec_engine_->SetTensorBuffer(tensor,input_data,data_size,exec_handle_);


}

void * GraphExecutor::GetTensorBuffer(Tensor * tensor)
{
      return exec_engine_->GetTensorBuffer(tensor,exec_handle_);
}


bool GraphExecutor::SetTensorData(Tensor * tensor, const void *  input_data,int data_size)
{
     int tensor_size=tensor->GetTotalSize();

     if(tensor_size!=data_size)
         return false;

  
     void * tensor_addr=GetTensorBuffer(tensor);

     if(tensor_addr==nullptr)
         return false;

     std::memcpy(tensor_addr,input_data,data_size);

     return true;
}

bool GraphExecutor::GetTensorData(Tensor * tensor,void *  output_data,int data_size)
{
     int tensor_size=tensor->GetTotalSize();

     if(tensor_size!=data_size)
         return false;


     void * tensor_addr=GetTensorBuffer(tensor);

     if(tensor_addr==nullptr)
         return false;

     std::memcpy(output_data,tensor_addr,data_size);

     return true;

}

bool GraphExecutor::Prerun(void)
{
   return exec_engine_->Prerun(exec_handle_);
}

bool GraphExecutor::Postrun(void)
{
   return exec_engine_->Postrun(exec_handle_);
}


int  GraphExecutor::WaitGraph(int try_wait)
{
    return exec_engine_->Wait(exec_handle_,exec_event_, try_wait);
}

bool GraphExecutor::SyncRun(void)
{
    return exec_engine_->SyncRun(exec_handle_);
}

bool GraphExecutor::Run(int block)
{

  if(!exec_engine_->Run(exec_handle_,exec_event_))
        return false;
 
   if(block)
       exec_engine_->Wait(exec_handle_,exec_event_,0);

    return true;
}

void GraphExecutor::ReleaseGraph(void)
{
    delete graph_;
}

void GraphExecutor::ReleaseExecHandle(void)
{
    exec_engine_->RemoveGraphExecutor(exec_handle_);
}

Graph *  GraphExecutor::GetOptimizedGraph(void)
{
     if(exec_handle_==nullptr || exec_engine_ == nullptr)
          return nullptr;

      return exec_engine_->GetOptimizedGraph(exec_handle_);
}

bool GraphExecutor::SetExecAttrEntry(const char * name, const void *val, int size)
{
      if(!strcmp("exec_policy",name))
	  {
	       exec_attr_.policy=(exec_policy_t)(*(int*)val);
	  }else if(!strcmp("exec_priority",name))
	  {
		   exec_attr_.priority=*(int *)val;
	  }
	  else if(!strcmp("kernel_mode",name))
	  {
		  exec_attr_.kernel_mode=*(int *)val;
	  }
	  else if(!strcmp("low_mem_mode",name))
	  {
		   int n=*(int *)val;

		   if(n)
			    exec_attr_.low_mem_mode=true;
		   else
			   exec_attr_.low_mem_mode=false;
	  }
	  else if(!strcmp("fc_mt",name))
	  {
		  int n=*(int *)val;
		  if(n)
			  exec_attr_.fc_mt=true;
		  else
			  exec_attr_.fc_mt=false;
	  }
	  else if(!strcmp("pooling_mt",name))
	  {
		  int n=*(int *)val;
		  if(n)
			  exec_attr_.pooling_mt=true;
		  else
			  exec_attr_.pooling_mt=false;
	  }
	  else
      {
		  return false;
	  }

	  return true;
}

bool GraphExecutor::GetExecAttrEntry(const char * name, void * val, int size)
{
      if(!strcmp("exec_policy",name))
	  {
	       *(int *)val=exec_attr_.policy;
	  }else if(!strcmp("exec_priority",name))
	  {
		   *(int *)val=exec_attr_.priority;
	  }
	  else if(!strcmp("kernel_mode",name))
	  {
		  *(int *)val=exec_attr_.kernel_mode;
	  }
	  else if(!strcmp("low_mem_mode",name))
	  {
		   if(exec_attr_.low_mem_mode)
			   *(int *)val=1;
		   else
			   *(int *)val=0;
	  }
	  else if(!strcmp("fc_mt",name))
	  {
		   if(exec_attr_.fc_mt)
			   *(int *)val=1;
		   else
			   *(int *)val=0;
	  }
	  else
	  {
		  return false;
	  }

	  return true;
}

bool GraphExecutor::BailoutSetAttr(const char * name, const void * val, int size)
{
	return exec_engine_->SetGraphAttr(exec_handle_,name,val,size); 
}

bool GraphExecutor::BailoutGetAttr(const char * name, void * val, int size)
{
	return exec_engine_->GetGraphAttr(exec_handle_,name,val,size);
}


void GraphExecutor::InitAttrIO(void)
{
	auto get_func=std::bind(&GraphExecutor::GetExecAttrEntry,this,
			                 std::placeholders::_1,
			                 std::placeholders::_2,
							 std::placeholders::_3);


	attr_io_.RegGetFunc("exec_policy",get_func);
	attr_io_.RegGetFunc("exec_priority",get_func);
	attr_io_.RegGetFunc("kernel_mode",get_func);
	attr_io_.RegGetFunc("low_mem_mode",get_func);
	attr_io_.RegGetFunc("fc_mt",get_func);
	attr_io_.RegGetFunc("pooling_mt",get_func);

	auto set_func=std::bind(&GraphExecutor::SetExecAttrEntry,this,
			                  std::placeholders::_1,
			                  std::placeholders::_2,
							  std::placeholders::_3);

	attr_io_.RegSetFunc("exec_policy",set_func);
	attr_io_.RegSetFunc("exec_priority",set_func);
	attr_io_.RegSetFunc("kernel_mode",set_func);
	attr_io_.RegSetFunc("low_mem_mode",set_func);
	attr_io_.RegSetFunc("fc_mt",set_func);
	attr_io_.RegSetFunc("pooling_mt",set_func);

	//bailout
	auto set_func2=std::bind(&GraphExecutor::BailoutSetAttr,this,std::placeholders::_1,
			                      std::placeholders::_2,std::placeholders::_3);

	auto get_func2=std::bind(&GraphExecutor::BailoutGetAttr,this,std::placeholders::_1,
			                      std::placeholders::_2,std::placeholders::_3);

	attr_io_.RegSetFunc(nullptr,set_func2);


	attr_io_.RegGetFunc(nullptr,get_func2);

}





} //namespace TEngine
