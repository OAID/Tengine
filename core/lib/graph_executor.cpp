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

   graph_=graph;
   graph_name_=graph_name;
   model_name_=model_name;

   std::string exec_engine_name="default";

   TEngineConfig::Get("exec.engine",exec_engine_name);

   if(!ExecEngineManager::SafeGet(exec_engine_name,exec_engine_))
   {
      LOG_ERROR()<<"No executor engine registered with name: "<<exec_engine_name<<"\n";
      delete graph;
      return false;
   }

   exec_handle_=exec_engine_->AddGraphExecutor(this);

   if(exec_handle_==nullptr)
   {
       delete graph;
       return false;
   }


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
    Node * node=graph_->FindNode(node_name);

    if(node==nullptr)
          return -1;

    return node->GetInputNum();
}

const std::string&  GraphExecutor::GetNodeInputTensor(const std::string& node_name, int idx)
{
    Node * node=graph_->FindNode(node_name);

    const Tensor * tensor=node->GetInputTensor(idx);

    return tensor->GetName();
}


int GraphExecutor::GetGraphOutputNodeNum(void)
{
    return graph_->output_nodes.size();
}

const std::string&  GraphExecutor::GetGraphOutputNodeName(int idx)
{
    std::vector<Node *>& outputs=graph_->output_nodes;
    Node * node=outputs[idx];

    return node->GetName();
}

int GraphExecutor::GetNodeOutputNum(const std::string& node_name)
{
    Node * node=graph_->FindNode(node_name);

    if(node==nullptr)
          return -1;

    return node->GetOutputNum();
}

const std::string&  GraphExecutor::GetNodeOutputTensor(const std::string& node_name, int idx)
{
    Node * node=graph_->FindNode(node_name);

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

     return true;
}

Tensor * GraphExecutor::FindTensor(const std::string& name)
{
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

          bool skip=false;

          for(unsigned int i=0;i<node->GetInputNum();i++)
          {
             TShape& shape=node->GetInputTensor(i)->GetShape();

             if(shape.GetSize()<=0)
             {
                 skip=true;
                 break;
             }
         }

        if(skip==true)
        {
            std::cout<<"skip node: "<<node->GetName()<<" due to input not ready\n";

             continue;
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
            std::cout<<"Infer Shaped failed"<<std::endl;
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


bool GraphExecutor::Run(int block)
{
    if(!exec_engine_->Run(exec_handle_,exec_event_))
        return false;

    exec_engine_->Wait(exec_handle_,exec_event_);
 
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



} //namespace TEngine
