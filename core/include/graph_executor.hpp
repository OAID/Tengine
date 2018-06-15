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
#ifndef __GRAPH_EXECUTOR_HPP__
#define __GRAPH_EXECUTOR_HPP__

#include "static_graph.hpp"
#include "graph.hpp"
#include "exec_engine.hpp"


namespace TEngine {

class RuntimeWorkspace;

using pass_func_t=std::function<bool(Graph *, const any& param)>;

class GraphPassManager: public SimpleObjectManager<GraphPassManager, pass_func_t> {};


class GraphExecutor {

public:

  GraphExecutor() {
       graph_=nullptr;
       graph_attached_=false;
       exec_handle_=nullptr;
	   exec_priority_=100;
  }

  ~GraphExecutor() { 
       if(graph_ && !graph_attached_) 
           ReleaseGraph();
       if(exec_handle_)
           ReleaseExecHandle();
   }

   bool CreateGraph(const std::string& graph_name, const std::string& model_name);

   bool AttachGraph(Graph * graph_);

   Graph * GetGraph(void) { return graph_;}
   Graph * GetOptimizedGraph(void); 

   RuntimeWorkspace * GetWorkspace(void) {  return ws_;}

   void SetWorkspace(RuntimeWorkspace * ws) { ws_=ws;}

   const std::string& GetGraphName(void) { return graph_name_; }
   const std::string& GetModelName(void) { return model_name_; }

   bool SetGraphInputNode(const std::vector<std::string>& node_name);
   bool SetGraphOutputNode(const std::vector<std::string>& node_name);

   int  GetGraphInputNodeNum(void);
   const std::string&  GetGraphInputNodeName(int idx);
   int  GetNodeInputNum(const std::string& node_name);
   const std::string&  GetNodeInputTensor(const std::string& node_name, int idx);

   int  GetGraphOutputNodeNum(void);
   const std::string&  GetGraphOutputNodeName(int idx);
   int  GetNodeOutputNum(const std::string& node_name);
   const std::string&  GetNodeOutputTensor(const std::string& node_name, int idx);

   Tensor * FindTensor(const std::string& name);
   Node * FindNode(const std::string& name);

   bool SetTensorBuffer(Tensor * tensor, void * buffer, int buffer_size);
   void *  GetTensorBuffer(Tensor * tensor);

   bool SetTensorData(Tensor * tensor, const void *  input_data,int data_size);
   bool GetTensorData(Tensor * tensor,void *  output_data,int data_size);

   Tensor * GetInputNodeTensor(unsigned int node_idx, unsigned int tensor_idx);
   Tensor * GetOutputNodeTensor(unsigned int node_idx, unsigned int tensor_idx);

   bool  RunPass(const std::string& pass_name,const any& param);

   bool InferShape(void);

   bool Prerun(void);

   bool Run(int block);
   bool SyncRun(void);

   int  WaitGraph(int try_wait);

   bool Postrun(void);

   void SetExecPolicy(const std::string& p) { exec_policy_=p;}
   void SetExecPriority(int priority)  { exec_priority_=priority;}
   const std::string& GetExecPolicy(void) { return exec_policy_;}
   int   GetExecPriority(void) { return exec_priority_;}

protected:
   void ReleaseGraph(void);
   void ReleaseExecHandle(void);

private:

   std::string graph_name_;
   std::string model_name_;
   std::string exec_policy_;
   int exec_priority_;

   RuntimeWorkspace * ws_;
   Graph * graph_;
   bool   graph_attached_;

   ExecEnginePtr exec_engine_;
   exec_handle_t exec_handle_;
   exec_event_t  exec_event_;

};

} //namespace TEngine

#endif
