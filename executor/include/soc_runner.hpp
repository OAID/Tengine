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
#ifndef __SOC_RUNNER_HPP__
#define __SOC_RUNNER_HPP__

#include <string>
#include <vector>
#include <functional>

#include "node_ops.hpp"

namespace TEngine {

class Graph;
using Subgraph=Graph;

struct CPUInfo {
   std::string cpu_type;
   std::string cpu_arch;
   int cpu_id;
   int l1_size;
   int l2_slice;
};

struct SocInfo {
   int cpu_number;
   int master_cpu;
   std::vector<int> cpu_list;
   std::string soc_name;
   std::vector<CPUInfo> cpu_info;
   int l3_size;

   void SetWorkingCPU(const std::vector<int>& new_cpu_list, int master)
   {
        master_cpu=master_cpu;
        cpu_list=new_cpu_list;
        cpu_number=cpu_list.size();
   }

};

bool GetPredefinedSoc(const std::string& soc_name, SocInfo& soc_info);
bool RegisterPredefinedSoc(const std::string& soc_name, const SocInfo& soc_info);

class SocRunner {
public:
   const std::string& GetSocName() {return soc_info.soc_name;}
   const SocInfo& GetSocInfo(void) {return soc_info;}
   void SetSocInfo(const SocInfo& info) {soc_info=info;}

   virtual bool Init(void)=0;
   virtual void Release(void)=0;

   virtual void * CreateGraphHandle(Subgraph * sub_graph)=0;
   virtual bool OptimizeGraph(void * graph_handle)=0;
   virtual bool Prerun(void * graph_handle)=0;
   virtual bool Run(void * graph_handle)=0;  //always block interface
   virtual bool Postrun(void * graph_handle)=0;
   virtual void ReleaseGraphHandle(void * graph_handle)=0;

   virtual ~SocRunner(){}

protected:
   SocInfo soc_info;

};


class CPURunner: public SocRunner {

public:

   struct GraphContext {
      Subgraph * orig_graph;
      Subgraph * optimized_graph;
   };


   bool Init(void) override {return true;};
   void Release(void) override {};


   void * CreateGraphHandle(Subgraph * sub_graph) override;
   void ReleaseGraphHandle(void * graph_handle) override;
   bool Prerun(void * graph_handle) override;
   bool Run(void * graph_handle) override; 
   bool Postrun(void * graph_handle) override;

   bool SetWorkingCPU(const std::vector<int>& cpu_list, int master);
   void SetHelper(const mem_alloc_t& alloc, const mem_free_t& free,
                 const task_dispatch_t&  dispatch);

 
   virtual bool OptimizeGraph(void * graph_handle) override;
   virtual bool BindNodeOps(Subgraph * graph);
   virtual bool AllocateMem(Subgraph * graph);
 
   virtual ~CPURunner(){} 

   mem_alloc_t mem_alloc;
   mem_free_t mem_free;
   task_dispatch_t task_dispatch;


};







} //namespace TEngine


#endif
