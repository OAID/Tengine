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
 
#include <vector>
#include <thread>

#include "soc_runner.hpp"
#include "logger.hpp"
#include "graph.hpp"
#include "node_ops.hpp"


namespace TEngine {


namespace demo_ops {

struct DemoOps: public MTNodeOps {
public:

   bool FloatPrerun(Node * node)
   {
      LOG_INFO()<<"float prerun done!\n";
      return true;
   }

   bool FloatPostrun(Node * node)
   {
      LOG_INFO()<<"float post run done!\n";
      return true;
   }

   bool FloatRun(Node * node)
   {
      LOG_INFO()<<"float run done!\n";
      return true;
   }

   bool IntPrerun(Node * node)
   {
      LOG_INFO()<<"int prerun done!\n";
      return true;
   }

   bool IntPostrun(Node * node)
   {
      LOG_INFO()<<"int post run done!\n";
      return true;
   }

   bool IntRun(Node * node)
   {
      LOG_INFO()<<"int run done!\n";
      return true;
   }

   bool MTIntRun(Node * node)
   {
        std::vector<sub_op_task> task_list;

        for(unsigned int i=0;i<cpu_map.size()*2;i++)
        {
           sub_op_task task;
           task.exec_func=std::move(std::bind(&DemoOps::IntAider,this,std::placeholders::_1,
                                    std::placeholders::_2,std::placeholders::_3));
           task.seq=i;
           task.data=(void *)((unsigned long)i);

           task_list.push_back(task);
        }

        IncRequest(task_list.size());

        task_dispatch(task_list,-1);

        WaitDone();     

        return true;
   }


   bool MTFloatRun(Node * node)
   {
        std::vector<sub_op_task> task_list;

        for(unsigned int i=0;i<cpu_map.size()*2;i++)
        {
           sub_op_task task;
           task.exec_func=std::bind(&DemoOps::FloatAider,this,std::placeholders::_1,
                                    std::placeholders::_2,std::placeholders::_3);
           task.seq=i;
           task.data=(void *)((unsigned long)i);

           task_list.push_back(task);
        }

        IncRequest(task_list.size());

        task_dispatch(task_list,-1);

        WaitDone();     

        return true;
   }
   

   bool IntAider(int cpu, int seq, void * data)
   {
      const std::string& cpu_type=cpu_map[cpu];

      if(cpu_type=="A53")
           A53IntAider(cpu, seq,data);
      else
           A72IntAider(cpu, seq,data);     

      IncDone(1);
      return true;
   }

   bool FloatAider(int cpu, int seq, void * data)
   {
      const std::string& cpu_type=cpu_map[cpu];

      if(cpu_type=="A53")
           A53FloatAider(cpu, seq,data);
      else
           A72FloatAider(cpu, seq,data);     

      IncDone(1);
      return true;
   }

   bool A72FloatAider(int cpu, int seq, void * data)
   {
       unsigned long n=(unsigned long)(data);

       LOG_INFO()<<"cpu: "<<cpu<<" A72 FLOAT called\n";
       LOG_INFO()<<"cpu: "<<cpu<<" will sleep "<<n<<" seconds\n";

       std::chrono::milliseconds sleep_time(n*1000);
       std::this_thread::sleep_for(sleep_time);

       LOG_INFO()<<"cpu: "<<cpu<<" DONE\n";

       return true;
   }

   bool A53FloatAider(int cpu, int seq, void * data)
   {
       unsigned long n=(unsigned long)(data);

       LOG_INFO()<<"cpu: "<<cpu<<" A53 FLOAT called\n";
       LOG_INFO()<<"cpu: "<<cpu<<" will sleep "<<n<<" seconds\n";
 
       std::chrono::milliseconds sleep_time(n*1000);
       std::this_thread::sleep_for(sleep_time);

       LOG_INFO()<<"cpu: "<<cpu<<" DONE\n";

       return true;
   }

      bool A72IntAider(int cpu, int seq, void * data)
   {
       unsigned long n=(unsigned long)(data);

       LOG_INFO()<<"cpu: "<<cpu<<" A72 INT called\n";
       LOG_INFO()<<"cpu: "<<cpu<<" will sleep "<<n<<" seconds\n";

       std::chrono::milliseconds sleep_time(n*1000);
       std::this_thread::sleep_for(sleep_time);

       LOG_INFO()<<"cpu: "<<cpu<<" DONE\n";

       return true;
   }

   bool A53IntAider(int cpu, int seq, void * data)
   {
       unsigned long n=(unsigned long)(data);

       LOG_INFO()<<"cpu: "<<cpu<<" A53 INT called\n";
       LOG_INFO()<<"cpu: "<<cpu<<" will sleep "<<n<<" seconds\n";
 
       std::chrono::milliseconds sleep_time(n*1000);
       std::this_thread::sleep_for(sleep_time);

       LOG_INFO()<<"cpu: "<<cpu<<" DONE\n";

       return true;
   }
   
  /*****************************************************/
   bool Prerun (Node * node) override
   {
       if(float_mode)
	   	return FloatPrerun(node);
	else
		return IntPrerun(node);
   }

   bool Run (Node * node) override
   {

        const std::string& master_type=cpu_map[master_cpu];

        std::cout<<"Run launched on : "<<master_type<<"\n";


   	if(float_mode)
   	{
   	     if(mt_mode)
		 	return MTFloatRun(node);
	     else
		 	return FloatRun(node);
   	}
	else
	{  
	     if(mt_mode)
		 	return MTIntRun(node);
	     else
		 	return IntRun(node);
		
	}
   }

    bool Postrun (Node * node) override
   {
       if(float_mode)
	   	return FloatPostrun(node);
	else
		return IntPostrun(node);
   }

   DemoOps() {float_mode=true; mt_mode=false;}
 
   int master_cpu;  
   std::vector<std::string> cpu_map;
   bool float_mode;
   bool mt_mode;
   
};



struct DemoSelector: public NodeOpsSelector {

DemoSelector(void) { op_name="DemoOp"; }

NodeOps * Select(SocInfo * info, Node * node)
{
     DemoOps * ops=new DemoOps();


           if(info->cpu_list.size()>1)
              ops->mt_mode=true;
           else
              ops->mt_mode=false;

     Tensor * input_tensor=node->GetInputTensor(0);
     
     if(input_tensor->GetDatatype()=="float32")
     {
           ops->float_mode=true;
     }
     else
     {
           ops->float_mode=false;
     }

    ops->cpu_map.resize(info->cpu_info.size());

    for(unsigned int i=0;i<info->cpu_list.size();i++)
    {
        int cpu_id=info->cpu_list[i];

        const CPUInfo& cpu=info->cpu_info[cpu_id];

        ops->cpu_map[cpu_id]=cpu.cpu_type;
    }

    ops->master_cpu=info->master_cpu;

    return ops;
}
};


} //namespace demo_ops

using namespace demo_ops;

void RegisterDemoOps(void)
{
   //register selector 
   DemoSelector * demo_selector=new  DemoSelector();
   NodeOpsRegistryManager::RegisterOPSelector(REF_REGISTRY_NAME,demo_selector);

}


} //namespace TEngine
