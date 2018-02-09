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
#include <iostream>
#include <functional>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "share_lib_parser.hpp"
#include "operator/demo_op.hpp"
#include "node_ops.hpp"
#include "test_soc_info.hpp"
#include "graph.hpp"

using namespace TEngine;

static std::mutex queue_lock;
static std::condition_variable queue_cv;
static std::queue<sub_op_task> task_queue;


bool task_dispatch(std::vector<sub_op_task>& tasks, int cpu)
{
	std::unique_lock<std::mutex> cv_lock(queue_lock,std::defer_lock);
	cv_lock.lock();

	for(unsigned int i=0;i<tasks.size();i++)
	{
		const sub_op_task& task=tasks[i];

		task_queue.push(task);
	}

	cv_lock.unlock();

	queue_cv.notify_all();

        return true;
}


void aider_thread(int cpu)
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpu,&mask);


	if(sched_setaffinity(0,sizeof(mask),&mask)<0)
	{
		std::cout<<"worker: failed to bind cpu: "<<cpu<<"\n";
	}

        std::cout<<"worker ready on cpu: "<<cpu<<"\n";

	while(true)
	{

		sub_op_task task;

		std::unique_lock<std::mutex> cv_lock(queue_lock);

                if(task_queue.empty())
		     queue_cv.wait(cv_lock,[]{return !task_queue.empty();});

		task=task_queue.front();
		task_queue.pop();

		cv_lock.unlock();

		task.exec_func(cpu,task.seq,task.data);     
	}


}

std::vector<std::thread *> thread_list;

void launch_aider_threads(SocInfo * soc_info)
{
    thread_list.resize(soc_info->cpu_list.size());

    for(unsigned int i=0;i<soc_info->cpu_list.size();i++)
    {
        int cpu_id=soc_info->cpu_list[i];

        thread_list[i]=new std::thread(aider_thread,cpu_id);
    }
}

Node * create_demo_op_node(bool is_float)
{
	Operator * op=OpManager::CreateOp("DemoOp");
	DemoOp * demo_op=dynamic_cast<DemoOp *>(op);

	Node * node=new Node("test_demo_op");

	node->SetOp(demo_op);

	Tensor * tensor=new Tensor("input");

	if(is_float)
		tensor->SetDataType("float32");
	else
		tensor->SetDataType("int8");

	tensor->SetType(kVarTensor);

	node->SetInputPort(0,tensor);

	tensor=new Tensor("output");

	if(is_float)
		tensor->SetDataType("float32");
	else
		tensor->SetDataType("int8");

	tensor->SetType(kVarTensor);

	node->SetOutputPort(0,tensor);

	return node;
}


void test_mt_mode(void)
{
	Node * float_node=create_demo_op_node(true);
	Node * int_node=create_demo_op_node(false);

	SocInfo * soc_info=TestGetSocInfo();
	NodeOps * float_ops=NodeOpsRegistryManager::FindNodeOps(soc_info,float_node);

	float_ops->SetHelper(std::malloc,std::free,task_dispatch);

	if(!float_ops->Prerun(float_node))
	{
		std::cout<<"Prerun failed\n";
	} 

	if(!float_ops->Run(float_node))
	{
		std::cout<<"Run failed\n";
	}


	if(!float_ops->Postrun(float_node))
	{
		std::cout<<"Postrun failed\n";
	} 

	std::cout<<"FLOAT TEST DONE\n";


       NodeOps * int_ops=NodeOpsRegistryManager::FindNodeOps(soc_info,int_node);

	int_ops->SetHelper(std::malloc,std::free,task_dispatch);

	if(!int_ops->Prerun(int_node))
	{
		std::cout<<"Prerun failed\n";
	} 

	if(!int_ops->Run(int_node))
	{
		std::cout<<"Run failed\n";
	}


	if(!int_ops->Postrun(int_node))
	{
		std::cout<<"Postrun failed\n";
	} 

	std::cout<<"INT TEST DONE\n";
 
	delete float_node;
        delete int_node;	

}

void test_st_mode(void)
{
	Node * float_node=create_demo_op_node(true);

        SocInfo  soc_info;

        soc_info.cpu_number=1;
        soc_info.master_cpu=0;
        soc_info.soc_name="SingCPU";

        CPUInfo cpu_info;
        cpu_info.cpu_type="A72";
        cpu_info.cpu_arch="arm64";
        cpu_info.cpu_id=4;

        soc_info.cpu_info.push_back(cpu_info);
        soc_info.cpu_list.push_back(0);

        NodeOps * float_ops=NodeOpsRegistryManager::FindNodeOps(&soc_info,float_node);

        float_ops->SetHelper(std::malloc,std::free,task_dispatch);

        if(!float_ops->Prerun(float_node))
        {
                std::cout<<"Prerun failed\n";
        }

        if(!float_ops->Run(float_node))
        {
                std::cout<<"Run failed\n";
        }


        if(!float_ops->Postrun(float_node))
        {
                std::cout<<"Postrun failed\n";
        }

        std::cout<<"FLOAT TEST DONE\n";

	delete float_node;

}


void sys_init(void)
{
	ShareLibParser p0("./build/operator/liboperator.so");
	p0.ExcecuteFunc<int()>("tengine_plugin_init");

	ShareLibParser p1("./build/serializer/libserializer.so");
	p1.ExcecuteFunc<int()>("tengine_plugin_init");

	ShareLibParser p2("./build/executor/libexecutor.so");
	p2.ExcecuteFunc<int()>("tengine_plugin_init");
}


int main(int argc, char * argv[])
{

	sys_init();

        launch_aider_threads(TestGetSocInfo());

        sleep(1);

	test_mt_mode();
        test_st_mode();

	return 0;

}


