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
#include "cpu_driver.hpp"
#include "tensor_mem.hpp"

namespace TEngine {

void * CPUCore::CreateGraphHandle(Subgraph * sub_graph)
{
     CPURunner * runner=dev->GetRunner();

     return runner->CreateGraphHandle(sub_graph);
}


void CPUCore::ReleaseGraphHandle(void * graph_handle)
{
     CPURunner * runner=dev->GetRunner();

     runner->ReleaseGraphHandle(graph_handle);
}

bool CPUCore::OptimizeGraph(void * graph_handle)
{
     CPURunner * runner=dev->GetRunner();

     return runner->OptimizeGraph(graph_handle);
}


bool CPUCore::Postrun(void * graph_handle)
{
     CPURunner * runner=dev->GetRunner();

     return runner->Postrun(graph_handle);
}


bool CPUCore::Prerun(void * graph_handle)
{
     CPURunner * runner=dev->GetRunner();

     return runner->Prerun(graph_handle);
}


bool CPUCore::RunGraph(void * graph_handle, Subgraph * sub_graph, dev_graph_cb_t graph_cb)
{
       	bool ret=RealRunGraph(graph_handle);

	if(graph_cb)
        	graph_cb(sub_graph,ret);

       return ret;
}

bool CPUCore::RealRunGraph(void * graph_handle)
{
     CPURunner * runner=dev->GetRunner();

     return runner->Run(graph_handle);
}

bool CPUCore::RunNode(void * graph_handle, Node * node, dev_node_cb_t node_cb)
{
     bool ret=RealRunNode(graph_handle,node);

     if(node_cb)
         node_cb(node,ret);

     return ret;
}

bool CPUCore::RealRunNode(void * graph_handle, Node * node)
{
    return true;
}

void   CPUCore::PushGraph(void * graph_handle, Subgraph * sub_graph, dev_graph_cb_t graph_cb)
{
        graph_task g_task;

        g_task.graph_handle=graph_handle;
        g_task.sub_graph=sub_graph;
        g_task.graph_cb=graph_cb;
        
	std::unique_lock<std::mutex> cv_lock(worker_lock,std::defer_lock);
	cv_lock.lock();

	graph_list.push(g_task);

	cv_lock.unlock();

	worker_cv.notify_one();
}

bool CPUCore::Idle(void)
{
	bool ret=true;
	std::unique_lock<std::mutex> cv_lock(worker_lock,std::defer_lock);

	cv_lock.lock();

	if(!graph_list.empty() || !node_list.empty() ||
			!worker_idle || !aider_idle)
		ret=false;

	cv_lock.unlock();

	return ret;
}

void CPUCore::GetTask(graph_task& g_task , node_task& n_task)
{
	std::unique_lock<std::mutex> cv_lock(worker_lock);

        if(graph_list.empty()&& node_list.empty())
	     worker_cv.wait(cv_lock,[this]{return (!graph_list.empty() || !node_list.empty());});

	if(!graph_list.empty())
	{
		g_task=graph_list.front();
		graph_list.pop();
	}

	if(!node_list.empty())
	{
		n_task=node_list.front();
		node_list.pop();
	}

	cv_lock.unlock();

}

void CPUCore::PushNode(void * graph_handle, Node *node, dev_node_cb_t node_cb)
{
        node_task n_task;

        n_task.graph_handle=graph_handle;
        n_task.node=node;
        n_task.node_cb=node_cb;

	std::unique_lock<std::mutex> cv_lock(worker_lock,std::defer_lock);
	cv_lock.lock();

	node_list.push(n_task);

	cv_lock.unlock();

	worker_cv.notify_one();

}


void CPUCore::CPUWorker(void)
{
	//bind itself to CPU
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(hw_cpu_id,&mask);

	if(sched_setaffinity(0,sizeof(mask),&mask)<0)
	{
		LOG_ERROR()<<"worker: failed to bind cpu: "<<hw_cpu_id<<"\n";
//		return;
	}

//        std::cout<<"Worker working on cpu: "<<hw_cpu_id<<"\n";

	worker_running=true;

	while(true)
        {
                graph_task g_task;
                node_task n_task;

                g_task.graph_handle=nullptr;
                n_task.graph_handle=nullptr;

		GetTask(g_task,n_task);

		if(g_task.graph_handle==nullptr && n_task.graph_handle==nullptr)
			break;

		if(g_task.graph_handle)
			RunGraph(g_task.graph_handle,g_task.sub_graph,g_task.graph_cb);
		if(n_task.graph_handle)
			RunNode(n_task.graph_handle,n_task.node,n_task.node_cb);

	}

	worker_running=false;

}

void CPUCore::CPUAider(void)
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(hw_cpu_id,&mask);

	if(sched_setaffinity(0,sizeof(mask),&mask)<0)
	{
		LOG_ERROR()<<"aider: failed to bind cpu: "<<hw_cpu_id<<"\n";
		//return;
	}

        //std::cout<<"Aider working on cpu: "<<hw_cpu_id<<"\n";

	aider_running=true;

	while(true)
	{

		sub_op_task task;

		task=dev->GetAiderTask();
             
         	if(task.exec_func==nullptr)
			break;
			
		task.exec_func(hw_cpu_id,task.seq,task.data);

	}

	aider_running=false;

}



bool CPUCore::LaunchWorker(void)
{
	if(master)
	{
            auto func=std::bind(&CPUCore::CPUWorker,this);

            worker=new std::thread(func);	
	}

	return true;
}

bool CPUCore::LaunchAider(void)
{
        auto func=std::bind(&CPUCore::CPUAider,this);

	aider=new std::thread(func);
	return true;
}

void CPUCore::StopWorker(void)
{
	PushGraph(nullptr,nullptr,nullptr);
}

void CPUCore::StopAider(void)
{
	std::vector<sub_op_task> dummy;

        sub_op_task task;

	task.exec_func=nullptr;
	
	dummy.push_back(task);

	dev->PushAiderTask(dummy,-1);
}


sub_op_task CPUDevice::GetAiderTask(void)
{
	std::unique_lock<std::mutex> cv_lock(aider_queue_lock);

        if(aider_task_list.empty())
	    aider_queue_cv.wait(cv_lock,[this]{return !aider_task_list.empty();});

	sub_op_task task=aider_task_list.front();
	aider_task_list.pop();

	cv_lock.unlock();

	return task;
}

bool   CPUDevice::PushAiderTask(std::vector<sub_op_task>& task_list, int cpu)
{
	std::unique_lock<std::mutex> cv_lock(aider_queue_lock,std::defer_lock);

	cv_lock.lock();

	for(auto task: task_list)
		aider_task_list.push(task);

	cv_lock.unlock();

	aider_queue_cv.notify_all();

        return true;
}


bool CPUDriver::InitializeDevice(Device * device)
{
	CPUDevice * cpu_device=dynamic_cast<CPUDevice *>(device);

	for(unsigned int i=0;i<cpu_device->compute_cores.size();i++)
	{
		CPUCore * cpu_core=cpu_device->compute_cores[i];
		cpu_core->LaunchWorker();
		cpu_core->LaunchAider();
	}

       cpu_device->BindDriver(this);
	cpu_device->dev_status=kDevStopped;

        CPURunner * runner=cpu_device->GetRunner();

        auto func=std::bind(&CPUDevice::PushAiderTask,cpu_device,std::placeholders::_1,
			                std::placeholders::_2);
		
        runner->SetHelper(std::malloc,std::free,func);


	return true;
}

bool CPUDriver::ReleaseDevice(Device * device)
{
	CPUDevice * cpu_device=dynamic_cast<CPUDevice *>(device);

	StopDevice(device);

	for(unsigned int i=0;i<cpu_device->compute_cores.size();i++)
	{
		CPUCore * cpu_core=cpu_device->compute_cores[i];

		cpu_core->StopWorker();
		cpu_core->StopAider();
	}

	for(unsigned int i=0;i<cpu_device->compute_cores.size();i++)
	{
		CPUCore * cpu_core=cpu_device->compute_cores[i];

		while(cpu_core->aider_running || cpu_core->worker_running)
			__asm__ __volatile__("":::"memory");

		if(cpu_core->worker)
                {
                        cpu_core->worker->join();
			delete cpu_core->worker;
                }

                cpu_core->aider->join();
		delete cpu_core->aider;
	}	

	cpu_device->dev_status=kDevRemoved;

	return true;
}


bool CPUDriver::StartDevice(Device * device) 
{
	CPUDevice * cpu_device=dynamic_cast<CPUDevice *>(device);

        if(cpu_device->dev_status==kDevInvalid ||
           cpu_device->dev_status==kDevRemoved)
	     return false;
	
	cpu_device->dev_status=kDevNormal;

	return true;
}

bool CPUDriver::StopDevice(Device * device) 
{
	CPUDevice * cpu_device=dynamic_cast<CPUDevice *>(device);

	if(cpu_device->dev_status==kDevStopped)
		return true;

	cpu_device->dev_status=kDevStopped;

	for(unsigned int i=0;i<cpu_device->compute_cores.size();i++)
	{
		CPUCore * cpu_core=cpu_device->compute_cores[i];

		while(!cpu_core->Idle())
		{
			//std::chrono::duration<int,std::milli> sleep_time(2);
                        std::chrono::milliseconds sleep_time(2);
			std::this_thread::sleep_for(sleep_time);
		}
	}


	return true;
}

dev_status_t CPUDriver::GetDeviceStatus(Device * device)
{
	CPUDevice * cpu_device=dynamic_cast<CPUDevice *>(device);
	return cpu_device->dev_status;
}

void * CPUDriver::CreateGraphHandle(Device * dev,Subgraph * graph) 
{
	DevContext * context=new DevContext();
        CPUDevice * cpu_device=dynamic_cast<CPUDevice *>(dev);
        CPUCore * cpu_core=cpu_device->GetMasterCPU();

        context->dev=cpu_device;
        context->runner_context=cpu_core->CreateGraphHandle(graph);

        context->sub_graph=graph;

        context->graph_cb=nullptr;
        context->node_cb=nullptr;

	return context;
}

void * CPUDriver::CreateGraphHandle(Device * dev) 
{
	DevContext * context=new DevContext();
	context->dev=dynamic_cast<CPUDevice *>(dev);

        context->runner_context=nullptr;
        context->graph_cb=nullptr;
        context->node_cb=nullptr;

	return context;
}

bool CPUDriver::ReleaseGraphHandle(Device * dev, void * graph_handle) 
{
	DevContext * context=reinterpret_cast<DevContext *>(graph_handle);
        CPUDevice * cpu_device=dynamic_cast<CPUDevice *>(dev);
        CPUCore * cpu_core=cpu_device->GetMasterCPU();

        cpu_core->ReleaseGraphHandle(context->runner_context);

	delete context;
	return true;
}


void  CPUDriver::SetGraphDoneHook(Device * dev, void * graph_handle, dev_graph_cb_t func) 
{
	DevContext * context=reinterpret_cast<DevContext *>(graph_handle);
        context->graph_cb=func;
}

void  CPUDriver::SetNodeDoneHook(Device * dev, void * graph_handle, dev_node_cb_t func) 
{
	DevContext * context=reinterpret_cast<DevContext *>(graph_handle);
        context->node_cb=func;
}


bool CPUDriver::OptimizeGraph(Device * dev, void * graph_handle, Subgraph * graph)
{

	DevContext * context=reinterpret_cast<DevContext *>(graph_handle);
        CPUDevice * cpu_device=context->dev;
        CPUCore * cpu_core=cpu_device->GetMasterCPU();

        if(context->runner_context==nullptr)
        {
             context->runner_context=cpu_core->CreateGraphHandle(graph);
        }

	context->sub_graph=graph;

	return cpu_core->OptimizeGraph(context->runner_context);
}

bool CPUDriver::OptimizeGraph(Device * dev, void * graph_handle) 
{
	DevContext * context=reinterpret_cast<DevContext *>(graph_handle);

	return OptimizeGraph(dev,graph_handle,context->sub_graph);
}


bool CPUDriver::Prerun(Device * dev, void * graph_handle)
{
	DevContext * context=reinterpret_cast<DevContext *>(graph_handle);
	CPUDevice * cpu_device=context->dev;
	CPUCore * cpu_core=cpu_device->GetMasterCPU();


	return cpu_core->Prerun(context->runner_context);
}

bool CPUDriver::SyncRun(Device * dev, void * graph_handle)
{
	DevContext * context=reinterpret_cast<DevContext *>(graph_handle);
	CPUDevice * cpu_device=context->dev;
        CPUCore * cpu_core=cpu_device->GetMasterCPU();

        return cpu_core->RealRunGraph(context->runner_context);
}

bool CPUDriver::Run(Device * dev, void * graph_handle)
{
	DevContext * context=reinterpret_cast<DevContext *>(graph_handle);
	CPUDevice * cpu_device=context->dev;
	CPUCore * cpu_core=cpu_device->GetMasterCPU();

	cpu_core->PushGraph(context->runner_context,context->sub_graph,context->graph_cb);

	return true;
}

bool CPUDriver::Postrun(Device * dev, void * graph_handle)
{
	DevContext * context=reinterpret_cast<DevContext *>(graph_handle);
	CPUDevice * cpu_device=context->dev;
	CPUCore * cpu_core=cpu_device->GetMasterCPU();

	cpu_core->Postrun(context->runner_context);	

	return true;
}



} //namespace TEngine
