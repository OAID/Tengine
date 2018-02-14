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
#include "cpu_engine.hpp"
#include "tensor_mem.hpp"
#include "prof_record.hpp"
#include "graph_optimizer.hpp"

#define ATTR_NODE_RUNNER "NodeRunner"

namespace TEngine {

CPUEngine::CPUEngine(void) {
	name="cpu_engine";
	initialized_=false;

}

CPUEngine::~CPUEngine(void) {
        
        int tr_num=thread_list.size();

        std::vector<sub_op_task> task_list;

        for(int i=0;i<tr_num;i++)
        {
           sub_op_task task;
           task.exec_func=nullptr;

           task_list.push_back(task);
        
        }
        
        TaskDispatch(task_list,-1);


	for (auto tr : thread_list)
        {
                tr->join();

		delete tr;
        }
}

exec_handle_t CPUEngine::AddGraphExecutor(GraphExecutor * graph_executor)
{

	if(!initialized_)
	{
		SocInfo soc_info;
                std::string soc_name="APQ8096";

		if(!GetPredefinedSoc(soc_name,soc_info))
		{
			XLOG_ERROR()<<"cannot get soc definition for "<<soc_name<<"\n";
			return nullptr;
		}

		/* TODO: adjust soc according to etc/config */

                /* bind itself to master cpu */

                int master_cpu=soc_info.master_cpu;
                
                cpu_set_t mask;
	        CPU_ZERO(&mask);
	        CPU_SET(master_cpu,&mask);

                if(sched_setaffinity(0,sizeof(mask),&mask)<0)
	        {
		     std::cout<<"master: failed to bind cpu: "<<master_cpu<<"\n";
	        }

                LaunchAider(&soc_info);

		backend_runner.SetSocInfo(soc_info);

                auto f=std::bind(&CPUEngine::TaskDispatch,this,std::placeholders::_1,
                                                 std::placeholders::_2);

		backend_runner.SetHelper(std::malloc,std::free,f);

		initialized_=true;        

	}


	exec_env * env=new exec_env();
	env->graph_executor=graph_executor;
	env->status=EXEC_STATUS_CREATED;

	Graph * graph=graph_executor->GetGraph();

	void * graph_handle=backend_runner.CreateGraphHandle(graph);

	env->graph_handle=graph_handle;

	any * ret=new any();

	(*ret)=env;

	return ret;
}

void * CPUEngine::GetTensorBuffer(Tensor * tensor, exec_handle_t h)
{
	return get_tensor_mem(tensor);
}

bool CPUEngine::SetTensorBuffer(Tensor * tensor, void *addr, int size, exec_handle_t h)
{
	return set_tensor_mem(tensor,addr,size,nullptr);
}

bool CPUEngine::Prerun(exec_handle_t h)
{
	exec_env * env=any_cast<exec_env *>(*h);


	void * graph_handle=env->graph_handle;

	if(!backend_runner.OptimizeGraph(graph_handle) ||
			!backend_runner.Prerun(graph_handle))
		return false;

	env->status=EXEC_STATUS_INITED;

	return true;
}


bool CPUEngine::SyncRun(exec_handle_t h)
{
	exec_env * env=any_cast<exec_env *>(*h);


	int s=env->status;

	if(s!= EXEC_STATUS_INITED && s!=EXEC_STATUS_DONE)
	{
		return false;
	}

	env->status=EXEC_STATUS_RUN;

	bool ret=backend_runner.Run(env->graph_handle);

	if(ret)
		env->status=EXEC_STATUS_DONE;
	else
		env->status=EXEC_STATUS_BAD;

	return true;
}


bool CPUEngine::Postrun(exec_handle_t h)
{
	exec_env * env=any_cast<exec_env *>(*h);

	return backend_runner.Postrun(env->graph_handle);
}


exec_status_t CPUEngine::GetStatus(exec_handle_t h) 
{
	exec_env * env=any_cast<exec_env *>(*h);

	return env->status;
}

const std::string& CPUEngine::GetStatusStr(const exec_status_t& status)
{
	static std::string created="CREATED";
	static std::string inited="INITED";
	static std::string run="RUN";
	static std::string done="DONE";
	static std::string bad="BAD";
	static std::string unknown="UNKNOWN";

	int s=any_cast<int>(status);

	switch(s)
	{
		case EXEC_STATUS_CREATED:
			return created;
		case EXEC_STATUS_INITED:
			return inited;
		case EXEC_STATUS_RUN:
			return run;
		case EXEC_STATUS_DONE:
			return done;
		case EXEC_STATUS_BAD:
			return bad;
		default:
			break;
	}

	return unknown; 
}

int CPUEngine::GetStatusCode(const exec_status_t& status)
{
	int s=any_cast<int>(status);

	return s;
}

std::string  CPUEngine::GetErrorStr(exec_handle_t h)
{
	return "NO ERROR:-)\n";
}

bool CPUEngine::RemoveGraphExecutor(exec_handle_t h)
{
	exec_env * env=any_cast<exec_env *>(*h);

	backend_runner.ReleaseGraphHandle(env->graph_handle);

	delete env;
	delete h;

	return true;
}

void CPUEngine::Aider(int cpu)
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
		     queue_cv.wait(cv_lock,[this]{return !task_queue.empty();});

		task=task_queue.front();

		task_queue.pop();

		cv_lock.unlock();

                if(task.exec_func==nullptr)
                      break;

		task.exec_func(cpu,task.seq,task.data);     
	}


}

void CPUEngine::LaunchAider(SocInfo * soc_info)
{
    thread_list.resize(soc_info->cpu_list.size());

    for(unsigned int i=0;i<soc_info->cpu_list.size();i++)
    {
        int cpu_id=soc_info->cpu_list[i];

        auto f=std::bind(&CPUEngine::Aider,this,std::placeholders::_1);

        thread_list[i]=new std::thread(f,cpu_id);
    }
}



bool CPUEngine::TaskDispatch(std::vector<sub_op_task>& tasks, int cpu)
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




} //namespace TEngine
