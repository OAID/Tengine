#ifndef __TEST_DEVICE_HPP__
#define __TEST_DEVICE_HPP__

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "soc_runner.hpp"

namespace TEngine {

struct TestDevice {

void  LaunchAider(void)
{
    thread_list.resize(soc_info.cpu_list.size());

    for(unsigned int i=0;i<soc_info.cpu_list.size();i++)
    {
        int cpu_id=soc_info.cpu_list[i];

        auto f=std::bind(&TestDevice::Aider,this,std::placeholders::_1);

        thread_list[i]=new std::thread(f,cpu_id);
    }

}

bool TaskDispatch(std::vector<sub_op_task>& tasks, int cpu)
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

void  Aider(int cpu)
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

        std::cout<<"worker exit on cpu: "<<cpu<<"\n";
}

void StopAider(void)
{
       int tr_num=thread_list.size();

        std::vector<sub_op_task> task_list;

        for(int i=0;i<tr_num;i++)
        {
           sub_op_task task;
           task.exec_func=nullptr;

           task_list.push_back(task);
        
        }
        
        TaskDispatch(task_list,-1);

}


void BindMaster(void)
{
        int cpu_id=soc_info.master_cpu;

       	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpu_id,&mask);

        if(sched_setaffinity(0,sizeof(mask),&mask)<0)
	{
		std::cout<<"master: failed to bind cpu: "<<cpu_id<<"\n";
	}

        std::cout<<"master ready on cpu: "<<cpu_id<<"\n";
}



SocInfo soc_info;

std::mutex queue_lock;
std::condition_variable queue_cv;
std::queue<sub_op_task> task_queue;

std::vector<std::thread *> thread_list;

};

} //namespace TEngine




#endif
