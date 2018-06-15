
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

#ifndef __WORKER_THREAD_HPP__
#define __WORKER_THREAD_HPP__

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>


namespace TEngine {


template <typename T>
class WorkerThread {

public:
   using process_t=std::function<void(const T&, int )>;
   using count_func_t=std::function<void(int)>;
   
   WorkerThread(const process_t& func, int cpu){bind_cpu_=cpu;Init(func);}
   WorkerThread(const process_t& func) {bind_cpu_=-1;Init(func);}

   ~WorkerThread()
   {
      if(worker_)
      {
         StopWorker();
         worker_->join();
         delete worker_;  
      }
   }

   void PushTask(const std::vector<T>& task_list)
   {
      std::unique_lock<std::mutex> cv_lock(*worker_lock_);

      for(auto task: task_list)
         task_queue_->push(task);

      if(inc_req_)
           inc_req_(task_list.size());

      cv_lock.unlock();

      worker_cv_->notify_all();
   }

   void SetQueue(std::queue<T> * task_queue,std::mutex * worker_lock,std::condition_variable *worker_cv)
   {
       task_queue_=task_queue;
       worker_lock_=worker_lock;
       worker_cv_=worker_cv;
   }

   void SetCount(const count_func_t& inc_req, const count_func_t& inc_done)
   {
        inc_req_=inc_req;
        inc_done_=inc_done;
   }


   void StopWorker(void)
   {
       std::unique_lock<std::mutex> cv_lock(*worker_lock_);
       quit_work_=true;
       cv_lock.unlock();

       worker_cv_->notify_all();
   }


   bool LaunchWorker(void)
   {
      auto func=std::bind(&WorkerThread::DoWork,this);
      worker_=new std::thread(func);
      return true;
   }

private:

   void DoWork(void)
   {
       //bind CPU first
       if(bind_cpu_>=0)
       {
          cpu_set_t mask;
       	  CPU_ZERO(&mask);
	  CPU_SET(bind_cpu_,&mask);

          if(sched_setaffinity(0,sizeof(mask),&mask)==0)
	            bind_done_=true;
       }
       
       while(true)
       {
          T task;
          GetTask(task);
       
          if(quit_work_)
             break;
           
          process_(task,bind_cpu_);

          if(inc_done_)
               inc_done_(1);
          
      }
   }
   
   void GetTask(T& task)
   {
     std::unique_lock<std::mutex> cv_lock(*worker_lock_);

     if(task_queue_->empty()&&!quit_work_)
        worker_cv_->wait(cv_lock,[this]{ return !task_queue_->empty() || quit_work_;});

      if(!quit_work_)
      {
         task=task_queue_->front();
         task_queue_->pop();
      }

     cv_lock.unlock();
   }
   
   
   void Init(const process_t& func)
   {
      bind_done_=false;
      quit_work_=false;

      process_=func;

      task_queue_=nullptr;
      worker_lock_=nullptr;
      worker_cv_=nullptr;

  //    LaunchWorker();
   }
   
   
  int bind_cpu_;
  bool bind_done_;
  bool quit_work_;
  process_t process_;
  
  std::queue<T> * task_queue_;
  std::mutex * worker_lock_;
  std::condition_variable *worker_cv_;
  std::thread * worker_;

  count_func_t inc_req_;
  count_func_t inc_done_;

};


} //namespace TEngine





#endif
