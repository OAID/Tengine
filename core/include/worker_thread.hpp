
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

<<<<<<< HEAD
#include <sys/time.h>
=======
#include <sched.h>
#include <sys/time.h>
#include <errno.h>
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
#include <string.h>

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

<<<<<<< HEAD
=======
#include "logger.hpp"

>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
namespace TEngine {

template <typename T> class WorkerThread
{
public:
    using process_t = std::function<void(const T&, int)>;
    using count_func_t = std::function<void(int)>;

    WorkerThread(const process_t& func, int cpu)
    {
        bind_cpu_ = cpu;
        Init(func);
    }
    WorkerThread(const process_t& func)
    {
        bind_cpu_ = -1;
        Init(func);
    }

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

        for(auto task : task_list)
            task_queue_->push(task);

        if(inc_req_)
            inc_req_(task_list.size());

        cv_lock.unlock();

        worker_cv_->notify_all();
    }

    void SetQueue(std::queue<T>* task_queue, std::mutex* worker_lock, std::condition_variable* worker_cv)
    {
        task_queue_ = task_queue;
        worker_lock_ = worker_lock;
        worker_cv_ = worker_cv;
    }

    void SetCount(const count_func_t& inc_req, const count_func_t& inc_done)
    {
        inc_req_ = inc_req;
        inc_done_ = inc_done;
    }

    void StopWorker(void)
    {
        std::unique_lock<std::mutex> cv_lock(*worker_lock_);
<<<<<<< HEAD
        quit_work_ = true;
        cv_lock.unlock();

        worker_cv_->notify_all();
    }

    bool LaunchWorker(void)
    {
        auto func = std::bind(&WorkerThread::DoWork, this);
=======

        quit_work_ = true;
        worker_cv_->notify_all();
        active_cv_.notify_one();

        cv_lock.unlock();
    }

    bool LaunchWorker(bool master_cpu=false)
    {
        auto func = std::bind(&WorkerThread::DoWork, this,master_cpu);
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
        worker_ = new std::thread(func);
        return true;
    }

<<<<<<< HEAD
private:
    void DoWork(void)
    {
        int task_done_count = 0;
        bool skip = false;

#ifdef CONFIG_MAX_RUN_TIME
        long start_time;

        struct timeval tv;

        gettimeofday(&tv, NULL);

        start_time = tv.tv_sec;
#endif
=======
    void Activate(int dispatch_cpu)
    {
         dispatch_cpu_=dispatch_cpu;
         active_count_++;

         std::unique_lock<std::mutex> cv_lock(*worker_lock_);
         active_cv_.notify_one();
    }

    void Deactivate(void)
    {
        active_count_--;
    }

private:
    void DoWork(bool master_cpu)
    {
        int task_done_count = 0;
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074

        // bind CPU first
        if(bind_cpu_ >= 0)
        {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(bind_cpu_, &mask);

            if(sched_setaffinity(0, sizeof(mask), &mask) == 0)
                bind_done_ = true;
        }

<<<<<<< HEAD
        while(true)
        {
            T task;
            GetTask(task);

            if(quit_work_)
                break;

#ifdef CONFIG_MAX_RUN_COUNT
            if(task_done_count > CONFIG_MAX_RUN_COUNT)
                skip = true;
#endif

#ifdef CONFIG_MAX_RUN_TIME
            if(!(task_done_count & 0x3fff))
            {
                struct timeval tv;

                gettimeofday(&tv, NULL);

                if((tv.tv_sec - start_time) >= CONFIG_MAX_RUN_TIME)
                    skip = true;
            }

#endif
            if(!skip)
                process_(task, bind_cpu_);

            if(inc_done_)
                inc_done_(1);

            task_done_count++;
        }
    }

    void GetTask(T& task)
    {
        std::unique_lock<std::mutex> cv_lock(*worker_lock_);

        if(task_queue_->empty() && !quit_work_)
            worker_cv_->wait(cv_lock, [this] { return !task_queue_->empty() || quit_work_; });

        if(!quit_work_)
        {
            task = task_queue_->front();
            task_queue_->pop();
        }

        cv_lock.unlock();
=======
	if(master_cpu)
	{
           // set scheduler
           struct sched_param sched_param;

           sched_param.sched_priority = 10;
           sched_setscheduler(0, SCHED_RR, &sched_param);
	}

	while(true)
        {
            while(active_count_>0)
            {
                T task;
    
                if(GetTask(task))
                {
                    process_(task, bind_cpu_);

                    if(inc_done_)
                        inc_done_(1);

                    task_done_count++;
                }
                else
                    std::this_thread::yield();
            }

            std::unique_lock<std::mutex> cv_lock(*worker_lock_);

            active_cv_.wait(cv_lock,[this]{return (active_count_>0 || quit_work_); });

            cv_lock.unlock();
 
            if(quit_work_)
                break;
        }
    }

    bool GetTask(T& task)
    {
        bool ret=false;
        std::unique_lock<std::mutex> cv_lock(*worker_lock_);

        if(bind_cpu_==dispatch_cpu_)
        {
            if(task_queue_->empty() && !quit_work_)
                worker_cv_->wait(cv_lock, [this] { return !task_queue_->empty() || quit_work_; });
        }

        if(!task_queue_->empty())
        {
            task = task_queue_->front();
            task_queue_->pop();
            ret=true;
        }

        cv_lock.unlock();

        return ret;
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
    }

    void Init(const process_t& func)
    {
        bind_done_ = false;
        quit_work_ = false;

        process_ = func;

        task_queue_ = nullptr;
        worker_lock_ = nullptr;
        worker_cv_ = nullptr;

<<<<<<< HEAD
=======
        active_count_=0;
        dispatch_cpu_=-100;

>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
        //    LaunchWorker();
    }

    int bind_cpu_;
    bool bind_done_;
    bool quit_work_;
    process_t process_;

<<<<<<< HEAD
=======
    std::atomic<unsigned int> active_count_;
    std::condition_variable  active_cv_;
    int dispatch_cpu_;


>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
    std::queue<T>* task_queue_;
    std::mutex* worker_lock_;
    std::condition_variable* worker_cv_;
    std::thread* worker_;

    count_func_t inc_req_;
    count_func_t inc_done_;
};

}    // namespace TEngine

#endif
