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
#ifndef __CPU_DRIVER_HPP__
#define __CPU_DRIVER_HPP__

#include <vector>
#include <string>
#include <atomic>
#include <queue>
#include <thread>
#include <condition_variable>


#include "graph.hpp"
#include "device_driver.hpp"
#include "soc_runner.hpp"

namespace TEngine {

class CPUDevice;

struct graph_task {
	void * graph_handle;
	Subgraph * sub_graph;
	dev_graph_cb_t graph_cb;
};

struct node_task {
	void * graph_handle;
	Node * node;
	dev_node_cb_t node_cb;
};


struct  CPUCore {


	void * CreateGraphHandle(Subgraph * sub_graph);
	void ReleaseGraphHandle(void * graph_handle);
	bool OptimizeGraph(void * graph_handle);


	bool Prerun(void * graph_handle);
	bool Postrun(void * graph_handle);
	bool RunGraph(void * graph_handle, Subgraph * sub_graph, dev_graph_cb_t graph_cb);
	bool RealRunGraph(void * graph_handle);



	bool RunNode(void * graph_handle, Node * node, dev_node_cb_t node_cb);
	bool RealRunNode(void * graph_handle, Node * node);   

	void PushGraph(void * graph_handle, Subgraph * sub_graph, dev_graph_cb_t graph_cb);
	void PushNode(void * graph_handle, Node *node, dev_node_cb_t node_cb);


	void GetTask(graph_task& g_task , node_task& n_task);

	bool LaunchWorker(void);
	bool LaunchAider(void);

	void StopWorker(void);
	void StopAider(void);


	bool Idle(void);

	void CPUWorker(void);       
	void CPUAider(void);


	CPUCore() {
		worker=nullptr;
		aider=nullptr;
		master=false;
		worker_running=false;
		aider_running=false;
		worker_idle=true;
		aider_idle=true;
	}

	std::string cpu_type;

	int hw_cpu_id;
	bool master;

	std::thread * worker;
	std::thread * aider;

	bool worker_running;
	bool aider_running;	  
	bool worker_idle;
	bool aider_idle;
	std::mutex worker_lock;
	std::condition_variable worker_cv;

	std::queue<graph_task> graph_list;
	std::queue<node_task> node_list;

	CPUDevice * dev;

};


class CPUDevice : public Device {

public:

	CPUDevice(const dev_id_t & dev_id): Device(dev_id) { backend_runner=nullptr;}
	CPUCore * GetMasterCPU(void) { return compute_cores[master_cpu_idx];}

	virtual ~CPUDevice() 
	{
		for(auto cpu : compute_cores)
			delete cpu;

		if(backend_runner)
			delete backend_runner;
	}

	sub_op_task  GetAiderTask(void);

	bool  PushAiderTask(std::vector<sub_op_task>& task_list, int cpu);

	CPURunner * GetRunner(void) { return backend_runner;}

	std::vector<CPUCore *> compute_cores;
	int master_cpu_idx;

	std::mutex aider_queue_lock;
	std::condition_variable aider_queue_cv;
	std::queue<sub_op_task> aider_task_list;
	
	dev_status_t dev_status;

	CPURunner * backend_runner;
};



class CPUDriver: public Driver {
public:
	struct  DevContext
	{
		CPUDevice * dev;
		Subgraph * sub_graph;
		void *  runner_context;
		dev_graph_cb_t graph_cb;
		dev_node_cb_t node_cb;
	};


	bool InitializeDevice(Device * device) override;
	bool ReleaseDevice(Device * device) override;

	bool StartDevice(Device * device) override;
	bool StopDevice(Device * device) override;
	dev_status_t GetDeviceStatus(Device * device) override;

	void * CreateGraphHandle(Device * dev,Subgraph * graph) override; 
	void * CreateGraphHandle(Device * dev) override;
	bool ReleaseGraphHandle(Device * dev, void * graph_handle) override;

	void  SetGraphDoneHook(Device * dev, void * graph_handle, dev_graph_cb_t func) override;
	void  SetNodeDoneHook(Device * dev, void * node_handle, dev_node_cb_t func) override;

	bool OptimizeGraph(Device * dev, void * graph_handle, Subgraph * graph) override;
	bool OptimizeGraph(Device * dev, void * graph_handle) override;

	bool Prerun(Device * dev, void * graph_handle) override;
	bool Run(Device * dev, void * graph_handle) override;
	bool SyncRun(Device * dev, void * graph_handle) override;
	bool Postrun(Device * dev, void * graph_handle) override;


	bool Prerun(Device * dev, void * node_handle, Node * node) override {return false;}
	bool Run(Device * dev, void * node_handle, Node * node) override {return false;}
	bool SyncRun(Device * dev, void * node_handle, Node * node) override {return false;}
	bool Postrun(Device * dev, void * node_handle, Node  * node) override {return false;}

	Subgraph * GetSubgraphFromHandle(void * graph_handle)
	{
		DevContext * context=reinterpret_cast<DevContext *>(graph_handle);
		return context->sub_graph;
	}

};


} //namespace TEngine

#endif
