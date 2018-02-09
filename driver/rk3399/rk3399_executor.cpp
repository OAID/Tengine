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


#include "rk3399_driver.hpp"
#include "rk3399_executor.hpp"


namespace TEngine {

void RK3399Executor::DevGetWorkload(DevWorkload& load)
{
	//TO BE IMPLEMENTED;
}


bool RK3399Executor::DevGetPerf(Subgraph * graph,int policy,GraphPerf& perf)
{
	//TO BE IMPLEMENTED;
	return false;
}

float RK3399Executor::DevGetFops(Subgraph * graph)
{
	//TO BE IMPLEMENTED;
	return 0;
}

void * RK3399Executor::DevCreateGraphHandle(Subgraph * graph)
{
	return backend_dev_->CreateGraphHandle(graph);

}

bool RK3399Executor::DevOptimzeGraph(void * graph_handle)
{
	return backend_dev_->OptimizeGraph(graph_handle);
}

bool RK3399Executor::DevPrerun(void * graph_handle)
{
	return backend_dev_->Prerun(graph_handle);
}

bool RK3399Executor::DevRun(void * graph_handle)
{
        auto f=std::bind(&RK3399Executor::OnSubgraphDone,this,std::placeholders::_1,
				std::placeholders::_2);

        backend_dev_->SetGraphDoneHook(graph_handle,dev_graph_cb_t(f));

	return backend_dev_->Run(graph_handle);
}

bool RK3399Executor::DevSyncRun(void * graph_handle)
{
	return backend_dev_->SyncRun(graph_handle);
}

bool RK3399Executor::DevPostrun(void * graph_handle)
{
	return backend_dev_->Postrun(graph_handle);
}

bool RK3399Executor::DevReleaseGraphHandle(void * graph_handle)
{
	return backend_dev_->ReleaseGraphHandle(graph_handle);
}


const dev_id_t& RK3399Executor::DevGetID(void)
{
	return backend_dev_->GetDeviceID();
}

const dev_type_t & RK3399Executor::DevGetType(void)
{
	return backend_dev_->GetDeviceType();
}

dev_status_t RK3399Executor::DevGetStatus(void)
{
	return backend_dev_->GetDeviceStatus();
}

bool RK3399Executor::Init(void)
{
	return true;
}

bool RK3399Executor::Release(void)
{
	return true;
}

void  RK3399Executor::UnbindDevice(void)
{
	backend_dev_=nullptr;
}

void RK3399Executor::BindDevice(Device *  device)
{
	CPUDevice * dev=dynamic_cast<CPUDevice *>(device);
	backend_dev_=dev;
}

bool RK3399Executor::DevStart(void) 
{
	return 	backend_dev_->Start();
}

bool RK3399Executor::DevStop(void)
{
	return  backend_dev_->Stop();
}


} //namespace TEngine



