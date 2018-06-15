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
#include "cpu_executor.hpp"


namespace TEngine {

bool  CPUExecutor::DevGetProposal(Subgraph * graph,int policy) 
{
      return backend_dev_->GetProposal(graph,policy);
}

void CPUExecutor::DevGetWorkload(DevWorkload& load)
{
         backend_dev_->GetWorkload(load);
}


bool CPUExecutor::DevGetPerf(Subgraph * graph,int policy,GraphPerf& perf)
{
	return backend_dev_->GetPerf(graph,policy,perf);
}

float CPUExecutor::DevGetFops(Subgraph * graph,int policy)
{
	return backend_dev_->GetFops(graph,policy);
}

int CPUExecutor::DevGetPolicyPriority(int policy)
{
        return backend_dev_->GetPolicyPriority(policy);
}

bool CPUExecutor::DevSetConfig(const char * config_name, const void * buffer, int size)
{
        return backend_dev_->SetDevConfig(config_name,buffer,size);
}

bool CPUExecutor::DevGetConfig(const char * config_name, void * buffer, int size)
{
        return backend_dev_->GetDevConfig(config_name,buffer,size);
}

bool CPUExecutor::DevDelConfig(const char * config_name)
{
        return backend_dev_->DelDevConfig(config_name);
}


void * CPUExecutor::DevCreateGraphHandle(Subgraph * graph)
{
	return backend_dev_->CreateGraphHandle(graph);

}

bool CPUExecutor::DevOptimizeGraph(void * graph_handle)
{
	return backend_dev_->OptimizeGraph(graph_handle);
}

Subgraph *  CPUExecutor::DevGetOptimizedGraph(void * graph_handle)
{
	return backend_dev_->GetOptimizedGraph(graph_handle);
}

bool CPUExecutor::DevPrerun(void * graph_handle)
{
	return backend_dev_->Prerun(graph_handle);
}

bool CPUExecutor::DevRun(void * graph_handle)
{
        auto f=std::bind(&CPUExecutor::OnSubgraphDone,this,std::placeholders::_1,
				std::placeholders::_2);

        backend_dev_->SetGraphDoneHook(graph_handle,dev_graph_cb_t(f));

	return backend_dev_->Run(graph_handle);
}

bool CPUExecutor::DevSyncRun(void * graph_handle)
{
	return backend_dev_->SyncRun(graph_handle);
}

bool CPUExecutor::DevPostrun(void * graph_handle)
{
	return backend_dev_->Postrun(graph_handle);
}

bool CPUExecutor::DevReleaseGraphHandle(void * graph_handle)
{
	return backend_dev_->ReleaseGraphHandle(graph_handle);
}


const dev_id_t& CPUExecutor::DevGetID(void)
{
	return backend_dev_->GetDeviceID();
}

const dev_type_t & CPUExecutor::DevGetType(void)
{
	return backend_dev_->GetDeviceType();
}

dev_status_t CPUExecutor::DevGetStatus(void)
{
	return backend_dev_->GetDeviceStatus();
}

bool CPUExecutor::Init(void)
{
	return true;
}

bool CPUExecutor::Release(void)
{
	return true;
}

void  CPUExecutor::UnbindDevice(void)
{
	backend_dev_=nullptr;
}

void CPUExecutor::BindDevice(Device *  device)
{
	CPUDevice * dev=dynamic_cast<CPUDevice *>(device);
	backend_dev_=dev;
}

bool CPUExecutor::DevStart(void) 
{
	return 	backend_dev_->Start();
}

bool CPUExecutor::DevStop(void)
{
	return  backend_dev_->Stop();
}


} //namespace TEngine



