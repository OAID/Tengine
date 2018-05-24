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
 * Copyright (c) 2018, Linaro Ltd
 * Author: haitao@openailab.com
 * Author: Manivannan Sadhasivam <manivannan.sadhasivam@linaro.org>
 * Author: Mark Charlebois <mark.charlebois@linaro.org>
 */
#include "apq8096_driver.hpp"
#include "apq8096_executor.hpp"


namespace TEngine {

void APQ8096Executor::DevGetWorkload(DevWorkload& load)
{
	//TO BE IMPLEMENTED;
}


bool APQ8096Executor::DevGetPerf(Subgraph * graph,int policy,GraphPerf& perf)
{
	//TO BE IMPLEMENTED;
	return false;
}

float APQ8096Executor::DevGetFops(Subgraph * graph)
{
	//TO BE IMPLEMENTED;
	return 0;
}

void * APQ8096Executor::DevCreateGraphHandle(Subgraph * graph)
{
	return backend_dev_->CreateGraphHandle(graph);

}

bool APQ8096Executor::DevOptimzeGraph(void * graph_handle)
{
	return backend_dev_->OptimizeGraph(graph_handle);
}

bool APQ8096Executor::DevPrerun(void * graph_handle)
{
	return backend_dev_->Prerun(graph_handle);
}

bool APQ8096Executor::DevRun(void * graph_handle)
{
        auto f=std::bind(&APQ8096Executor::OnSubgraphDone,this,std::placeholders::_1,
				std::placeholders::_2);

        backend_dev_->SetGraphDoneHook(graph_handle,dev_graph_cb_t(f));

	return backend_dev_->Run(graph_handle);
}

bool APQ8096Executor::DevSyncRun(void * graph_handle)
{
	return backend_dev_->SyncRun(graph_handle);
}

bool APQ8096Executor::DevPostrun(void * graph_handle)
{
	return backend_dev_->Postrun(graph_handle);
}

bool APQ8096Executor::DevReleaseGraphHandle(void * graph_handle)
{
	return backend_dev_->ReleaseGraphHandle(graph_handle);
}


const dev_id_t& APQ8096Executor::DevGetID(void)
{
	return backend_dev_->GetDeviceID();
}

const dev_type_t & APQ8096Executor::DevGetType(void)
{
	return backend_dev_->GetDeviceType();
}

dev_status_t APQ8096Executor::DevGetStatus(void)
{
	return backend_dev_->GetDeviceStatus();
}

bool APQ8096Executor::Init(void)
{
	return true;
}

bool APQ8096Executor::Release(void)
{
	return true;
}

void  APQ8096Executor::UnbindDevice(void)
{
	backend_dev_=nullptr;
}

void APQ8096Executor::BindDevice(Device *  device)
{
	CPUDevice * dev=dynamic_cast<CPUDevice *>(device);
	backend_dev_=dev;
}

bool APQ8096Executor::DevStart(void) 
{
	return 	backend_dev_->Start();
}

bool APQ8096Executor::DevStop(void)
{
	return  backend_dev_->Stop();
}


} //namespace TEngine



