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

#include <sched.h>
#include <chrono>
#include <functional>

#include "tensor_mem.hpp"
#include "caffe_driver.hpp"
#include "caffe_executor.hpp"
#include "tengine_config.hpp"
#include "node_ops.hpp"
#include "logger.hpp"


#define CAFFE_NODE_DRIVER_NAME  "CaffeNode"
#define CAFFE_NODE_DEV_ID       "sw.caffe.cpu.node"

namespace TEngine {


CaffeNodeDriver::CaffeNodeDriver(void)
{
     SetName(CAFFE_NODE_DRIVER_NAME);

     dev_id_.push_back(CAFFE_NODE_DEV_ID);
}

CaffeNodeDriver::~CaffeNodeDriver(void)
{
}

bool CaffeNodeDriver::Prerun(Device * dev, void * node_handle, Node * node)
{

       int output_number=node->GetOutputNum();

       for(int i=0;i<output_number;i++)
       {
           Tensor * otensor = node->GetOutputTensor(i);

           if(!get_tensor_mem(otensor))
           {
	      int mem_size = otensor->GetTotalSize();
	      void* addr = std::malloc(mem_size);

	      set_tensor_mem(otensor,addr,mem_size,std::free);
           }
       }

       NodeOps * ops=NodeOpsRegistryManager::FindNodeOps(REF_REGISTRY_NAME,NULL,node);

       node->SetAttr("CaffeNodeOps",ops);

       return true;
}

bool CaffeNodeDriver::SyncRun(Device * dev, void * node_handle, Node * node)
{
        NodeOps * ops=any_cast<NodeOps *>(node->GetAttr("CaffeNodeOps"));

       std::cout<<"caffe run node: "<<node->GetName()<<"\n";

       ops->Run(node);

       return true;
}

bool CaffeNodeDriver::Run(Device * dev, void * node_handle, Node * node) 
{
	bool ret=SyncRun(dev,node_handle,node);

	DevContext * context=reinterpret_cast<DevContext *>(node_handle);

	if(context->node_cb)
		context->node_cb(node,ret);

	return ret; 
}

bool CaffeNodeDriver::Postrun(Device * dev, void * node_handle, Node  * node)
{
        Tensor * otensor=node->GetOutputTensor(0);
	free_tensor_mem(otensor);

        NodeOps * ops=any_cast<NodeOps *>(node->GetAttr("CaffeNodeOps"));

        ops->Release();

	return true;
}

bool CaffeNodeDriver::InitDev(NodeDevice * device)
{

     return true;
}

bool CaffeNodeDriver::ProbeDevice(const dev_id_t& dev_id) 
{
    CaffeNodeDevice * dev=new CaffeNodeDevice(dev_id);

    InitializeDevice(dev);
    dev->SetName(dev_id);

    dev_table_.push_back(dev);

    return true;

}


bool CaffeNodeDriver::DestroyDevice(Device * device) 
{
	CaffeNodeDevice * caffe_dev=dynamic_cast<CaffeNodeDevice *>(device);

	if(caffe_dev->dev_status!=kDevStopped)
		return false;

	ReleaseDevice(caffe_dev);

        auto ir=dev_table_.begin();

        while((*ir)!=caffe_dev && ir!=dev_table_.end())
        {
             ir++;
        }

        dev_table_.erase(ir);

        delete caffe_dev;

	return true;
}



//////////////////////////////////////////////

void CaffeDriverInit(void)
{
    CaffeNodeDriver * caffe_node_driver=new CaffeNodeDriver();

    std::cout<<"Register Driver: "<<caffe_node_driver->GetName()<<"\n";

    DriverManager::RegisterDriver(caffe_node_driver->GetName(),caffe_node_driver);

    auto dev_executor_factory=DevExecutorFactory::GetFactory();

    int n=caffe_node_driver->GetDevIDTableSize();

    for(int i=0;i<n;i++)
        dev_executor_factory->RegisterInterface<CaffeNodeExecutor,const dev_id_t&>
                (caffe_node_driver->GetDevIDByIdx(i));

    LOG_INFO()<<"Caffe Node Driver Initialized\n";
}







} //namespace TEngine

