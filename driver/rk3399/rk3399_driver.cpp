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
#include "rk3399_driver.hpp"
#include "rk3399_executor.hpp"
#include "logger.hpp"


namespace TEngine {


RK3399Driver::RK3399Driver(void)
{
	SetName("RK3399");

	id_table_.push_back("cpu.rk3399.a53.all");
	id_table_.push_back("cpu.rk3399.a72.all");
	id_table_.push_back("cpu.rk3399.cpu.all");
	id_table_.push_back("cpu.rk3399.a72.0");
	id_table_.push_back("cpu.rk3399.a53.2");

}


int RK3399Driver::ProbeDevice(void) 
{
	int number=0;

	for(unsigned int i=0; i<id_table_.size();i++)
	{
		const dev_id_t& dev_id=id_table_[i];
		if(ProbeDevice(dev_id))
			number++;
	}

	return number;
}

bool RK3399Driver::ProbeDevice(const dev_id_t& dev_id)
{
	if(dev_id==std::string("cpu.rk3399.a72.all"))
	{
		CPUDevice * dev=new A72Device();

		dev->SetName(dev_id);

		InitializeDevice(dev);

		device_table_[dev->GetName()]=dev;

		return true;
	}
        else
        if(dev_id == std::string("cpu.rk3399.a53.all"))
        {
		CPUDevice * dev=new A53Device();

		dev->SetName(dev_id);

		InitializeDevice(dev);

		device_table_[dev->GetName()]=dev;

                return true;

        }
        else if(dev_id == std::string("cpu.rk3399.cpu.all"))
        {
                CPUDevice * dev=new RK3399Device();

                dev->SetName(dev_id);

                InitializeDevice(dev);

                device_table_[dev->GetName()]=dev;

                return true;
        }
        else if (dev_id == "cpu.rk3399.a72.0")
        {
               CPUDevice * dev=new  SingleA72Device();
               dev->SetName(dev_id);
               InitializeDevice(dev);
               device_table_[dev->GetName()]=dev;
               return true;
        }
        else if(dev_id == "cpu.rk3399.a53.2")
        {
               CPUDevice * dev=new  SingleA53Device();
               dev->SetName(dev_id);
               InitializeDevice(dev);
               device_table_[dev->GetName()]=dev;
               return true;
        }

	return false;
}

int RK3399Driver::DestroyDevice(void)
{
	auto ir=device_table_.begin();
	auto end=device_table_.end();

	int count=0;

	while(ir!=end)
	{
		if(DestroyDevice(ir->second))
			count++;
		else
			ir++;
	}

	return count;
}

bool RK3399Driver::DestroyDevice(Device * device) 
{
	CPUDevice * cpu_dev=reinterpret_cast<CPUDevice *>(device);

	if(GetDeviceStatus(cpu_dev)!=kDevStopped)
		return false;

	ReleaseDevice(cpu_dev);

	device_table_.erase(cpu_dev->GetName());

	delete cpu_dev;

	return true;
}

int RK3399Driver::GetDeviceNum(void) 
{
	return device_table_.size();
}

Device * RK3399Driver::GetDevice(int idx) 
{
	auto ir=device_table_.begin();
	auto end=device_table_.end();

	int i;

	for(i=0;i<idx && ir!=end;i++,ir++);

	if(ir==end)
		return nullptr;

	return ir->second;
}

Device * RK3399Driver::GetDevice(const std::string& name) 
{
	if(device_table_.count(name)==0)
		return nullptr;
	return device_table_[name];
}



int RK3399Driver::GetDevIDTableSize()
{	
     return id_table_.size();
}

const dev_id_t& RK3399Driver::GetDevIDbyIdx(int idx)
{
	return id_table_[idx];
}


bool RK3399Driver::GetWorkload(Device * dev, DevWorkload& load)
{
    //TO BE IMPLEMENTED
    return false;
}

bool RK3399Driver::GetPerf(Device * dev, Subgraph * graph,int policy,GraphPerf& perf)
{
    //TO BE IMPLEMENTED
   return false;
}

float RK3399Driver::GetFops(Device * dev, Subgraph * graph) 
{
    //TO BE IMPLEMENTED
    return false;
}


} //namespace TEngine

