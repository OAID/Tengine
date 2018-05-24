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
 * Copyright (c) 2018, Linaro Ltd
 * Author: Manivannan Sadhasivam <manivannan.sadhasivam@linaro.org>
 */

#include <sched.h>
#include <chrono>
#include <functional>

#include "tensor_mem.hpp"
#include "hikey960_driver.hpp"
#include "hikey960_executor.hpp"
#include "logger.hpp"


namespace TEngine {


HIKEY960Driver::HIKEY960Driver(void)
{
	SetName("HIKEY960");

	id_table_.push_back("cpu.hikey960.a53.all");
	id_table_.push_back("cpu.hikey960.a73.all");
	id_table_.push_back("cpu.hikey960.cpu.all");
	id_table_.push_back("cpu.hikey960.a73.0");
	id_table_.push_back("cpu.hikey960.a53.2");

}


int HIKEY960Driver::ProbeDevice(void) 
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

bool HIKEY960Driver::ProbeDevice(const dev_id_t& dev_id)
{
	if(dev_id==std::string("cpu.hikey960.a73.all"))
	{
		CPUDevice * dev=new A73Device();

		dev->SetName(dev_id);

		InitializeDevice(dev);

		device_table_[dev->GetName()]=dev;

		return true;
	}
        else
        if(dev_id == std::string("cpu.hikey960.a53.all"))
        {
		CPUDevice * dev=new HKA53Device();

		dev->SetName(dev_id);

		InitializeDevice(dev);

		device_table_[dev->GetName()]=dev;

                return true;

        }
        else if(dev_id == std::string("cpu.hikey960.cpu.all"))
        {
                CPUDevice * dev=new HIKEY960Device();

                dev->SetName(dev_id);

                InitializeDevice(dev);

                device_table_[dev->GetName()]=dev;

                return true;
        }
        else if (dev_id == "cpu.hikey960.a73.0")
        {
               CPUDevice * dev=new  SingleA73Device();
               dev->SetName(dev_id);
               InitializeDevice(dev);
               device_table_[dev->GetName()]=dev;
               return true;
        }
        else if(dev_id == "cpu.hikey960.a53.2")
        {
               CPUDevice * dev=new  HKSingleA53Device();
               dev->SetName(dev_id);
               InitializeDevice(dev);
               device_table_[dev->GetName()]=dev;
               return true;
        }
	return false;
}

int HIKEY960Driver::DestroyDevice(void)
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

bool HIKEY960Driver::DestroyDevice(Device * device) 
{
	CPUDevice * cpu_dev=reinterpret_cast<CPUDevice *>(device);

	if(GetDeviceStatus(cpu_dev)!=kDevStopped)
		return false;

	ReleaseDevice(cpu_dev);

	device_table_.erase(cpu_dev->GetName());

	delete cpu_dev;

	return true;
}

int HIKEY960Driver::GetDeviceNum(void) 
{
	return device_table_.size();
}

Device * HIKEY960Driver::GetDevice(int idx) 
{
	auto ir=device_table_.begin();
	auto end=device_table_.end();

	int i;

	for(i=0;i<idx && ir!=end;i++,ir++);

	if(ir==end)
		return nullptr;

	return ir->second;
}

Device * HIKEY960Driver::GetDevice(const std::string& name) 
{
	if(device_table_.count(name)==0)
		return nullptr;
	return device_table_[name];
}



int HIKEY960Driver::GetDevIDTableSize()
{	
     return id_table_.size();
}

const dev_id_t& HIKEY960Driver::GetDevIDbyIdx(int idx)
{
	return id_table_[idx];
}


bool HIKEY960Driver::GetWorkload(Device * dev, DevWorkload& load)
{
    //TO BE IMPLEMENTED
    return false;
}

bool HIKEY960Driver::GetPerf(Device * dev, Subgraph * graph,int policy,GraphPerf& perf)
{
    //TO BE IMPLEMENTED
   return false;
}

float HIKEY960Driver::GetFops(Device * dev, Subgraph * graph) 
{
    //TO BE IMPLEMENTED
    return false;
}


} //namespace TEngine

