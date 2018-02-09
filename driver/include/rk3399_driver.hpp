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

#ifndef __RK3399_DRIVER_HPP__
#define __RK3399_DRIVER_HPP__

#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>

#include "device_driver.hpp"
#include "cpu_driver.hpp"

namespace TEngine {

class RK3399Executor;


class RK3399Driver : public CPUDriver {
public:
	
	RK3399Driver();
	
	int ProbeDevice(void) override;
	bool ProbeDevice(const dev_id_t& dev_id) override;
	
	int DestroyDevice(void) override;
	bool DestroyDevice(Device * device) override;

	int GetDeviceNum(void) override;
	Device * GetDevice(int idx) override;
        Device * GetDevice(const std::string& name) override;

       bool GetWorkload(Device * dev, DevWorkload& load) override;
       bool GetPerf(Device * dev, Subgraph * graph,int policy,GraphPerf& perf) override;
       float GetFops(Device * dev, Subgraph * graph) override;

	int GetDevIDTableSize();
	const dev_id_t& GetDevIDbyIdx(int idx);

protected:

	std::vector<dev_id_t> id_table_;
        std::unordered_map<std::string, Device *> device_table_;	
};



class RK3399Runner: public CPURunner {
public: 
      RK3399Runner(void) {
         if(!GetPredefinedSoc("RK3399",soc_info))
         {
            throw(std::runtime_error("SOC RK3399 is not defined yet"));
         }
      }

};


class A72Device : public CPUDevice {
public:
	A72Device(): CPUDevice("cpu.rk3399.a72.all")
	{
	      
	      for(unsigned int i=4;i<6;i++)
	      {
          	       CPUCore * cpu= new CPUCore();
	      		cpu->hw_cpu_id=i;
			cpu->cpu_type="A72";
			cpu->dev=static_cast<CPUDevice*>(this);
			compute_cores.push_back(cpu);
	      }


              master_cpu_idx=1;
              compute_cores[master_cpu_idx]->master=true;

              RK3399Runner * runner=new RK3399Runner();

              std::vector<int> real_cpu={4,5};

              runner->SetWorkingCPU(real_cpu,5);

              backend_runner=runner;
	}
       
};

class A53Device: public CPUDevice {
public:
	A53Device(): CPUDevice("cpu.rk3399.a53.all") 
	{
	      for(unsigned int i=0;i<4;i++)
	      {
	                CPUCore * cpu=new CPUCore();
	      		cpu->hw_cpu_id=i;
			cpu->cpu_type="A53";
			cpu->dev=static_cast<CPUDevice*>(this);
			compute_cores.push_back(cpu);
	      }

	      compute_cores[0]->master=true;
              master_cpu_idx=0;

              RK3399Runner * runner=new RK3399Runner();

              std::vector<int> real_cpu={0,1,2,3};

              runner->SetWorkingCPU(real_cpu,0);

              backend_runner=runner;
       
	}
	
};


class RK3399Device: public CPUDevice {
public:
	RK3399Device(): CPUDevice("cpu.rk3399.cpu.all") 
       {
	      for(unsigned int i=0;i<6;i++)
	      {
	              CPUCore * cpu=new CPUCore();
	     
	      		cpu->hw_cpu_id=i;
			cpu->dev=static_cast<CPUDevice*>(this);
			
			if(i<4)
				cpu->cpu_type="A53";
			else
				cpu->cpu_type="A72";
	       
        	       compute_cores.push_back(cpu);
	     }

	    compute_cores[4]->master=true;
            master_cpu_idx=4;

            RK3399Runner * runner=new RK3399Runner();

            backend_runner=runner;

	}


};

class SingleA72Device : public CPUDevice {
public:
        SingleA72Device(): CPUDevice("cpu.rk3399.a72.0")
        {

              for(unsigned int i=4;i<5;i++)
              {
                       CPUCore * cpu= new CPUCore();
                        cpu->hw_cpu_id=i;
                        cpu->cpu_type="A72";
                        cpu->dev=static_cast<CPUDevice*>(this);
                        compute_cores.push_back(cpu);
              }


              master_cpu_idx=0;
              compute_cores[master_cpu_idx]->master=true;

              RK3399Runner * runner=new RK3399Runner();

              std::vector<int> real_cpu={4};

              runner->SetWorkingCPU(real_cpu,4);

              backend_runner=runner;
        }

};

class SingleA53Device : public CPUDevice {
public:
        SingleA53Device(): CPUDevice("cpu.rk3399.a53.2")
        {

              for(unsigned int i=2;i<3;i++)
              {
                       CPUCore * cpu= new CPUCore();
                        cpu->hw_cpu_id=i;
                        cpu->cpu_type="A53";
                        cpu->dev=static_cast<CPUDevice*>(this);
                        compute_cores.push_back(cpu);
              }


              master_cpu_idx=0;
              compute_cores[master_cpu_idx]->master=true;

              RK3399Runner * runner=new RK3399Runner();

              std::vector<int> real_cpu={2};

              runner->SetWorkingCPU(real_cpu,2);

              backend_runner=runner;
        }

};




} //namespace TEngine



#endif
