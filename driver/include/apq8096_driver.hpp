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

#ifndef __APQ8096_DRIVER_HPP__
#define __APQ8096_DRIVER_HPP__

#include <atomic>
#include <thread>
#include <queue>
#include <condition_variable>

#include "device_driver.hpp"
#include "cpu_driver.hpp"

namespace TEngine {

class APQ8096Executor;


class APQ8096Driver : public CPUDriver {
public:
    
    APQ8096Driver();
    
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



class APQ8096Runner: public CPURunner {
public: 
    APQ8096Runner(void) {
        if(!GetPredefinedSoc("APQ8096",soc_info)) {
            throw(std::runtime_error("SoC APQ8096 is not defined yet"));
        }
    }
};


class APQ8096Device: public CPUDevice {
public:
    APQ8096Device(): CPUDevice("cpu.apq8096.kryo.all") 
    {
        for(unsigned int i=0;i<4;i++)
        {
            CPUCore * cpu=new CPUCore();

            cpu->hw_cpu_id=i;
            cpu->dev=static_cast<CPUDevice*>(this);

            cpu->cpu_type="A72";

            compute_cores.push_back(cpu);
        }

        master_cpu_idx=0;
        compute_cores[master_cpu_idx]->master=true;

        APQ8096Runner * runner=new APQ8096Runner();

        backend_runner=runner;
    }
};

class SingleKryoDevice : public CPUDevice {
public:
    SingleKryoDevice(unsigned int cluster_id): CPUDevice(cluster_id == 0 ? "cpu.apq8096.kryo.0" : "cpu.apq8096.kryo.2")
    {
        CPUCore * cpu= new CPUCore();
        cpu->hw_cpu_id=0;
        cpu->cpu_type="A72";
        cpu->dev=static_cast<CPUDevice*>(this);
        compute_cores.push_back(cpu);

        master_cpu_idx=0;
        compute_cores[master_cpu_idx]->master=true;

        APQ8096Runner * runner=new APQ8096Runner();

        std::vector<int> real_cpu={0};

        runner->SetWorkingCPU(real_cpu,0);

        backend_runner=runner;
    }
};

class KryoClusterDevice : public CPUDevice {
public:
    KryoClusterDevice(unsigned int cluster_id): CPUDevice(cluster_id == 0 ? "cpu.apq8096.c0.all" : "cpu.apq8096.c1.all")
    {
        if(cluster_id >= 2) {
            cluster_id = 1;
        }

        unsigned int start = cluster_id*2;
        unsigned int end = start+2;

        for(unsigned int i=start;i<end;i++)
        {
            CPUCore * cpu= new CPUCore();
            cpu->hw_cpu_id=i;
            cpu->cpu_type="A72";
            cpu->dev=static_cast<CPUDevice*>(this);
            compute_cores.push_back(cpu);
        }

        master_cpu_idx=0;
        compute_cores[master_cpu_idx]->master=true;

        APQ8096Runner * runner=new APQ8096Runner();

        std::vector<int> real_cpu={static_cast<int>(start), static_cast<int>(start+1)};

        runner->SetWorkingCPU(real_cpu,start);

        backend_runner=runner;
    }
};

} //namespace TEngine

#endif
