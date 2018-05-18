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
#include <iostream>
#include <functional>

#include "logger.hpp"
#include "rk3399_driver.hpp"
#include "rk3399_executor.hpp"
#include "hikey960_driver.hpp"
#include "hikey960_executor.hpp"

extern "C" {
    int tengine_plugin_init(void);
}

using namespace TEngine;

int tengine_plugin_init(void)
{

    RK3399Driver * rk3399=new RK3399Driver();
    HIKEY960Driver * hikey960=new HIKEY960Driver();
    DriverManager::RegisterDriver(rk3399->GetName(),rk3399);
    DriverManager::RegisterDriver(hikey960->GetName(),hikey960);

    //Executor Factory registration
    auto dev_executor_factory=DevExecutorFactory::GetFactory();

    //for each dev_id in driver rk3399, regiser one executor 
    int n=rk3399->GetDevIDTableSize();

    for(int i=0;i<n;i++) {
         dev_executor_factory->
                RegisterInterface<RK3399Executor,const dev_id_t&>(rk3399->GetDevIDbyIdx(i));
    }

    //for each dev_id in driver hikey960, regiser one executor 
    n=hikey960->GetDevIDTableSize();
    
    for(int i=0;i<n;i++) {
         dev_executor_factory->
                RegisterInterface<HIKEY960Executor,const dev_id_t&>(hikey960->GetDevIDbyIdx(i));
    }
   
    std::cout<<"DEV ENGINE PLUGIN INITED\n";

    return 0;
}
