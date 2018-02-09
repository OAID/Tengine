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

#include <string>

#include "logger.hpp"
#include "device_driver.hpp"
#include "dev_executor.hpp"


namespace TEngine {

bool DriverManager::RegisterDriver(const std::string& name, Driver * driver)
{
      return SafeAdd(name,driver);
}

bool DriverManager::UnregisterDriver(const std::string& name)
{
       Driver * driver;
	   
	if(!SafeGet(name,driver))
		return false;

	return UnregisterDriver(driver);
	
}

bool DriverManager::UnregisterDriver(Driver * driver)
{
	if(driver->GetDeviceNum())
		return false;

	return SafeRemove(driver->GetName());
}

Driver * DriverManager::GetDriver(const std::string& name)
{
       Driver * driver;
	   
	if(!SafeGet(name,driver))
	         return nullptr;

	return driver;
}

Device * DriverManager::RealGetDevice(const dev_id_t& dev_id)
{
     auto ir=begin();
     auto ir_end=end();

     while(ir!=ir_end)
     {
        Driver * driver=ir->second;

        int dev_num=driver->GetDeviceNum();

        for(int i=0;i<dev_num;i++)
        {
            Device * dev=driver->GetDevice(i);

            if(dev->GetDeviceID()==dev_id)
                return dev;
        }

        ir++;
     }

     return nullptr;
}

Device * DriverManager::GetDevice(const dev_id_t& dev_id)
{
      auto manager=GetInstance();

      return manager->RealGetDevice(dev_id);
}

bool DriverManager::LoadDevice(Driver * driver,Device * device)
{
	const dev_id_t & dev_id=device->GetDeviceID();

	DevExecutor * dev_executor=DevExecutorFactory::GetFactory()->Create(dev_id,dev_id);

        if(dev_executor)
        {
	    dev_executor->SetName(device->GetName());
            dev_executor->Init();
		
	    dev_executor->BindDevice(device);

	    DevExecutorManager::RegisterDevExecutor(dev_executor);

	    device->Start();
	    dev_executor->Start();
        }
        else
        {
	    device->Start();
        }
	
        return true;
}

bool DriverManager::UnloadDevice(Driver * driver,Device * device)
{
        const dev_id_t & dev_id=device->GetDeviceID();
        DevExecutor * dev_executor=nullptr;

        DevExecutorManager::SafeGet(dev_id,dev_executor);

        if(dev_executor)
        {

	   if(!dev_executor->Stop() || !device->Stop())
		return false;

	   DevExecutorManager::UnregisterDevExecutor(dev_executor);
	
           dev_executor->UnbindDevice();
	   dev_executor->Release();
	
	   delete dev_executor;
       }
       else
       {
           if(!device->Stop())
              return false;
       }

	driver->DestroyDevice(device);

	return true;
	
}


bool DriverManager::LoadDevice(Driver * driver)
{
	int dev_num=driver->ProbeDevice();
	
	if(dev_num==0)
		return false;

	for(int i=0;i<dev_num;i++)
	{
	       Device * device=driver->GetDevice(i);
		
		if(!LoadDevice(driver,device))
			return false;			
	}

	return true;
}


bool DriverManager::UnloadDevice(Driver * driver)
{
	int dev_num=driver->GetDeviceNum();
	
	if(dev_num==0)
		return true;

	for(int i=0;i<dev_num;i++)
	{
	       Device * device=driver->GetDevice(0);
		
		if(!UnloadDevice(driver,device))
			return false;			
	}

	return true;	
}

void DriverManager::ProbeDevice(void)
{

   /* Load pre-assigned device  */
   int count=0;
   std::string prefix="driver.probe.";

   while(true)
   {
        std::string key_name=prefix+std::to_string(count);
        std::string driver_name;

        if(!TEngineConfig::Get(key_name,driver_name))
             break;

        count++;

        Driver * driver=GetDriver(driver_name);

        if(driver)
        {
             LoadDevice(driver);
             int dev_num=driver->GetDeviceNum();

             LOG_INFO()<<"Driver: "<<driver_name<<" probed "<<dev_num<<" devices\n";
        }
   }
}

void DriverManager::ReleaseDevice(void)
{
    //loop over all registered driver, and unload the device the driver probed
    auto manager=GetInstance();

    auto ir=manager->begin();
    auto ir_end=manager->end();

    while(ir!= ir_end)
    {
         Driver * driver=ir->second;

         UnloadDevice(driver);         

         ir++;    
    }

}


} //namespace TEngine
