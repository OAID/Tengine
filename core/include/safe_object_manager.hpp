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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __SAFE_OBJECT_MANAGER_HPP__
#define __SAFE_OBJECT_MANAGER_HPP__

#include <mutex>
#include "tengine_config.hpp"
#include "tengine_lock.hpp"

#include "simple_object_manager.hpp"

namespace TEngine {

template <typename M, typename T> 
class SimpleObjectManagerWithLock: public  SimpleObjectManager<M,T> 
{
public:
    static bool SafeFind(const std::string& name)
    {
        auto manager=SimpleObjectManager<M,T>::GetInstance();

        manager->Lock();

        bool ret=true;

        if(manager->count(name)==0)
             ret=false;

         manager->Unlock();

         return ret;
    }

    static bool SafeGet(const std::string& name, T& val)
    {
        auto manager=SimpleObjectManager<M,T>::GetInstance();
        bool ret=true;

        manager->Lock();

        if(manager->count(name))
           val=(*manager)[name];
        else
            ret=false;

        manager->Unlock();

         return ret;
    }

    static bool SafeAdd(const std::string& name, const T& data)
    {
        auto manager=SimpleObjectManager<M,T>::GetInstance();

        manager->Lock();

        if(manager->count(name))
        {
             manager->Unlock();
             return false;
        }

        (*manager)[name]=data;

        manager->Unlock();

        return true;
    }


    static bool SafeReplace(const std::string& name, const T& data)
    {
        auto manager=SimpleObjectManager<M,T>::GetInstance();

        manager->Lock();

        if(manager->count(name)==0)
        {
             manager->Unlock();

             return false;
        }
         
        T& old_data=(*manager)[name];
        FreeObject(old_data);
        
        (*manager)[name]=data;

        manager->Unlock();

        return true;
    }

    static bool  SafeRemove(const std::string& name)
    {
       auto manager=SimpleObjectManager<M,T>::GetInstance();

       manager->Lock();

       auto ir=(*manager).begin();
       auto end=(*manager).end();

       while(ir!=end)
       {
           if(ir->first == name)
           {
               SimpleObjectManager<M,T>::FreeObject(ir->second);
               manager->erase(ir);

               manager->Unlock();
               return true;
           }

           ir++;
       }
     
       manager->Unlock();
       return false;  
    }

    static bool  SafeRemoveOnly(const std::string& name)
    {
       auto manager=SimpleObjectManager<M,T>::GetInstance();

       manager->Lock();

       auto ir=(*manager).begin();
       auto end=(*manager).end();

       while(ir!=end)
       {
           if(ir->first == name)
           {
               manager->erase(ir);
               manager->Unlock();
               return true;
           }

           ir++;
       }

       manager->Unlock();
       return false;

    }

    ~SimpleObjectManagerWithLock() 
    {
    }


   void Lock(void)
   {
       TEngineLock(my_mutex);
   }

   void Unlock(void)
   {
       TEngineUnlock(my_mutex);
   }

private:
   std::mutex my_mutex;
    
};




} //namespace TEngine

#endif
