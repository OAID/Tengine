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
#ifndef __SIMPLE_OBJECT_MANAGER_HPP__
#define __SIMPLE_OBJECT_MANAGER_HPP__

#include <string>
#include <unordered_map>

namespace TEngine {

template <typename M, typename T> 
class SimpleObjectManager: public std::unordered_map<std::string, T >  
{
public:
    static M * GetInstance(void)
    {
        static M instance;
        return &instance;
    }

    static bool Find(const std::string& name)
    {
        auto manager=GetInstance();

        if(manager->count(name)==0)
             return false;

         return true;
    }

    static T& Get(const std::string& name)
    {
        auto manager=GetInstance();

        return (*manager)[name];
    }

    static bool Get(const std::string& name, T& val)
    {
        auto manager=GetInstance();

        if(manager->count(name))
        {
           val=manager->at(name);
           return true;
        }
        
        return false;
    }

    static bool Add(const std::string& name, const T& data)
    {
        auto manager=GetInstance();

        if(manager->count(name))
               return false;

         (*manager)[name]=data;

        return true;
    }


    static bool Replace(const std::string& name, const T& data)
    {
        auto manager=GetInstance();

        if(manager->count(name)==0)
               return false;
         
        T& old_data=(*manager)[name];
        FreeObject(old_data);
        
        (*manager)[name]=data;

        return true;
    }

    static bool   Remove(const std::string& name)
    {
       auto manager=GetInstance();
       auto ir=(*manager).begin();
       auto end=(*manager).end();

       while(ir!=end)
       {
           if(ir->first == name)
           {
               FreeObject(ir->second);
               manager->erase(ir);
               return true;
           }

           ir++;
       }
     
       return false;  
    }

    template <typename U>
    static typename std::enable_if<std::is_pointer<U>::value,void>::type 
    FreeObject(U& obj)
    {
        delete obj;
    }
    
    template <typename U>
    static typename std::enable_if<!std::is_pointer<U>::value,void>::type 
    FreeObject(U& obj)
    {
    }


    virtual ~SimpleObjectManager() 
    {
       auto manager=GetInstance();
       auto ir=(*manager).begin();
       auto end=(*manager).end();

       while(ir!=end)
       {
        FreeObject(ir->second);  
        ir++;
       }
    }
};




} //namespace TEngine

#endif
