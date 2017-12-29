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
#ifndef __GENERIC_FACTORY__
#define __GENERIC_FACTORY__

#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <typeindex>
#include <type_traits>

#include "any.hpp"
#include "type_name.hpp"
#include "logger.hpp"

namespace TEngine {


class BaseFactory
{
public:

   using creator_map_t=std::unordered_map<std::string, any>;
   using creator_type_t=std::unordered_multimap<std::string,std::string>; //saved the creator type info

   /* only accept funtion call */
   template <typename  func>
   typename std::enable_if<std::is_function<typename std::remove_pointer<func>::type>::value,bool>::type
   RegisterCreator(const std::string&name, func creator, bool replace=false)
   {
       using func_t=typename std::remove_reference<decltype(*creator)>::type;

       return RegisterCreator(name,std::function<func_t>(creator),replace);

   }

   template <class T, typename ... Args>
   bool RegisterCreator(const std::string&name, std::function<T *(Args ...)> creator, bool replace=false)
   {
         const std::string key=name+typeid(creator).name();

          LOG_DEBUG()<<"Register: name "<<name<<"\n";

          LOG_DEBUG()<<"Function: "<<GetTypeName(typeid(creator).name())<<"\n";

         if(RegisterCreator(key,any(creator),replace))
         {
             type_map_.emplace(name,GetTypeName(typeid(creator).name()));
             return true;
         }
         else
         {
             return false;
         }
   }

    template <class T, typename ... Args>
    T * Create(const std::string& name, Args&& ... args)
    {

         using func_t=std::function<T*(Args...)>;

         std::string key=name+typeid(func_t).name();

         auto it=creator_map_.find(key);

         if(it==creator_map_.end())
          {
                LOG_ERROR()<<"failed to find "<<name<< " with func type: "<<GetTypeName(typeid(func_t).name())<<"\n";
                return nullptr;
          }

         func_t creator=any_cast<func_t>(it->second);

         return creator(std::forward<Args>(args)...);
   }

   std::vector<std::string> CreatorInfo(const std::string& name)
   {
         std::vector<std::string> result;

         auto range=type_map_.equal_range(name);

         for(auto ir=range.first;ir!=range.second;ir++)
         {
             result.emplace_back(ir->second);
         }

         return result;
   }

protected:
   bool RegisterCreator(const std::string& key, any creator, bool replace=false)
   {
       if(creator_map_.count(key) && !replace)
       {
           return false;
       }
        
       creator_map_.emplace(key,creator);

       return true;
   }

    creator_map_t creator_map_;
    creator_type_t type_map_;

};


class GenericFactory: public BaseFactory {

public:

    template <class T, typename ... Args>
    bool RegisterConstructor(const std::string& name, bool replace=false)
    {
         std::function<T*(Args ...)> creator=[](Args...args){ return new T(std::forward<Args>(args)...); };
         return RegisterCreator(name,creator,replace);
    }

    template <class T, typename Derived, typename ... Args>
    typename std::enable_if<std::is_base_of<T,Derived>::value, bool>::type
    
    RegisterInterface(const std::string&name, bool replace=false)
    {
        std::function<T*(Args ...)> creator=[](Args ... args)
        {  return new Derived(std::forward<Args>(args)...); };

        return  RegisterCreator(name,creator,replace);
    }
   
};


template <typename T>

class SpecificFactory: public GenericFactory
{

public:

   template<typename ... Args>
   T * Create(const std::string& name, Args&& ... args)
   {

        using func_t=std::function<T*(Args...)>;

        LOG_DEBUG()<<"Create Function: "<<GetTypeName(typeid(func_t).name())<<"\n";
        
        return GenericFactory::Create<T,Args...>(name,std::forward<Args>(args) ...);

   }
   
  template <typename ... Args>
  bool RegisterConstructor(const std::string& name, bool replace=false)
  {
        return GenericFactory::RegisterConstructor<T,Args...>(name,replace);
  }

  template <typename Derived, typename ... Args>
  typename std::enable_if<std::is_base_of<T,Derived>::value, bool>::type

  RegisterInterface(const std::string&name, bool replace=false)
  {
        return  GenericFactory::RegisterInterface<T,Derived,Args...>(name,replace);
  }

  static SpecificFactory<T> * GetFactory()
  {
       static SpecificFactory<T> instance;

       return &instance;
  }

};


} //namespace TEngine

#endif
