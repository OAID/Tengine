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
#ifndef __OP_IMPLEMENTOR_HPP__
#define __OP_IMPLEMENTOR_HPP__

#include <map>
#include <memory>
#include <vector>

namespace TEngine {


class Node;
class TEngine;

template <typename ... Args>
struct OpImplementor {
    virtual bool Prerun(Node * node, ExecEngine * engine)=0;
    virtual bool Run(Node * node, ExecEngine * engine)=0;
    virtual bool Postrun(Node * node, ExecEngine * engine)=0;
    virtual bool Support(Args ... args)=0;

    virtual ~ OpImplementor() {}

    std::string name;

    int priority;

    using Ptr=std::shared_ptr<OpImplementor>;

};


template < typename T, typename ... Args>
class ImplementorManager{

public:
   static ImplementorManager * GetInstance(void)
   {
        static ImplementorManager * instance_ptr=new ImplementorManager();

        return instance_ptr;
   }
   static T * Select(Args ... args) 
   {
        ImplementorManager * manager=GetInstance();
        return manager->Find(std::forward<Args>(args)...);
   }

   static void Register(T * impl)
   {
        ImplementorManager * manager=GetInstance();

        typename T::Ptr ptr(impl);

        manager->impl_map_[impl->priority]=ptr;
   }

   T * Find(Args ... args)
   {
        auto begin=impl_map_.begin();
        auto end=impl_map_.end();

        for(auto it=begin;it!=end;it++)
        {
            auto op_impl=it->second;
            if(op_impl->Support(std::forward<Args>(args)...))
                 return dynamic_cast<T*>(op_impl.get());
        }

        return nullptr;
   }

   std::vector<T*> List(void)
   {
        ImplementorManager * manager=GetInstance();
        std::vector<T*> result;
          
        auto begin=impl_map_.begin();
        auto end=impl_map_.end();

        for(auto it=begin;it!=end;it++)
        {
            auto op_impl=it->second;

            result.push_back(op_impl);
        }

        return result;
   }

protected:
   std::map<int,typename T::Ptr> impl_map_;

};




};


#endif
