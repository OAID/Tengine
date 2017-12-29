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
#ifndef __VARARG_HPP__
#define __VARARG_HPP__


/*provide utilities to print var args*/

#include <iostream>
#include <typeinfo>

#include "type_name.hpp"

static inline void VarArgPrint(){}

template<typename T, typename ... Args>
void VarArgPrint(std::ostream& os, T first,Args...args)
{
     os<<first<<std::endl;

     VarArgPrint(os,args...);
}    


template<int i>
void ArgPrint(std::ostream& os) {}

template<int i, typename T, typename ... Args>
void ArgPrint(std::ostream& os, T first, Args ... args)
{
    os<<i<<": "<<TEngine::GetTypeName(typeid(first).name())<<" "<<first<<"\n";

    ArgPrint<i+1,Args...>(os,args...);
}


#endif
