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
#ifndef __PARAMETER_HPP__
#define __PARAMETER_HPP__


#include <functional>

#include "base_object.hpp"

namespace TEngine {

  
using entry_parser_t=std::function<void(BaseObject&)>;

template <typename T>
bool ConvertSpecialAny(T& entry, const std::type_info & info, any& data);


#define DECLARE_PARSER_STRUCTURE(param) \
        static void Parse(param& param_obj, BaseObject * p_obj)\


#define DECLARE_PARSER_ENTRY(entry) \
               {\
                   typedef decltype(param_obj.entry) type0;\
                   any&    content=(*p_obj)[#entry];\
                   if(typeid(type0)== content.type()) \
			param_obj.entry=any_cast<type0>(content);	\
                   else\
                   {\
                     if(!ConvertSpecialAny(param_obj.entry,content.type(),content))\
                        std::cerr<<"cannot parser entry: "<<#entry<<std::endl;\
                   }\
               }

#define DECLARE_CUSTOM_PARSER_ENTRY(entry) \
               {\
                   typedef decltype(param_obj.entry) type0;\
                   any&    content=(*p_obj)[#entry];\
                   if(typeid(type0)== content.type()) \
			param_obj.entry=any_cast<type0>(content);	\
                   else\
                   {\
                       std::cerr<<"cannot parser entry: "<<#entry<<std::endl;\
                   }\
               }



} //namespace TEngine

#endif
