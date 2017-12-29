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
#ifndef __OPERATOR_MANAGER_HPP__
#define __OPERATOR_MANAGER_HPP__

#include <string>

#include "safe_object_manager.hpp"
#include "attribute.hpp"

namespace TEngine {

class Operator;

class OpManager: public SimpleObjectManagerWithLock<OpManager, Operator *> 
{

public: 
    static any GetOpDefParam(const std::string& op_name);

    static bool AddOpAttr(const std::string& op_name, const std::string& attr_name, const any& val);
    static bool RemoveOpAttr(const std::string& op_name, const std::string& attr_name);
    static bool GetOpAttr(const std::string& op_name,const std::string& attr_name, any& val);

    static Operator * CreateOp(const std::string& op_name);

};


} //namespace

#endif
