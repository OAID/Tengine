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
#include <iostream>

#include "operator.hpp"
#include "operator_manager.hpp"

namespace TEngine {

template class SpecificFactory<Operator>;
// template SpecificFactory<Operator> SpecificFactory<Operator>::instance;
template OpManager SimpleObjectManagerWithLock<OpManager, Operator*>::instance;

any OpManager::GetOpDefParam(const std::string& op_name)
{
    Operator* op;

    if(SafeGet(op_name, op))
        return op->GetDefParam();
    else
        return any();
}

bool OpManager::AddOpAttr(const std::string& op_name, const std::string& attr_name, const any& val)
{
    Operator* op;

    if(!SafeGet(op_name, op))
        return false;

    op->SetAttr(attr_name, val);

    return true;
}

bool OpManager::RemoveOpAttr(const std::string& op_name, const std::string& attr_name)
{
    Operator* op;

    if(!SafeGet(op_name, op))
        return false;

    if(!op->ExistAttr(attr_name))
        return false;

    op->RemoveAttr(attr_name);

    return true;
}

bool OpManager::GetOpAttr(const std::string& op_name, const std::string& attr_name, any& val)
{
    Operator* op;

    if(!SafeGet(op_name, op))
        return false;

    if(!op->ExistAttr(attr_name))
        return false;

    val = op->GetAttr(attr_name);

    return true;
}

Operator* OpManager::CreateOp(const std::string& op_name)
{
    Operator* op;

    if(!SafeGet(op_name, op))
        return nullptr;

    return op->Clone();
}

}    // namespace TEngine
