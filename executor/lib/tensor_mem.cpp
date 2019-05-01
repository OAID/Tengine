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
#include "graph.hpp"

#include "tensor_mem.hpp"

namespace TEngine {

void* get_tensor_mem(const Tensor* tensor)
{
    if(tensor->GetType() == kConstTensor)
        return tensor->GetMemAddr();

    TensorMemPtr ptr;

    if(get_tensor_memptr(tensor, ptr) && ptr.get() != nullptr)
        return ptr.get()->GetMem();

    return nullptr;
}

int get_tensor_mem_size(const Tensor* tensor)
{
    if(tensor->GetType() == kConstTensor)
        return tensor->GetTotalSize();

    TensorMemPtr ptr;

    if(get_tensor_memptr(tensor, ptr) && ptr.get() != nullptr)
        return ptr.get()->GetSize();
    else
        return 0;
}

bool get_tensor_memptr(const Tensor* tensor, TensorMemPtr& ptr)
{
    if(tensor->ExistAttr("tensor_mem"))
    {
        ptr = any_cast<TensorMemPtr>(tensor->GetAttr("tensor_mem"));
        return true;
    }

    return false;
}

bool set_tensor_mem(Tensor* tensor, void* addr, int size, mem_release_t releaser)
{
    if(addr == nullptr || size == 0)
        return false;
    if(tensor->GetType() == kConstTensor)
    {
        LOG_DEBUG() << __FUNCTION__ << ": set const tensor " << tensor->GetName() << " mem: " << addr << "\n";

        tensor->FreeTensor();
        tensor->SetMemAddr(addr);

        if(releaser)
        {
            tensor->SetAttr("free_mem", 1);
        }

        return true;
    }

    TensorMemPtr ptr(new TensorMem());

    ptr.get()->SetMem(addr, size, releaser);

    set_tensor_mem(tensor, ptr);

    return true;
}

void set_tensor_mem(Tensor* tensor, const TensorMemPtr& ptr)
{
    tensor->SetAttr("tensor_mem", ptr);
}

void free_tensor_mem(Tensor* tensor)
{
    if(tensor->GetType() == kConstTensor)
        return;

    if(tensor->ExistAttr("tensor_mem"))
        tensor->RemoveAttr("tensor_mem");
}

}    // namespace TEngine
