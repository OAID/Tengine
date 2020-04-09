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
#ifndef __TENSOR_MEM_HPP__
#define __TENSOR_MEM_HPP__

#include <memory>
#include <functional>

namespace TEngine {

using mem_release_t = std::function<void(void*)>;

class Tensor;

class TensorMem
{
public:
    TensorMem()
    {
        mem_addr_ = nullptr;
        mem_size_ = 0;
        releaser_ = nullptr;
    }
    ~TensorMem()
    {
        if(mem_addr_)
        {
            if(releaser_)
                releaser_(mem_addr_);
        }
    }

    void SetMem(void* addr, int size, mem_release_t releaser)
    {
        mem_addr_ = addr;
        mem_size_ = size;
        releaser_ = releaser;
    }

    int GetSize(void)
    {
        return mem_size_;
    }
    void* GetMem(void)
    {
        return mem_addr_;
    }

private:
    void* mem_addr_;
    int mem_size_;
    mem_release_t releaser_;
};

using TensorMemPtr = std::shared_ptr<TensorMem>;

bool get_tensor_memptr(const Tensor*, TensorMemPtr&);
void set_tensor_mem(Tensor*, const TensorMemPtr&);

void* get_tensor_mem(const Tensor*);
int get_tensor_mem_size(const Tensor*);
bool set_tensor_mem(Tensor*, void*, int, mem_release_t);
void free_tensor_mem(Tensor*);

}    // namespace TEngine

#endif
