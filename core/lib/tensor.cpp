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
#include <string>

#include "data_type.hpp"
#include "tensor.hpp"
#include "static_graph.hpp"
#include "node.hpp"

namespace TEngine {

void Tensor::Reshape(const TShape& shape)
{
    if(shape_ == shape)
        return;

    reshaped_count_ = consumer.size();

    shape_ = shape;
}

void Tensor::UpdateReshapeCount(void)
{
    reshaped_count_--;

    if(reshaped_count_ < 0)
    {
        LOG_ERROR() << "tensor: " << name_ << " has bad reshaped_count: " << ( int )reshaped_count_ << "\n";
    }
}

unsigned int Tensor::GetTotalSize(void) const
{
    unsigned int elem_size = DataType::GetTypeSize(data_type_);
    unsigned int elem_num = shape_.GetSize();

    return elem_size * elem_num;
}

Node* Tensor::GetConsumerNode(int idx)
{
    NodePort* port = consumer[idx];
    return port->owner;
}

static std::string MapTypeToString(TensorType type)
{
    if(type == kVarTensor)
        return "Var";
    else if(type == kConstTensor)
        return "Const";
    else if(type == kInputTensor)
        return "Input";
    else
        return "Unknown";
}

void Tensor::DumpTensor(std::ostream& os) const
{
    os << name_ << " type: " << MapTypeToString(type_) << " datatype: " << DataType::GetTypeName(data_type_)
       << " Shape: ";
    shape_.DumpShape(os);
}

void Tensor::FreeMem(void)
{
    FreeTensor();

    if(static_tensor_)
    {
        if(static_tensor_->mem_addr)
            std::free(static_tensor_->mem_addr);

        static_tensor_->mem_addr = nullptr;
        static_tensor_ = nullptr;
    }
}

void Tensor::BindStaticTensor(StaticConstTensor* static_tensor)
{
    static_tensor_ = static_tensor;
}

}    // namespace TEngine
