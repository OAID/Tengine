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
#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <iostream>
#include <string>
#include <atomic>

#include "base_object.hpp"
#include "tensor_shape.hpp"
#include "logger.hpp"

namespace TEngine {

class Node;
struct NodePort;
struct StaticConstTensor;

struct QuantParam
{
    int zero_point;
    float scale;
    int width;

    QuantParam()
    {
        zero_point = 0;
        scale = 1;
        width = 32;
    }
};

class Tensor : public BaseObject
{
public:
    Tensor(const std::string& name)
    {
        name_ = name;
        data_type_ = TENGINE_DT_FP32;
        static_tensor_ = nullptr;
        reshaped_count_ = 0;
        producer = nullptr;
    }
    virtual ~Tensor()
    {
        FreeTensor();
    }

    void FreeTensor(void)
    {
        if(type_ == kConstTensor && ExistAttr("free_mem") && ExistAttr("mem_addr"))
        {
            void* mem = any_cast<void*>(GetAttr("mem_addr"));
            std::free(mem);

            RemoveAttr("free_mem");
            RemoveAttr("mem_addr");
        }
    }

    Tensor(const Tensor& o)
        : BaseObject(o), producer(o.producer), consumer(o.consumer), quant_param_(o.quant_param_), type_(o.type_),
          name_(o.name_), data_type_(o.data_type_), shape_(o.shape_), static_tensor_(o.static_tensor_){};

    Tensor& operator=(const Tensor& rhs) = delete;

    const std::string& GetName(void) const
    {
        return name_;
    }
    void SetName(const std::string& n)
    {
        name_ = n;
    }

    const TShape& GetShape(void) const
    {
        return shape_;
    }
    TShape& GetShape(void)
    {
        return shape_;
    }

    void Reshape(const TShape& shape);

    /* used by consumer node to signal the input tensor that the reshape event has been taken */
    void UpdateReshapeCount(void);
    bool Reshaped(void)
    {
        return reshaped_count_ > 0;
    }

    int GetDataType(void) const
    {
        return data_type_;
    }
    void SetDataType(int dtype)
    {
        data_type_ = dtype;
    }

    unsigned int GetTotalSize() const;
    void DumpTensor(std::ostream& os) const;

    int GetTypeInt(void) const
    {
        switch(type_)
        {
            case kVarTensor:
                return TENSOR_TYPE_VAR;
            case kConstTensor:
                return TENSOR_TYPE_CONST;
            case kInputTensor:
                return TENSOR_TYPE_INPUT;
            case kDepTensor:
                return TENSOR_TYPE_DEP;
            default:
                return TENSOR_TYPE_UNKNOWN;
        }
    }

    TensorType GetType(void) const
    {
        return type_;
    }
    void SetType(TensorType t)
    {
        type_ = t;
    }

    void SetType(int t)
    {
        switch(t)
        {
            case TENSOR_TYPE_VAR:
                type_ = kVarTensor;
                break;
            case TENSOR_TYPE_CONST:
                type_ = kConstTensor;
                break;
            case TENSOR_TYPE_INPUT:
                type_ = kInputTensor;
                break;
            case TENSOR_TYPE_DEP:
                type_ = kDepTensor;
                break;
            default:
                type_ = kUnknownTensor;
                break;
        }
    }

    void AddConsumer(NodePort* in_port)
    {
        consumer.push_back(in_port);
    }

    bool RemoveConsumer(NodePort* in_port)
    {
        auto ir = consumer.begin();

        while(ir != consumer.end())
        {
            if((*ir) == in_port)
            {
                consumer.erase(ir);
                return true;
            }
            ir++;
        }

        return false;
    }

    NodePort* producer;
    std::vector<NodePort*> consumer;

    Node* GetConsumerNode(int idx);

    /* note: as tensor.hpp is defined in representation level,
       so that the memory allocated is only valid for const tensor
       to hold the trained parameters
       please use get_tensor_mem()/set_tensor_mem() to get/set tensor memory
       in operator run functioins

     */

    void* GetMemAddr(void) const
    {
        if(!ExistAttr("mem_addr"))
            return nullptr;

        return any_cast<void*>(GetAttr("mem_addr"));
    }

    void SetMemAddr(void* addr)
    {
        (*this)["mem_addr"] = addr;
    }

    void FreeMem(void);
    void BindStaticTensor(StaticConstTensor*);

    std::vector<QuantParam>* GetQuantParam(void)
    {
        return &quant_param_;
    }

protected:
    std::vector<QuantParam> quant_param_;
    TensorType type_;
    std::string name_;
    int data_type_;
    TShape shape_;

    StaticConstTensor* static_tensor_;

    std::atomic<int> reshaped_count_;
};

}    // namespace TEngine

#endif
