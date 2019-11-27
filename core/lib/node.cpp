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
#include <assert.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <string>

#include "tengine_errno.hpp"
#include "static_graph.hpp"
#include "node.hpp"
#include "tensor.hpp"
#include "logger.hpp"

namespace TEngine {

void Node::DumpNode(void)
{
    LOG_INFO() << "\nidx: " << GetNodeIndex() << " Node: " << name_ << " OP: " << GetOp()->GetName();

    if(dynamic_shape_)
        LOG_INFO() << " Dynamic";

    LOG_INFO() << "\n";

    LOG_INFO() << "\tInput: " << inputs_.size() << " Output: " << outputs_.size() << std::endl;

    LOG_INFO() << "\tInput Tensors:" << std::endl;
    for(unsigned int i = 0; i < inputs_.size(); i++)
    {
        Tensor* p = inputs_[i]->tensor;
        LOG_INFO() << "\t\t" << inputs_[i]->port_index << ": ";

        if(p->producer)
        {
            Node* parent;
            parent = p->producer->owner;

            LOG_INFO() << " from node: " << parent->GetName() << " \ttensor: ";
        }

        p->DumpTensor(LOG_INFO());

        LOG_INFO() << "\n";
    }

    LOG_INFO() << "\tOutput Tensors:" << std::endl;
    for(unsigned int i = 0; i < outputs_.size(); i++)
    {
        Tensor* p = outputs_[i]->tensor;
        LOG_INFO() << "\t\t" << outputs_[i]->port_index << ": ";
        p->DumpTensor(LOG_INFO());
        LOG_INFO() << " connects to: " << p->consumer.size() << " nodes \n";

        for(unsigned k = 0; k < p->consumer.size(); k++)
        {
            Node* child;
            child = p->consumer[k]->owner;
            LOG_INFO() << "\t\tC" << k << ": " << child->GetName() << "\n";
        }

        LOG_INFO() << "\n";
    }
}

int Node::GetParentNum(void)
{
    return GetInputNum();
}

Node* Node::GetParentNode(int idx)
{
    NodePort* port = inputs_[idx].get();
    Tensor* tensor = port->tensor;

    return tensor->producer->owner;
}

float Node::GetFops(void)
{
    std::vector<TShape> inputs;

    for(unsigned int i = 0; i < GetInputNum(); i++)
    {
        Tensor* tensor = GetInputTensor(i);
        inputs.push_back(tensor->GetShape());
    }

    std::vector<TShape> outputs;

    for(unsigned int i = 0; i < GetOutputNum(); i++)
    {
        Tensor* tensor = GetOutputTensor(i);
        outputs.push_back(tensor->GetShape());
    }

    return op_->GetFops(inputs, outputs);
}

void Node::MergeAttr(Node* orig)
{
    auto ir = orig->dict_map_.begin();
    const auto end = orig->dict_map_.end();

    while(ir != end)
    {
        dict_map_[ir->first].swap(orig->dict_map_[ir->first]);
        ir++;
    }
}

/* code for attr get/set/add */

int NodeAddParamGeneric(void* node, const char* param_name, const char* type_name, int param_size)
{
    Node* real_node = ( Node* )node;

    if(!real_node->ExistAttr(ATTR_CUSTOM_ATTR))
    {
        node_custom_attr_map_t dummy_map;
        real_node->SetAttr(ATTR_CUSTOM_ATTR, dummy_map);
    }

    node_custom_attr_map_t* attr_map = any_cast<node_custom_attr_map_t>(&real_node->GetAttr(ATTR_CUSTOM_ATTR));

    assert(attr_map->count(param_name) == 0);

    CustomNodeAttr attr_entry;

    attr_entry.type_name = type_name;
    attr_entry.attr_size = param_size;

    (*attr_map)[param_name] = attr_entry;

    return 0;
}

int NodeGetParamGeneric(void* node, const char* param_name, const char* type_name, void* param_val, int size)
{
    Node* real_node = ( Node* )node;

    Operator* op = real_node->GetOp();

    if(op->GetParamItem(param_name, type_name, param_val))
        return 0;

    /* check custom attr */
    if(!real_node->ExistAttr(ATTR_CUSTOM_ATTR))
    {
        set_tengine_errno(ENOENT);
        return -1;
    }

    node_custom_attr_map_t* attr_map = any_cast<node_custom_attr_map_t>(&real_node->GetAttr(ATTR_CUSTOM_ATTR));

    if(attr_map->count(param_name) == 0)
    {
        set_tengine_errno(ENOENT);
        return -1;
    }

    CustomNodeAttr* attr_entry = &attr_map->at(param_name);

    if((size != attr_entry->attr_size) ||
       (type_name && attr_entry->type_name && strcmp(type_name, attr_entry->type_name)))
    {
        set_tengine_errno(EINVAL);
        return -1;
    }

    void* data = attr_entry->mem.data();

    memcpy(param_val, data, size);

    return 0;
}

int NodeSetParamGeneric(void* node, const char* param_name, const char* type_name, const void* param_val, int size)
{
    Node* real_node = ( Node* )node;

    Operator* op = real_node->GetOp();

    if(op->SetParamItem(param_name, type_name, param_val))
        return 0;

    /* check custom attr */
    if(!real_node->ExistAttr(ATTR_CUSTOM_ATTR))
    {
        set_tengine_errno(ENOENT);
        return -1;
    }

    node_custom_attr_map_t* attr_map = any_cast<node_custom_attr_map_t>(&real_node->GetAttr(ATTR_CUSTOM_ATTR));

    if(attr_map->count(param_name) == 0)
    {
        set_tengine_errno(ENOENT);
        return -1;
    }

    CustomNodeAttr* attr_entry = &attr_map->at(param_name);

    if((size != attr_entry->attr_size) ||
       (type_name && attr_entry->type_name && strcmp(type_name, attr_entry->type_name)))
    {
        set_tengine_errno(EINVAL);
        return -1;
    }

    attr_entry->mem.resize(size);

    void* data = attr_entry->mem.data();

    memcpy(data, param_val, size);

    return 0;
}

}    // namespace TEngine
