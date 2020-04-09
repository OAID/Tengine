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
#include <functional>
#include <algorithm>

#include "static_graph.hpp"
#include "static_graph_interface.hpp"
#include "logger.hpp"

namespace TEngine {

StaticGraph::~StaticGraph(void)
{
    for(auto p : mem_src)
        free(p);

    if(release_func)
        release_func(dev_handle);
}

StaticGraph* CreateStaticGraph(const std::string& name)
{
    StaticGraph* graph = new StaticGraph();

    SetGraphInternalName(graph, name);

    return graph;
}

void DestroyStaticGraph(StaticGraph* graph)
{
    delete graph;
}

const void* GetGraphContext(StaticGraph* graph)
{
    return graph->exec_context;
}

void SetGraphDevHandle(StaticGraph* graph, void* release_func, void* dev_handle)
{
    typedef void (*release_t)(void*);

    graph->release_func = ( release_t )release_func;
    graph->dev_handle = dev_handle;
}

void SetGraphLayout(StaticGraph* graph, int layout)
{
    graph->graph_layout = layout;
}

void SetModelLayout(StaticGraph* graph, int layout)
{
    graph->model_layout = layout;
}

void SetModelFormat(StaticGraph* graph, int model_format)
{
    graph->model_format = model_format;
}

void SetModelSubFormat(StaticGraph* graph, int model_subformat)
{
    graph->model_subformat = model_subformat;
}

void SetGraphInternalName(StaticGraph* graph, const std::string& name)
{
    graph->model_name = name;
}

void SetGraphIdentity(StaticGraph* graph, const std::string& domain, const std::string& name,
                      const std::string& version)
{
    graph->domain = domain;
    graph->name = name;
    graph->version = version;
}

void SetGraphSource(StaticGraph* graph, const std::string& source)
{
    graph->source = source;
}

void SetGraphSourceFormat(StaticGraph* graph, const std::string& format)
{
    graph->source_format = format;
}

void SetGraphConstTensorFile(StaticGraph* graph, const std::string& fname)
{
    graph->const_tensor_file = fname;
}

// if attr_name exist, return false
bool AddGraphAttr(StaticGraph* graph, const std::string& attr_name, any&& value)
{
    Attribute& attrs = graph->attrs;

    if(attrs.ExistAttr(attr_name))
        return false;

    attrs.SetAttr(attr_name, std::move(value));

    return true;
}

StaticNode* FindNode(StaticGraph* graph, const std::string& node_name)
{
    int node_number = graph->node_list.size();

    for(int i = 0; i < node_number; i++)
    {
        StaticNode* node = graph->node_list[i].get();

        if(node->name == node_name)
            return node;
    }

    return nullptr;
}

StaticTensor* FindTensor(StaticGraph* graph, const std::string& tensor_name)
{
    int tensor_number = graph->tensor_list.size();

    for(int i = 0; i < tensor_number; i++)
    {
        StaticTensor* tensor = graph->tensor_list[i].get();

        if(tensor->name == tensor_name)
            return tensor;
    }

    return nullptr;
}

StaticTensor* FindConstTensor(StaticGraph* graph, const std::string& tensor_name)
{
    auto ir = graph->const_tensor_map.begin();
    auto end = graph->const_tensor_map.end();

    while(ir != end)
    {
        StaticTensor* tensor = ir->second.get();

        if(tensor->name == tensor_name)
            return tensor;

        ir++;
    }

    return nullptr;
}

void AddGraphInputNode(StaticGraph* graph, StaticNode* node)
{
    graph->input_node_list.push_back(node->index);
}

void AddGraphOutputNode(StaticGraph* graph, StaticNode* node)
{
    graph->output_node_list.push_back(node->index);
}

bool CheckGraphIntegraityByEdge(StaticGraph* graph)
{
    /*go through all tensors and check if the tensor's producer and consumer's info are correct */
    StaticTensor* tensor;

    for(unsigned int i = 0; i < graph->tensor_list.size(); i++)
    {
        tensor = graph->tensor_list[i].get();

        /* check index */
        if(tensor->index != ( int )i)
        {
            LOG_ERROR() << "tensor: " << tensor->name << " index mismatch: real " << i << " record " << tensor->index
                        << "\n";
            return false;
        }

        /* check producer */

        NodeSynapse node_entry = tensor->producer;

        StaticNode* node = graph->node_list[node_entry.node_index].get();

        if(node->index != node_entry.node_index)
        {
            LOG_ERROR() << "node: " << node->name << " index mismatch: real " << node_entry.node_index;
            LOG_ERROR() << " record " << node->index << "\n";
            return false;
        }

        /* check producer */

        if(node_entry.entry_index >= ( int )node->output_tensor_list.size() ||
           node->output_tensor_list[node_entry.entry_index] != tensor->index)
        {
            LOG_ERROR() << "Producer mismatch: tensor " << tensor->name << " node " << node->name << "\n";
            return false;
        }

        /* if the node has no input tensor, the op must be const or input */
        if(node->input_tensor_list.size() == 0)
        {
            StaticOp* op = node->op.get();

            if(op->name != "Const" && op->name != "InputOp")
            {
                LOG_ERROR() << "node " << node->name << " has no input while op is: " << op->name << "\n";
                return false;
            }
        }

        /* if the tensor has no consumer, the node must be an output node of graph */

        if(tensor->consumer.size() == 0)
        {
            int found = 0;

            for(unsigned int n = 0; n < graph->output_node_list.size(); n++)
            {
                if(graph->output_node_list[n] == node->index)
                {
                    found = 1;
                    break;
                }
            }

            if(!found)
            {
                LOG_DEBUG() << "tensor: " << tensor->name << " created by node: " << node->name << " is not consumed\n";
                LOG_DEBUG() << "add the node: " << node->name << " into output list\n";
                graph->output_node_list.push_back(node->index);
                // return false; //do not look this as an error....
            }
        }

        /* check consumer */

        for(unsigned k = 0; k < tensor->consumer.size(); k++)
        {
            node_entry = tensor->consumer[k];

            node = graph->node_list[node_entry.node_index].get();

            if(node->index != node_entry.node_index)
            {
                LOG_ERROR() << "node: " << node->name << " index mismatch: real " << node_entry.node_index;
                LOG_ERROR() << " record " << node->index << "\n";
                return false;
            }

            if(node_entry.entry_index >= ( int )node->input_tensor_list.size() ||
               node->input_tensor_list[node_entry.entry_index] != tensor->index)
            {
                LOG_ERROR() << "Consumer mismatch: tensor " << tensor->name << " node " << node->name << "\n";
                return false;
            }
        }
    }

    /* sort the output node list in ascending order of index */
    std::sort(graph->output_node_list.begin(), graph->output_node_list.end(), std::less<int>());

    return true;
}

bool CheckGraphIntegraityByNode(StaticGraph* graph)
{
    /* go through all nodes */

    return true;
}

bool CheckGraphIntegraity(StaticGraph* graph)
{
    return CheckGraphIntegraityByEdge(graph) && CheckGraphIntegraityByNode(graph);
}

StaticNode* CreateStaticNode(StaticGraph* graph, const std::string& node_name)
{
    /* the most important thing is to set the node idx */

    int node_idx = graph->node_list.size();
    StaticNodePtr node_ptr(new StaticNode());

    node_ptr->name = node_name;
    node_ptr->index = node_idx;

    graph->node_list.emplace_back(node_ptr);

    return node_ptr.get();
}

const std::string& GetNodeName(StaticNode* node)
{
    return node->name;
}

int AddNodeInputTensor(StaticNode* node, StaticTensor* tensor)
{
    int input_idx = node->input_tensor_list.size();

    node->input_tensor_list.push_back(tensor->index);

    AddTensorConsumer(tensor, node, input_idx);

    return input_idx;
}

int AddNodeOutputTensor(StaticNode* node, StaticTensor* tensor)
{
    int out_idx = node->output_tensor_list.size();
    node->output_tensor_list.push_back(tensor->index);

    SetTensorProducer(tensor, node, out_idx);

    return out_idx;
}

void SetNodeOp(StaticNode* node, StaticOp* op)
{
    node->op.reset(op);
}

StaticOp* GetNodeOp(StaticNode* node)
{
    return node->op.get();
}

StaticTensor* GetNodeOutputTensor(StaticGraph* graph, StaticNode* node, int idx)
{
    int tensor_idx = node->output_tensor_list[idx];

    return graph->tensor_list[tensor_idx].get();
}

StaticTensor* GetNodeInputTensor(StaticGraph* graph, StaticNode* node, int idx)
{
    int tensor_idx = node->input_tensor_list[idx];

    return graph->tensor_list[tensor_idx].get();
}

StaticOp* CreateStaticOp(StaticGraph* graph, const std::string& op_name)
{
    StaticOp* op = new StaticOp();
    op->name = op_name;
    return op;
}

void SetOperatorDynamicShape(StaticOp* op)
{
    op->dynamic_shape = true;
}

void SetOperatorParam(StaticOp* op, any&& param)
{
    op->param = std::move(param);
}

void AddOperatorAttr(StaticOp* op, const std::string& attr_name, any&& val)
{
    op->attrs.SetAttr(attr_name, std::move(val));
}

any& GetOperatorParam(StaticOp* op)
{
    return op->param;
}

StaticTensor* CreateStaticTensor(StaticGraph* graph, const std::string& name)
{
    int tensor_idx = graph->tensor_list.size();

    StaticTensorPtr tensor_ptr(new StaticTensor());

    tensor_ptr->index = tensor_idx;
    tensor_ptr->name = name;
    tensor_ptr->type = kVarTensor;
    graph->tensor_list.push_back(tensor_ptr);

    return tensor_ptr.get();
}

void SetTensorDim(StaticTensor* tensor, const std::vector<int>& dims)
{
    tensor->dims = dims;
}

const std::vector<int>& GetTensorDim(StaticTensor* tensor)
{
    return tensor->dims;
}

void SetTensorDataType(StaticTensor* tensor, int data_type)
{
    tensor->data_type = data_type;
}

void SetTensorType(StaticTensor* tensor, int type)
{
    tensor->type = type;
}

int SetTensorSize(StaticTensor* tensor, int size)
{
    tensor->mem_size = size;
    return 0;
}

void SetTensorProducer(StaticTensor* tensor, StaticNode* node, int idx)
{
    tensor->producer.node_index = node->index;
    tensor->producer.entry_index = idx;
}

void AddTensorConsumer(StaticTensor* tensor, StaticNode* node, int idx)
{
    NodeSynapse entry;
    entry.node_index = node->index;
    entry.entry_index = idx;
    tensor->consumer.emplace_back(entry);
}

StaticNode* GetTensorProducer(StaticGraph* graph, StaticTensor* tensor)
{
    size_t node_idx = tensor->producer.node_index;

    if(node_idx >= graph->node_list.size())
        return nullptr;

    return graph->node_list[node_idx].get();
}

StaticTensor* CreateStaticConstTensor(StaticGraph* graph, const std::string& name)
{
    int tensor_idx = graph->tensor_list.size();

    StaticTensorPtr tensor_ptr(new StaticConstTensor());

    tensor_ptr->index = tensor_idx;
    tensor_ptr->name = name;
    tensor_ptr->type = kConstTensor;

    graph->tensor_list.push_back(tensor_ptr);

    graph->const_tensor_map[name] = tensor_ptr;

    return dynamic_cast<StaticTensor*>(tensor_ptr.get());
}

void* GetConstTensorBuffer(StaticTensor* tensor)
{
    StaticConstTensor* const_tensor = dynamic_cast<StaticConstTensor*>(tensor);
    return const_tensor->mem_addr;
}

void SetConstTensorBuffer(StaticTensor* tensor, void* addr)
{
    StaticConstTensor* const_tensor = dynamic_cast<StaticConstTensor*>(tensor);
    const_tensor->mem_addr = addr;
}

void SetConstTensorFileLocation(StaticTensor* tensor, int offset, int file_size)
{
    StaticConstTensor* const_tensor = dynamic_cast<StaticConstTensor*>(tensor);

    const_tensor->file_offset = offset;
    const_tensor->file_size = file_size;
}

const std::string& GetTensorName(StaticTensor* tensor)
{
    return tensor->name;
}

/* the dump family */

void DumpStaticNode(StaticGraph* graph, StaticNode* node, std::ostream& os)
{
    os << " " << node->name << " idx: " << node->index;
    os << " input: " << node->input_tensor_list.size() << " output: " << node->output_tensor_list.size();
    os << "\top: " << node->op->name << "\n";

    for(unsigned int i = 0; i < node->input_tensor_list.size(); i++)
    {
        int index = node->input_tensor_list[i];
        StaticTensorPtr tensor_ptr = graph->tensor_list[index];

        os << "\tI" << i << ": " << tensor_ptr->name << " type: " << tensor_ptr->type;
        os << " data_type: " << tensor_ptr->data_type << " ";

        if(tensor_ptr->dims.size())
        {
            os << "\tshape: [";

            for(unsigned int k = 0; k < tensor_ptr->dims.size(); k++)
                os << " " << tensor_ptr->dims[k];

            os << "]";
        }

        os << "\n";
    }

    for(unsigned int i = 0; i < node->output_tensor_list.size(); i++)
    {
        int index = node->output_tensor_list[i];
        StaticTensorPtr tensor_ptr = graph->tensor_list[index];

        os << "\tO" << i << ": " << tensor_ptr->name << " type: " << tensor_ptr->type;
        os << " data_type: " << tensor_ptr->data_type << " ";

        if(tensor_ptr->dims.size())
        {
            os << "\tshape: [";

            for(unsigned int k = 0; k < tensor_ptr->dims.size(); k++)
                os << " " << tensor_ptr->dims[k];

            os << "]";
        }
        os << "\n";
    }
}

void DumpStaticGraph(StaticGraph* graph)
{
    std::ostream& os = std::cout;

    os << "content of graph: " << graph->model_name << "\n";
    os << "graph identity\t\tdoman: " << graph->domain << " name: " << graph->name << " version: " << graph->version
       << "\n";
    os << "graph source format: " << graph->source_format << " source: " << graph->source << "\n";
    os << "Input node: " << graph->input_node_list.size() << "\n";

    for(unsigned int i = 0; i < graph->input_node_list.size(); i++)
    {
        int node_idx = graph->input_node_list[i];
        StaticNodePtr node_ptr = graph->node_list[node_idx];

        os << "\tI" << i << ": " << node_ptr->name << "\n";
    }

    os << "Output node: " << graph->output_node_list.size() << "\n";

    for(unsigned int i = 0; i < graph->output_node_list.size(); i++)
    {
        int node_idx = graph->output_node_list[i];
        StaticNodePtr node_ptr = graph->node_list[node_idx];

        os << "\tO" << i << ": " << node_ptr->name << "\n";
    }

    os << "Node list: " << graph->node_list.size() << "\n";

    for(unsigned int i = 0; i < graph->node_list.size(); i++)
    {
        os << i << ": ";

        StaticNodePtr node_ptr = graph->node_list[i];

        DumpStaticNode(graph, node_ptr.get(), os);

        os << "\n";
    }
}

}    // namespace TEngine
