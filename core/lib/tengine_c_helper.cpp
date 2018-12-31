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
 * Copyright (c) 2018, Open AI Lab
 * Author: haitao@openailab.com
 */
#include <stdarg.h>
#include <assert.h>
#include <string.h>

#include <map>
#include <set>

#include "cpu_device.h"
#include "tengine_c_api.h"
#include "tengine_c_compat.h"
#include "tengine_c_helper.hpp"

#include "data_layout.hpp"
#include "exec_context.hpp"
#include "graph_executor.hpp"
#include "tengine_errno.hpp"
#include "static_graph_interface.hpp"
#include "serializer.hpp"

namespace TEngine {

extern int NodeSetParamGeneric(void* node, const char* param_name, const void* type_info, const void* param_val,
                               int size);
extern int NodeGetParamGeneric(void* node, const char* param_name, const void* type_info, void* param_val, int size);
extern int NodeAddParamGeneric(void* node, const char* param_name, const void* type_info, int size);
}    // namespace TEngine

using namespace TEngine;

void __attribute__((constructor)) first_init(void)
{
    NamedData<DataLayout>::InitPredefinedData();
}

void set_cpu_list(const char* cpu_list_str)
{
    char* copy_str = strdup(cpu_list_str);

    std::vector<int> cpu_list;

    char* p = strtok(copy_str, ",");

    while(p)
    {
        int cpu_id = strtoul(p, NULL, 10);
        cpu_list.push_back(cpu_id);
        p = strtok(NULL, ",");
    }

    int* int_buf = cpu_list.data();

    set_working_cpu(int_buf, cpu_list.size());

    free(copy_str);
}

int remove_model(const char* model_name)
{
    int ret = 0;

    StaticGraphManager::Get();

    if(StaticGraphManager::Find(model_name))
        StaticGraphManager::Remove(model_name);
    else
    {
        ret = -1;
        set_tengine_errno(ENOENT);
    }

    StaticGraphManager::Put();

    return ret;
}

int destroy_runtime_graph(graph_t graph)
{
    GraphExecutor* executor = reinterpret_cast<GraphExecutor*>(graph);

    delete executor;

    return 0;
}

int dump_model(const char* model_name)
{
    StaticGraphPtr graph_ptr;

    if(StaticGraphManager::SafeGet(model_name, graph_ptr))
    {
        DumpStaticGraph(graph_ptr.get());
        return 0;
    }

    set_tengine_errno(ENOENT);

    return -1;
}

int node_get_attr_generic(void* node, const char* param_name, const void* type_info, void* param_val, int param_size)
{
    return NodeGetParamGeneric(node, param_name, type_info, param_val, param_size);
}

int node_set_attr_generic(void* node, const char* param_name, const void* type_info, const void* param_val,
                          int param_size)
{
    return NodeSetParamGeneric(node, param_name, type_info, param_val, param_size);
}

int node_add_attr(void* node, const char* param_name, const void* type_info, int param_size)
{
    return NodeAddParamGeneric(node, param_name, type_info, param_size);
}

static int real_vload_model(context_t exec_context, const char* model_name, const char* model_format, const void* addr,
                            int mem_size, va_list argp)
{
    SerializerPtr serializer;

    if(!SerializerManager::SafeGet(model_format, serializer))
    {
        LOG_ERROR() << "Get serializer failed, unknown model format: " << model_format << "\n";
        set_tengine_errno(EINVAL);
        return -1;
    }

    StaticGraph* static_graph = CreateStaticGraph(model_name);

    static_graph->exec_context = exec_context;

    int saved_file_number = serializer->GetFileNum();

    if(mem_size == 0)    // file mode
    {
        std::vector<std::string> file_list;
        file_list.push_back(( const char* )addr);

        for(int i = 1; i < saved_file_number; i++)
        {
            const char* file = va_arg(argp, const char*);
            file_list.emplace_back(file);
        }

        if(!serializer->LoadModel(file_list, static_graph) || !CheckGraphIntegraity(static_graph))
        {
            delete static_graph;
            return -1;
        }
    }
    else
    {
        std::vector<const void*> addr_list;
        std::vector<int> size_list;

        addr_list.push_back(addr);
        size_list.push_back(mem_size);

        for(int i = 1; i < saved_file_number; i++)
        {
            addr = va_arg(argp, const void*);
            mem_size = va_arg(argp, int);

            addr_list.push_back(addr);
            size_list.push_back(mem_size);
        }

        if(!serializer->LoadModel(addr_list, size_list, static_graph) || !CheckGraphIntegraity(static_graph))
        {
            delete static_graph;
            return -1;
        }
    }

    va_end(argp);

    if(!StaticGraphManager::Add(std::string(model_name), StaticGraphPtr(static_graph)))
    {
        XLOG_ERROR() << "replicated model name detected: " << model_name << " should not happen\n";
        set_tengine_errno(EBADSLT);
        return -1;
    }

    return 0;
}

static int vload_model(context_t exec_context, const char* model_name, const char* model_format, const void* addr,
                       int mem_size, va_list argp)
{
    StaticGraphManager::Get();

    if(StaticGraphManager::Find(model_name))
    {
        set_tengine_errno(EEXIST);
        StaticGraphManager::Put();
        return -1;
    }

    int ret = real_vload_model(exec_context, model_name, model_format, addr, mem_size, argp);

    StaticGraphManager::Put();

    return ret;
}

int vload_file_model(context_t exec_context, const char* model_name, const char* model_format, const char* fname,
                     va_list argp)
{
    return vload_model(exec_context, model_name, model_format, fname, 0, argp);
}

int vload_mem_model(context_t exec_context, const char* model_name, const char* model_format, const void* addr,
                    int mem_size, va_list argp)
{
    return vload_model(exec_context, model_name, model_format, addr, mem_size, argp);
}

int save_graph_internal(graph_t graph, const char* model_format, const char* fname, va_list argp)
{
    /* Get the serializer according to model_format */
    SerializerPtr serializer;
    if(!SerializerManager::SafeGet(model_format, serializer))
    {
        LOG_ERROR() << "unknown model format: " << model_format << "\n";
        set_tengine_errno(ENOENT);
        return -1;
    }

    /* Create file list */
    std::vector<std::string> file_list;
    file_list.push_back(fname);

    for(unsigned int i = 1; i < serializer->GetFileNum(); i++)
    {
        const char* file = va_arg(argp, const char*);
        file_list.emplace_back(file);
    }
    va_end(argp);

    /* Get runtime graph pointer */
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);
    Graph* g = executor->GetGraph();

    /* Save the graph to the files */
    if(!serializer->SaveModel(file_list, g))
        return -1;

    return 0;
}

graph_t create_graph_in_context(context_t exec_context, const char* graph_name, const char* model_name)
{
    GraphExecutor* executor = new GraphExecutor();

    if(!executor->CreateGraph(exec_context, graph_name, model_name))
    {
        delete executor;
        return nullptr;
    }

    return executor;
}

const char* get_model_name(graph_t graph)
{
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);

    if(executor->GetModelName().empty())
        return nullptr;
    else
        return executor->GetModelName().c_str();
}

extern void operator_plugin_init(void);
extern void serializer_plugin_init(void);
extern void executor_plugin_init(void);
extern void driver_plugin_init(void);

namespace TEngine {

void InitAllPlugin(void)
{
    operator_plugin_init();
    serializer_plugin_init();
    executor_plugin_init();
    driver_plugin_init();
}

struct tensor_entry
{
    Tensor* tensor;
    Node* node;
    Graph* graph;
    GraphExecutor* executor;
    bool consumed;
};

void replace_input_tensor(Tensor* new_tensor, Tensor* old_tensor)
{
    for(unsigned int i = 0; i < old_tensor->consumer.size(); i++)
    {
        NodePort* port = old_tensor->consumer[i];

        port->tensor = new_tensor;

        new_tensor->consumer.push_back(port);
    }

    old_tensor->consumer.clear();
}

static int check_graph_inputs(Graph* graph, std::map<std::string, tensor_entry>& out_map)
{
    int input_node_num = graph->input_nodes.size();
    std::vector<Node*> remove_node;
    std::vector<Tensor*> remove_tensor;

    for(int j = 0; j < input_node_num; j++)
    {
        Node* in_node = graph->input_nodes[j];

        if(in_node->GetInputNum() == 0)
        {
            Tensor* in_tensor = in_node->GetOutputTensor(0);

            auto ir = out_map.find(in_tensor->GetName());

            if(ir != out_map.end())
            {
                replace_input_tensor(ir->second.tensor, in_tensor);

                assert(in_tensor->consumer.size() == 0);

                remove_node.push_back(in_node);
                ir->second.consumed = true;
            }
        }
        else
        {
            for(unsigned int l = 0; l < in_node->GetInputNum(); l++)
            {
                Tensor* in_tensor = in_node->GetInputTensor(l);

                if(in_tensor->producer == nullptr)
                {
                    auto ir = out_map.find(in_tensor->GetName());

                    if(ir != out_map.end())
                    {
                        replace_input_tensor(ir->second.tensor, in_tensor);

                        assert(in_tensor->consumer.size() == 0);

                        remove_tensor.push_back(in_tensor);
                        ir->second.consumed = true;
                    }
                }
            }
        }
    }

    for(unsigned int i = 0; i < remove_node.size(); i++)
    {
        graph->RemoveNode(remove_node[i]);
    }

    for(unsigned int i = 0; i < remove_tensor.size(); i++)
    {
        graph->RemoveTensor(remove_tensor[i]);
    }

    return (remove_node.size() + remove_tensor.size());
}

GraphExecutor* do_merge_graph(std::vector<GraphExecutor*>& exec_list)
{
    std::map<std::string, tensor_entry> out_map;

    /* collect all output tensors */

    for(unsigned int i = 0; i < exec_list.size(); i++)
    {
        GraphExecutor* executor = exec_list[i];
        Graph* graph = executor->GetGraph();

        for(unsigned int n = 0; n < graph->output_nodes.size(); n++)
        {
            Node* node = graph->output_nodes[n];

            for(unsigned int m = 0; m < node->GetOutputNum(); m++)
            {
                Tensor* tensor = node->GetOutputTensor(m);

                if(tensor->consumer.size() == 0)
                {
                    tensor_entry entry;
                    entry.tensor = tensor;
                    entry.node = node;
                    entry.graph = graph;
                    entry.executor = executor;
                    entry.consumed = false;

                    if(out_map.count(tensor->GetName()))
                    {
                        LOG_ERROR() << "duplicated output tensor found: " << tensor->GetName() << "\n";
                        set_tengine_errno(EINVAL);
                        return nullptr;
                    }

                    out_map[tensor->GetName()] = entry;
                }
            }
        }
    }

    std::set<Graph*> keep_input_set;

    for(unsigned int i = 0; i < exec_list.size(); i++)
    {
        Graph* graph = exec_list[i]->GetGraph();

        int ret = check_graph_inputs(graph, out_map);

        if(ret == 0)
        {
            keep_input_set.insert(graph);
        }
    }

    GraphExecutor* first_executor = exec_list[0];

    std::string m_graph_name = first_executor->GetGraph()->GetName() + ".merged";
    ExecContext* m_context = ( ExecContext* )first_executor->GetExecAttr()->exec_context;

    GraphExecutor* m_executor = ( GraphExecutor* )create_graph_in_context(m_context, m_graph_name.c_str(), nullptr);
    Graph* m_graph = m_executor->GetGraph();

    for(unsigned int i = 0; i < exec_list.size(); i++)
    {
        Graph* graph = exec_list[i]->GetGraph();

        for(unsigned int j = 0; j < graph->seq_nodes.size(); j++)
        {
            Node* node = graph->seq_nodes[j];

            m_graph->AddNode(node, false);

            for(unsigned int l = 0; l < node->GetOutputNum(); l++)
            {
                Tensor* tensor = node->GetOutputTensor(l);

                m_graph->AddTensor(tensor, false);
            }
        }

        for(unsigned int j = 0; j < graph->input_nodes.size(); j++)
        {
            Node* node = graph->input_nodes[j];

            bool still_input_node = true;

            for(unsigned int l = 0; l < node->GetInputNum(); l++)
            {
                still_input_node = false;
                Tensor* tensor = node->GetInputTensor(l);

                if(tensor->producer == nullptr)
                {
                    still_input_node = true;
                    break;
                }
            }

            if(still_input_node)
                m_graph->input_nodes.push_back(node);
        }

        for(unsigned int j = 0; j < graph->output_nodes.size(); j++)
        {
            Node* node = graph->output_nodes[j];

            int output_tensor_number = 0;
            int output_tensor_in_map = 0;

            for(unsigned int l = 0; l < node->GetOutputNum(); l++)
            {
                Tensor* tensor = node->GetOutputTensor(l);

                if(tensor->consumer.size() == 0)
                {
                    output_tensor_number++;
                }

                if(out_map.count(tensor->GetName()))
                {
                    output_tensor_in_map++;
                }
            }

            if(output_tensor_number)
                m_graph->output_nodes.push_back(node);

            /* no output tensor has been consumed */
            if(output_tensor_number == output_tensor_in_map)
            {
                if(keep_input_set.count(graph))
                {
                    LOG_ERROR() << "graph: " << graph->GetName() << " is disconnected with others\n";
                    set_tengine_errno(EFAULT);
                    delete m_executor;
                    return nullptr;
                }
            }
        }
    }

    return m_executor;
}

}    // namespace TEngine
