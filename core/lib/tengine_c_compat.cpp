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
#include <string.h>

#include "tengine_c_compat.h"
#include "tengine_c_helper.hpp"
#include "exec_context.hpp"

#include "data_layout.hpp"
#include "graph_executor.hpp"

using namespace TEngine;

int init_tengine_library(void)
{
    return init_tengine();
}

void release_tengine_library(void)
{
    return release_tengine();
}

int load_model(const char* model_name, const char* model_format, const char* fname, ...)
{
    context_t exec_context = ExecContext::GetDefaultContext();

    va_list argp;
    va_start(argp, fname);

    return vload_file_model(exec_context, model_name, model_format, fname, argp);
}

int save_model(graph_t graph, const char* file_format, const char* fname, ...)
{
    va_list argp;
    va_start(argp, fname);

    return save_graph_internal(graph, file_format, fname, argp);
}

graph_t create_runtime_graph(const char* graph_name, const char* model_name, context_t context)
{
    if(context == nullptr)
        context = ExecContext::GetDefaultContext();

    return create_graph_in_context(context, graph_name, model_name);
}

int check_graph_valid(graph_t graph)
{
    if(graph == nullptr)
        return 0;
    return 1;
}

int check_tensor_valid(tensor_t tensor)
{
    if(tensor == nullptr)
        return 0;

    return 1;
}

void put_graph_tensor(tensor_t tensor)
{
    release_graph_tensor(tensor);
}

void put_graph_node(node_t node)
{
    release_graph_node(node);
}

int set_graph_config(graph_t graph, const char* name, void* val, int size)
{
    return set_graph_attr(graph, name, val, size);
}

int get_node_param_int(node_t node, const char* param_name, int* param_val)
{
    return get_node_attr_int(node, param_name, param_val);
}

int get_node_param_float(node_t node, const char* param_name, float* param_val)
{
    return get_node_attr_float(node, param_name, param_val);
}

int get_node_param_pointer(node_t node, const char* param_name, void* param_val)
{
    return get_node_attr_pointer(node, param_name, param_val);
}

int get_node_param_generic(node_t node, const char* param_name, const void* type_info, void* param_val, int size)
{
    return get_node_attr_generic(node, param_name, type_info, param_val, size);
}

int infer_shape(graph_t graph)
{
    GraphExecutor* executor = static_cast<GraphExecutor*>(graph);

    if(!executor->InferShape())
        return -1;
    return 0;
}

int set_tensor_layout(tensor_t tensor, const char* layout)
{
    std::string real_layout = layout;
    const DataLayout* data_layout = DataLayout::GetLayout(real_layout);
    if(data_layout == nullptr)
        return -1;
    Tensor* real_tensor = reinterpret_cast<Tensor*>(tensor);

    TShape shape = real_tensor->GetShape();

    shape.SetDataLayout(real_layout);
    real_tensor->Reshape(shape);

    return 0;
}

int get_tensor_layout(tensor_t tensor, char* layout)
{
    Tensor* real_tensor = reinterpret_cast<Tensor*>(tensor);

    TShape shape = real_tensor->GetShape();
    const std::string& data_layout = shape.GetDataLayout();
    if(data_layout.empty())
        return -1;
    int len = strlen(data_layout.c_str());
    memcpy(layout, data_layout.c_str(), len);

    return 0;
}
