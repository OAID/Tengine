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
#ifndef __TENGINE_C_HELPER_HPP__
#define __TENGINE_C_HELPER_HPP__

#include <stdarg.h>
#include <vector>

extern "C" {

int node_add_attr(node_t node, const char* attr_name, const char* type_name, int size);

int node_get_attr_generic(void* node, const char* param_name, const char* type_name, void* param_val, int param_size);
int node_set_attr_generic(void* node, const char* param_name, const char* type_name, const void* param_val,
                          int param_size);

void set_cpu_list(const char* cpu_list_str);

int vload_file_model(context_t exec_context, const char* model_name, const char* model_format, const char* fname,
                     va_list argp);
int vload_mem_model(context_t exec_context, const char* model_name, const char* model_format, const void* addr,
                    int mem_size, va_list argp);

graph_t create_graph_in_context(context_t exec_context, const char* graph_name, const char* model_name);

int save_graph_internal(graph_t graph, const char* file_format, const char* fname, va_list argp);

int get_model_format(graph_t graph);

const char* get_model_name(graph_t graph);

const char* get_tengine_hcl_version();
}

namespace TEngine {

class GraphExecutor;

int InitAllPlugin(void);

GraphExecutor* do_merge_graph(std::vector<GraphExecutor*>& exec_list);

}    // namespace TEngine

#endif
