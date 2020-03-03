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
#include <vector>
#include <functional>

#include "onnx_serializer.hpp"

#include "share_lib_parser.hpp"
#include "graph.hpp"

const char* model_file = "./tests/data/sqz.onnx.model";

using namespace TEngine;

int main(void)
{
    ShareLibParser p0("./build/operator/liboperator.so");
    p0.ExcecuteFunc<int()>("tengine_plugin_init");

    ShareLibParser p1("./build/serializer/libserializer.so");
    p1.ExcecuteFunc<int()>("tengine_plugin_init");

    SerializerPtr p_onnx;

    if(!SerializerManager::SafeGet("onnx", p_onnx))
    {
        std::cout << "No onnx registered in object manager\n";
        return 1;
    }

    StaticGraph* graph = CreateStaticGraph("test");

    std::vector<std::string> flist;

    flist.push_back(model_file);

    if(!p_onnx->LoadModel(flist, graph))
    {
        std::cout << "Load model failed\n";
        return 1;
    }

    std::cout << "Load model successfully\n";

    DumpStaticGraph(graph);

    if(CheckGraphIntegraity(graph))
        std::cout << "check passed\n";

    StaticGraphPtr graph_ptr(graph);

    Graph* runtime_graph = Graph::CreateFromStatic(graph_ptr);

    if(runtime_graph == nullptr)
    {
        std::cout << "Create runtime graph failed\n";
        return 1;
    }

    // runtime_graph->RemoveNoChildTensor();

    std::cout << "RUNTIME GRAPH\n";
    runtime_graph->DumpGraph();

    SerializerManager::Remove("onnx");

    std::cout << "ALL TEST DONE\n";

    return 0;
}
