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
#include "share_lib_parser.hpp"
#include "caffe_serializer.hpp"
#include "graph.hpp"
#include "graph_executor.hpp"

const char* model_file = "./tests/data/squeezenet_v1.1.caffemodel";

using namespace TEngine;

int main(void)
{
    ShareLibParser p0("./build/operator/liboperator.so");
    p0.ExcecuteFunc<int()>("tengine_plugin_init");

    ShareLibParser p1("./build/serializer/libserializer.so");
    p1.ExcecuteFunc<int()>("tengine_plugin_init");

    SerializerPtr p_caffe;

    if(!SerializerManager::SafeGet("caffe_single", p_caffe))
    {
        std::cout << "No caffe registered in object manager\n";
        return 1;
    }

    StaticGraph* graph = CreateStaticGraph("test");

    std::vector<std::string> flist;

    flist.push_back(model_file);

    if(!p_caffe->LoadModel(flist, graph))
    {
        std::cout << "Load model failed\n";
        return 1;
    }

    std::cout << "Load model successfully\n";

    DumpStaticGraph(graph);

    if(CheckGraphIntegraity(graph))
        std::cout << "check passed\n";

    StaticGraphPtr graph_ptr(graph);

    StaticGraphManager::SafeAdd(graph->name, graph_ptr);

    GraphExecutor executor;

    if(!executor.CreateGraph("runtime", graph->name))
    {
        std::cout << "create graph from static graph: " << graph->name << "failed\n";

        return 1;
    }

    Graph* runtime_graph = executor.GetGraph();

    Node* node = runtime_graph->FindNode("accuracy");

    runtime_graph->RemoveNode(node);

    node = runtime_graph->FindNode("accuracy_top5");

    runtime_graph->RemoveNode(node);

    Tensor* dummy_tensor = runtime_graph->FindTensor("label_data_1_split_0");
    runtime_graph->RemoveTensor(dummy_tensor);

    runtime_graph->SanitizeGraph();

    const std::string& node_name = executor.GetGraphInputNodeName(0);

    const std::string& tensor_name = executor.GetNodeOutputTensor(node_name, 0);

    Tensor* tensor = executor.FindTensor(tensor_name);

    std::vector<int> dim = {1, 3, 227, 227};

    TShape& shape = tensor->GetShape();

    shape.SetDim(dim);

    if(!executor.InferShape())
    {
        std::cout << "InferShape failed\n";
        return 1;
    }

    runtime_graph->DumpGraph();

    SerializerManager::Remove("caffe_single");

    std::cout << "ALL TEST DONE\n";
    return 0;
}
