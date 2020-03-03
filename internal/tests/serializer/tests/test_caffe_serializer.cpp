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

// const char * model_file="./tests/data/squeezenet_v1.1.caffemodel";
const char* model_file = "./tests/data/mtcnn_mdl/det1.caffemodel";
// TODO:"cannot find load function for operator: HDF5Data"

using namespace TEngine;

int main(void)
{
    ShareLibParser p0("./build/operator/liboperator.so");
    p0.ExcecuteFunc<int()>("tengine_plugin_init");

    ShareLibParser p1("./build/serializer/libserializer.so");
    p1.ExcecuteFunc<int()>("tengine_plugin_init");

    SerializerPtr p_caffe;

    SerializerManager::SafeGet("caffe_single", p_caffe);

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

    SerializerManager::Remove("caffe_single");

    std::cout << "release graph\n";

    DestroyStaticGraph(graph);

    std::cout << "ALL TEST DONE\n";
    return 0;
}
