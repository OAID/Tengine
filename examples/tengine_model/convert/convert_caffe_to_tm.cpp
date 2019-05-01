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
 * Author: jingyou@openailab.com
 */
#include <iostream>
#include <fstream>
#include <unistd.h>
#include "tengine_c_api.h"
#include "common.hpp"

int main(int argc, char* argv[])
{
    std::string proto_file;
    std::string model_file;
    std::string output_tmfile;

    int res;
    while((res = getopt(argc, argv, "p:m:o:h")) != -1)
    {
        switch(res)
        {
            case 'p':
                proto_file = optarg;
                break;
            case 'm':
                model_file = optarg;
                break;
            case 'o':
                output_tmfile = optarg;
                break;
            case 'h':
                std::cout << "[Usage]: " << argv[0] << " [-h] [-p proto_file] [-m model_file] [-o output_tmfile]\n";
                return 0;
            default:
                break;
        }
    }

    if(proto_file.empty())
    {
        std::cerr << "Please specify the -p option to indicate the input proto file.\n";
        return -1;
    }
    if(model_file.empty())
    {
        std::cerr << "Please specify the -m option to indicate the input model file.\n";
        return -1;
    }
    if(output_tmfile.empty())
    {
        std::cerr << "Please specify the -o option to indicate the output tengine model file.\n";
        return -1;
    }

    // check input files
    if(!check_file_exist(proto_file) || !check_file_exist(model_file))
        return -1;

    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return -1;

    // create graph
    graph_t graph = create_graph(nullptr, "caffe", proto_file.c_str(), model_file.c_str());
    if(graph == nullptr)
    {
        std::cerr << "Create graph failed.\n";
        std::cerr << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    const char* env = std::getenv("TM_NO_OPTIMIZE");
    if(env == nullptr)
    {
        // optimize graph
        int optimize_only = 1;
        if(set_graph_attr(graph, "optimize_only", &optimize_only, sizeof(int)) < 0)
        {
            std::cerr<<"set optimize only failed\n";
            return -1;
        }

        if(prerun_graph(graph) < 0)
        {
            std::cerr<<"prerun failed\n";
            return -1;
        }
    }

    // save the tengine model file
    if(save_graph(graph, "tengine", output_tmfile.c_str()) < 0)
    {
        std::cerr << "Create tengine model file failed.\n";
        std::cerr << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }
    std::cout << "Create tengine model file done: " << output_tmfile << "\n";

    destroy_graph(graph);
    release_tengine();
    return 0;
}
