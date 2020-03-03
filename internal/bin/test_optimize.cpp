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
#include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "tengine_c_api.h"

const char* text_file = "./models/mobilenet_deploy.prototxt";
const char* model_file = "./models/mobilenet.caffemodel";

int main(int argc, char* argv[])
{
    init_tengine();

    if(request_tengine_version("1.0") < 0)
        return -1;

    graph_t graph = create_graph(nullptr, "caffe", text_file, model_file);

    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return -1;
    }

    int optimize_only = 1;

    if(set_graph_attr(graph, "optimize_only", &optimize_only, sizeof(int)) < 0)
    {
        std::cerr << "set optimize only failed\n";
        return -1;
    }

    int ret_prerun = prerun_graph(graph);
    if(ret_prerun < 0)
    {
        std::printf("prerun failed\n");
        return -1;
    }

    dump_graph(graph);

    save_graph(graph, "tengine", "/tmp/mobilenet.tm");

    postrun_graph(graph);

    destroy_graph(graph);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
