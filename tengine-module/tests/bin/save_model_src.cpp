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
#include <time.h>

#include "tengine_c_api.h"

const char* text_file = "./models/sqz.prototxt";
const char* model_file = "./models/squeezenet_v1.1.caffemodel";

int main(int argc, char* argv[])
{
    std::string model_name = "squeeze_net";

    init_tengine();

    if(request_tengine_version("0.9") < 0)
        return 1;

    graph_t graph = create_graph(nullptr, "caffe", text_file, model_file);

    if(graph == nullptr)
    {
        std::cerr << "Create graph failed\n";
        std::cerr << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }

    /*
       src_tm is the serializer name
       squeeze_net is the model name,
       which is used when load source model
    */

    if(save_graph(graph, "src_tm", model_name.c_str()) < 0)
    {
        std::cerr << "Save graph failed\n";
        return 1;
    }

    /* free resource */

    destroy_graph(graph);

    release_tengine();

    return 0;
}
