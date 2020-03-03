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
#include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <string.h>
#include <string>
#include "tengine_c_api.h"
using namespace std;
std::string model_name = "./models/tf_conv_test.pb";

bool read_data(std::string file_path, float* data)
{
    fstream in_file;
    in_file.open(file_path, ios::in);
    string buff;
    if(!in_file.is_open())
    {
        std::cout << "Open File Failed\n";
        return false;
    }
    int i = 0;
    while(getline(in_file, buff))
    {
        char* s_input = ( char* )buff.c_str();
        const char* split = ",";
        char* p = strtok(s_input, split);
        float a;
        while(p != NULL)
        {
            a = atof(p);
            data[i++] = a;
            p = strtok(NULL, split);
        }
    }
    in_file.close();

    return true;
}

int main(int argc, char* argv[])
{
    init_tengine();

    graph_t graph = create_graph(nullptr, "tensorflow", model_name.c_str());
    // set_graph_layout(graph,TENGINE_LAYOUT_NHWC);
    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return 1;
    }

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    int dim[4] = {1, 224, 224, 3};

    set_tensor_shape(input_tensor, dim, 4);

    int input_size = get_tensor_buffer_size(input_tensor);

    float* input_data = ( float* )malloc(input_size);

    read_data("./tests/data/tf_conv_test_input.txt", input_data);

    set_tensor_buffer(input_tensor, input_data, input_size);

    prerun_graph(graph);

    run_graph(graph, 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

    float* out_data = ( float* )get_tensor_buffer(output_tensor);

    for(int i = 0; i < 10; ++i)
    {
        printf("%f\n", out_data[i]);
    }

    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
