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
 * Copyright (c) 2019, Open AI Lab
 * Author: zpluo@openailab.com
 */
#include <unistd.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <time.h>

#include "tengine_c_api.h"

std::string model_name1 = "./models/LSTM/Fused_Neural_Net-symbol.json";
std::string model_name2 = "./models/LSTM/Fused_Neural_Net-0100.params";

int main(int argc, char* argv[])
{
    int steps = 1;
    int res;

    while((res = getopt(argc, argv, "n:")) != -1)
    {
        switch(res)
        {
            case 'n':
                steps = strtoul(optarg, NULL, 10);
                break;
            default:
                break;
        }
    }

    init_tengine();

    graph_t graph = create_graph(nullptr, "mxnet", model_name1.c_str(), model_name2.c_str());

    // set_graph_layout(graph,TENGINE_LAYOUT_NCHW);
    // dump_graph(graph);

    if(graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return 1;
    }

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);

    int dim[3] = {steps, 28, 28};

    set_tensor_shape(input_tensor, dim, 3);

    int input_size = get_tensor_buffer_size(input_tensor);
    float* input_data = ( float* )malloc(input_size);

    for(unsigned int i = 0; i < input_size / sizeof(float); i++)
        input_data[i] = 45;

    set_tensor_buffer(input_tensor, input_data, input_size);

    // std::cout<<"intensr "<<input_tensor<<" buffsize"<<get_tensor_buffer_size(input_tensor)<<
    // "  menptr: "<<get_tensor_buffer(input_tensor)<<"\n";
    prerun_graph(graph);
    // std::cout<<"after pre run intensr "<<input_tensor<<" buffsize"<<get_tensor_buffer_size(input_tensor)<<
    // "  menptr: "<<get_tensor_buffer(input_tensor)<<"\n";

    // dump_graph(graph);

    run_graph(graph, 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

    std::cout << "final hidden state after " << steps << " steps:\n";

    float* output_data = ( float* )get_tensor_buffer(output_tensor);
    int output_size = get_tensor_buffer_size(output_tensor);

    for(unsigned int i = 0; i < output_size / sizeof(float); i++)
    {
        if((i % 4) == 0)
            printf("\n%d:\t", i);
        printf("  %.12g", output_data[i]);
    }

    printf("\n");

    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);
    postrun_graph(graph);
    destroy_graph(graph);
    free(input_data);

    std::cout << "ALL TEST DONE\n";

    release_tengine();
    return 0;
}
