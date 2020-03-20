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
 * Author: chunyinglv@openailab.com
 */
#include <iostream>
#include <iomanip>
#include <string>
#include "tengine_c_api.h"
#include <sys/time.h>

const char* model_file = "./models/minimumtest.pb";

int main(int argc, char* argv[])
{
    int channel = 3;

    if(argc < 3)
    {
        std::cout << "[Usage]: " << argv[0] << " <model> <img_h> [<img_w> <channel>]\n";
        return 1;
    }

    // init tengine
    init_tengine();
    if(request_tengine_version("0.9") < 0)
        return 1;

    // create graph
    //    std::string mdl_name = argv[1];
    graph_t graph = create_graph(nullptr, "tensorflow", model_file);    // mdl_name.c_str());
    if(graph == nullptr)
    {
        std::cout << "Create graph failed\n";
        std::cout << "errno: " << get_tengine_errno() << "\n";
        return 1;
    }
    dump_graph(graph);
    // input
    int img_h = atoi(argv[2]);
    int img_w = img_h;
    if(argc >= 4)
        img_w = atoi(argv[3]);

    if(argc >= 5)
        channel = atoi(argv[4]);
    printf("h=%d w=%d , c=%d \n", img_h, img_w, channel);
    int img_size = img_h * img_w * channel;

    float* input_data = ( float* )malloc(sizeof(float) * img_size);
    for(int i = 0; i < img_size; i++)
        input_data[i] = (i % 123) * 0.017;

    float* input_data2 = ( float* )malloc(sizeof(float) * img_size);
    for(int k = 0; k < img_size; k++)
        input_data2[k] = 0.547;

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    tensor_t input_tensor1 = get_graph_input_tensor(graph, 1, 0);
    int dims[] = {1, channel, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    set_tensor_shape(input_tensor1, dims, 4);

    prerun_graph(graph);

    // dump_graph(graph);

    set_tensor_buffer(input_tensor, input_data, sizeof(float) * img_size);
    set_tensor_buffer(input_tensor1, input_data2, sizeof(float) * img_size);

    run_graph(graph, 1);

    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

    float* output_data = ( float* )get_tensor_buffer(output_tensor);
    int output_size = get_tensor_buffer_size(output_tensor);

    for(unsigned int i = 0; i < output_size / sizeof(float); i++)
        printf("output %d: %g\n", i, output_data[i]);

    release_graph_tensor(input_tensor);
    release_graph_tensor(output_tensor);

    free(input_data);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    std::cout << "ALL TEST DONE\n";

    return 0;
}
